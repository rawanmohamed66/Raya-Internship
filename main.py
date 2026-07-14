import json
import os
import imaplib
import email
import smtplib
from email.header import decode_header
from email.mime.text import MIMEText
import chromadb

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
MODEL_NAME = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"


# ─────────────────────────────────────────────
# 0. Fetch real unread emails from Gmail
# ─────────────────────────────────────────────

def fetch_unread_emails(max_emails=1):
    """Connects to Gmail via IMAP and fetches the most recent unread email(s)."""
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
    mail.select("inbox")

    # search for unread emails only
    status, messages = mail.search(None, "UNSEEN")
    email_ids = messages[0].split()

    if not email_ids:
        mail.logout()
        return []

    # take only the LAST (most recent) unread email, ignore all others
    latest_id = email_ids[-1]

    fetched_emails = []
    status, msg_data = mail.fetch(latest_id, "(RFC822)")
    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)

    subject, encoding = decode_header(msg["Subject"])[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding or "utf-8")

    from_ = msg.get("From")

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode(errors="ignore")
                break
    else:
        body = msg.get_payload(decode=True).decode(errors="ignore")

    fetched_emails.append({
        "label": subject,
        "from": from_,
        "text": body.strip()
    })

    mail.logout()
    return fetched_emails


# ─────────────────────────────────────────────
# 0.5. Send the reply back to the customer
# ─────────────────────────────────────────────

def send_reply(to_address, subject, body):
    """Sends the final reply email back to the customer via Gmail SMTP."""
    msg = MIMEText(body)
    msg["Subject"] = f"Re: {subject}"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_address

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, to_address, msg.as_string())

    print(f"✅ Reply sent to {to_address}")


# ─────────────────────────────────────────────
# 1. Load knowledge base from JSON
# ─────────────────────────────────────────────

def load_knowledge_base(filepath="knowledge_base.json"):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


# ─────────────────────────────────────────────
# 2. Insert documents into ChromaDB
# ─────────────────────────────────────────────

def build_chroma_collection(knowledge_base):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()

    vectorstore = Chroma(
        collection_name="adib_knowledge",
        embedding_function=embedding_function,
        client=chroma_client,
    )

    documents = []
    for item in knowledge_base:
        doc = Document(
            page_content=item["content"],
            metadata={"title": item["title"], "id": item["id"]}
        )
        documents.append(doc)

    vectorstore.add_documents(documents)
    return vectorstore


# ─────────────────────────────────────────────
# 3. Extract fields using Groq function calling
# ─────────────────────────────────────────────

def extract_email_fields(email_text):
    client = OpenAI(api_key=GROQ_API_KEY, base_url=BASE_URL)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_customer_info",
                "description": "Extract key fields from a customer banking support email.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_name": {"type": "string", "description": "Full name of the customer. Empty string if not found."},
                        "request_type": {"type": "string", "description": "Type of request, e.g. card activation, cashback inquiry. Empty string if not found."},
                        "card_type": {"type": "string", "description": "Type of card mentioned, e.g. debit, credit, prepaid, cashback. Empty string if not found."},
                        "priority": {"type": "string", "description": "Priority level: high, medium, or low."}
                    },
                    "required": ["customer_name", "request_type", "card_type", "priority"]
                }
            }
        }
    ]

    messages = [
        {"role": "system", "content": "You are a banking support assistant. Extract information from customer emails."},
        {"role": "user", "content": f"Extract the required fields from this customer email:\n\n{email_text}"}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "extract_customer_info"}}
    )

    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)


# ─────────────────────────────────────────────
# 4. Search ChromaDB using request_type
# ─────────────────────────────────────────────

def search_knowledge(vectorstore, query, top_k=3):
    return vectorstore.similarity_search(query, k=top_k)


# ─────────────────────────────────────────────
# 4.5. Agents (Researcher + Reviewer)
# ─────────────────────────────────────────────

class SimpleAgent:
    def __init__(self, role, goal, backstory, llm):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm = llm

    def run(self, task_description):
        prompt = f"Role: {self.role}\nGoal: {self.goal}\nBackstory: {self.backstory}\n\nTask:\n{task_description}"
        response = self.llm.invoke(prompt)
        return response.content

agent_llm = ChatOpenAI(api_key=GROQ_API_KEY, model=MODEL_NAME, base_url=BASE_URL)

researcher = SimpleAgent(
    role="Credit Card Policy Researcher",
    goal="Summarize the exact card policy that applies to the customer request in 1–2 sentences.",
    backstory="You are meticulous and only state what exists inside the retrieved banking policy. Never invent information.",
    llm=agent_llm
)

def run_researcher(extracted_data, retrieved_policy, original_email):
    extracted_json = json.dumps(extracted_data)
    policy_text = ""
    for i, doc in enumerate(retrieved_policy, start=1):
        title = doc.metadata.get("title", "Unknown")
        policy_text += f"\nDocument {i} - {title}:\n{doc.page_content}\n"

    task_description = f"Customer Email:\n{original_email}\n\nExtracted Data:\n{extracted_json}\n\nRetrieved Policy:\n{policy_text}\n"
    return researcher.run(task_description)


reviewer = SimpleAgent(
    role="Quality Reviewer",
    goal="Review the drafted email response and make sure it is professional, accurate, and complete.",
    backstory="You are a senior banking support reviewer. You check that the response is polite, addresses the customer by name, and does not contain made-up information.",
    llm=agent_llm
)

def run_reviewer(resolver_output, original_email):
    task_description = f"Original Customer Email:\n{original_email}\n\nDrafted Response:\n{resolver_output}\n\nPlease review and return the final polished email response."
    return reviewer.run(task_description)


# ─────────────────────────────────────────────
# 5. Resolver Agent
# ─────────────────────────────────────────────

def resolver_agent(email_text, extracted_fields, researcher_summary=None):
    llm = ChatOpenAI(api_key=GROQ_API_KEY, model=MODEL_NAME, base_url=BASE_URL)

    context_section = f"Researcher Summary:\n{researcher_summary}" if researcher_summary else "No policy documents were retrieved for this request type."

    prompt = f"""You are a professional customer support agent for ADIB (Abu Dhabi Islamic Bank).

You received the following customer email:
---
{email_text}
---

Extracted customer information:
{json.dumps(extracted_fields, indent=2)}

Relevant knowledge base documents:
{context_section}

Instructions:
- Write a professional and helpful banking support response.
- Use ONLY the information from the retrieved documents above.
- Do NOT make up any information or policies not mentioned in the documents.
- Address the customer by name if available.
- Keep the tone polite, clear, and professional.
- Sign off as: ADIB Customer Support Team

Write the outgoing email response now:
"""

    response = llm.invoke(prompt)
    return response.content


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    knowledge_base = load_knowledge_base()
    vectorstore = build_chroma_collection(knowledge_base)

    print(f"Documents in ChromaDB collection: {vectorstore._collection.count()}\n")

    # fetch the most recent unread email instead of hardcoded samples
    print("Fetching unread emails from Gmail...\n")
    sample_emails = fetch_unread_emails(max_emails=1)

    if not sample_emails:
        print("No unread emails found in inbox.")
        return

    for email_info in sample_emails:
        label = email_info["label"]
        email_text = email_info["text"]
        sender = email_info.get("from", "Unknown")

        print("=" * 40)
        print(f"========== Incoming Email ==========")
        print(f"[{label}] From: {sender}")
        print(email_text)

        extracted_fields = extract_email_fields(email_text)

        print("\n========== Extracted Fields ==========")
        print(json.dumps(extracted_fields, indent=2))

        request_type = extracted_fields.get("request_type", "").lower()

        if "inquiry" in request_type or "inquir" in request_type:
            print("\n[Route: Inquiry]")

            search_query = extracted_fields.get("request_type", email_text)
            retrieved_docs = search_knowledge(vectorstore, search_query)

            print("\n========== Retrieved Documents ==========")
            for i, doc in enumerate(retrieved_docs, start=1):
                title = doc.metadata.get("title", "Unknown")
                print(f"\n[{i}] {title}")
                print(doc.page_content)

            researcher_summary = run_researcher(extracted_fields, retrieved_docs, email_text)

            print("\n========== Researcher Summary ==========")
            print(researcher_summary)

            resolver_output = resolver_agent(email_text, extracted_fields, researcher_summary)

        else:
            print("\n[Route: Non-Inquiry]")
            resolver_output = resolver_agent(email_text, extracted_fields)

        print("\n========== Resolver Response ==========")
        print(resolver_output)

        final_email = run_reviewer(resolver_output, email_text)

        print("\n========== Reviewer Output ==========")
        print(final_email)

        print("\n========== OUTGOING EMAIL (SENDING NOW) ==========")
        print(final_email)

        # extract the actual email address from the "From" header (e.g. "Name <email@example.com>")
        sender_email = sender
        if "<" in sender and ">" in sender:
            sender_email = sender.split("<")[1].split(">")[0]

        send_reply(sender_email, label, final_email)

        print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    main()