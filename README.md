# adib-email-fetching

An automated banking support pipeline that fetches real customer emails, extracts requests, retrieves relevant policies via RAG, and generates and sends replies automatically using a multi-agent (Researcher–Resolver–Reviewer) system.

## How It Works

1. **Fetch** — Connects to Gmail via IMAP and pulls the latest unread customer email.
2. **Extract** — Uses LangChain function calling (via Groq) to pull structured fields from the email (customer name, request type, card type, priority).
3. **Route** — Classifies the request as either a general inquiry or a direct action request.
4. **Retrieve (RAG)** — For inquiries, searches a ChromaDB knowledge base of ADIB card policies to find the relevant information.
5. **Multi-Agent Response** — Three agents work together to build the final reply:
   - **Researcher**: Summarizes the exact policy relevant to the request
   - **Resolver**: Drafts a concrete, actionable response
   - **Reviewer**: Polishes the reply for tone, accuracy, and completeness
6. **Send** — Automatically emails the final reply back to the customer via Gmail SMTP.

This means a real customer email can go in one end, and a fully AI-generated, policy-grounded, ready-to-send reply comes out the other — with no manual steps in between.

## What's in This Repo

- `main.py` — the full pipeline (fetch → extract → route → retrieve → agents → send)
- `requirements.txt` — all required Python dependencies
- `knowledge_base.json` — the ADIB card policy documents used for RAG

## How to Run It

1. **Clone the repository**
```bash
   git clone https://github.com/USERNAME/adib-email-fetching.git
   cd adib-email-fetching
```

2. **Install the dependencies**
```bash
   pip install -r requirements.txt
```

3. **Create a `.env` file** in the project root with the following:
GROQ_API_KEY=your_groq_api_key_here
GMAIL_ADDRESS=your_gmail_address_here
GMAIL_APP_PASSWORD=your_gmail_app_password_here
   > Note: `GMAIL_APP_PASSWORD` is **not** your regular Gmail password — it's a 16-character App Password generated from your Google Account (requires 2-Step Verification to be enabled). Get one at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords).

4. **Send yourself a test email** (leave it unread in your inbox), then run:
```bash
   python main.py
```

The script will fetch your latest unread email, process it through the full pipeline, print each stage's output, and automatically send the AI-generated reply back to the sender.
