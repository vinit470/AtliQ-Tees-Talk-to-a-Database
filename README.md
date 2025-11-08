# 🛍️ AtliQ Tees – Natural Language to SQL LLM Assistant

This is an **end-to-end LLM-powered system** built using **Google Gemini + LangChain** that allows users to interact with a **MySQL database using plain English**.  

The system understands user questions, converts them into **MySQL queries**, executes the queries on the database, and returns clear answers — without the user needing to know SQL.

---

## 🚀 Use Case

AtliQ Tees is a T-shirt retail business that stores its **inventory, sales, and discount** data in a MySQL database.  
Store managers often ask analytics or inventory-related questions, such as:

| Example Question | What the System Does |
|-----------------|---------------------|
| *"How many white color Adidas t-shirts do we have left in stock?"* | Generates a `SELECT COUNT()` query and returns the exact count. |
| *"How much revenue will we generate if we sell all extra-small size t-shirts after applying discounts?"* | Calculates total possible sales based on current stock and discount logic. |

This system **automatically** generates the right SQL query and executes it — saving time and ensuring accuracy.

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| LLM | Google Gemini (via `langchain-google-genai`) |
| Framework | LangChain + LangChain Experimental |
| Database | MySQL (local or cloud) |
| Vector Store (for few-shot learning) | ChromaDB |
| Embeddings | HuggingFace Sentence Transformers |
| Environment Handling | python-dotenv |
| UI (Optional) | Streamlit |

---

## 📦 Installation

```bash
git clone <your-repo-url>
cd <your-project-folder>
pip install -r requirements.txt
