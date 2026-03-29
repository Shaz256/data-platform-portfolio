from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from app.data import medical_data

# Use fake embeddings (no API key needed)
embeddings = FakeEmbeddings(size=384)

# Create vector DB
db = FAISS.from_texts(medical_data, embeddings)

def get_response(query: str):
    docs = db.similarity_search(query, k=2)

    results = [doc.page_content for doc in docs]

    if not results:
        return "No relevant medical data found."

    # Simple response generation
    response = "Based on your records:\n"
    for r in results:
        response += f"- {r}\n"

    return response











'''from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from app.data import medical_data

embeddings = OpenAIEmbeddings()

# Create vector DB
db = FAISS.from_texts(medical_data, embeddings)

llm = OpenAI()

def get_response(query: str):
    docs = db.similarity_search(query)

    context = " ".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a medical assistant.

    Patient data:
    {context}

    Question:
    {query}
    """

    return llm(prompt)'''