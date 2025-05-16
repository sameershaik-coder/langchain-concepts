from langchain_openai import OpenAIEmbeddings
import faiss
import numpy as np

# Instantiate the OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)

# Example documents
documents = [
    "LangChain is a framework for building applications powered by LLMs.",
    "OpenAI GPT models are powerful language models that can generate text."
]


from langchain_core.vectorstores import InMemoryVectorStore



vectorstore = InMemoryVectorStore.from_texts(
    documents,
    embedding=embeddings,
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is OpenAI?")

# show the retrieved document's content
result = retrieved_documents[0].page_content

print(result)