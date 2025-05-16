from langchain_community.document_loaders import CSVLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.prompts import ChatPromptTemplate

file_path = os.path.join("data", "imdb_top_1000.csv")
loader = CSVLoader(
            file_path=file_path,
            encoding='utf-8',  # Explicitly set UTF-8 encoding
            csv_args={
                'delimiter': ',',
                'quotechar': '"',
                'fieldnames': [
                    "Poster_Link", "Series_Title", "Released_Year", "Certificate",
                    "Runtime", "Genre", "IMDB_Rating", "Overview", "Meta_score",
                    "Director", "Star1", "Star2", "Star3", "Star4", "No_of_Votes", "Gross"
                ]
            },
        )
docs = loader.load()
print(f"Loaded {len(docs)} documents from {file_path}")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50
)
splits = splitter.split_documents(docs)
print(f"Split into {len(splits)} chunks")

store = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()
)

retriever = store.as_retriever()
print("Vector store created and retriever initialized.")
# Example usage of the retriever
query = "What are the top-rated movies in 2020?"
results = retriever.get_relevant_documents(query)
print(f"Found {len(results)} relevant documents for the query: '{query}'")


llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    temperature=0.0, 
    max_tokens=1000
)
    
prompt_template = """Answer the following question based on the provided context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
messages = prompt.format_messages(
    context="\n\n".join([doc.page_content for doc in results[:3]]),
    question=query
)
response = llm.invoke(messages)
print(f"Response: {response.content}")
# This code snippet demonstrates how to load a CSV file without using pandas,
# split the documents, create a vector store, and generate a response using RAG (Retrieval-Augmented Generation).

