import pandas as pd
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_csv_rag.ipynb

class CSVBasedRAG:
    def __init__(self):
        load_dotenv()
        self.encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        self.vectorstore = None
        self.llm = None
        
        # Define the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context. 
            Use only the information from the context to answer questions. If you cannot find the answer 
            in the context, say "I cannot answer this based on the available information."
            
            Context: {context}"""),
            ("human", "{question}")
        ])
        
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load CSV with automatic encoding detection"""
        for encoding in self.encodings:
            try:
                logger.info(f"Attempting to read with {encoding} encoding...")
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    na_values=['NA', 'N/A', ''],
                    keep_default_na=True
                )
                logger.info(f"Successfully read {len(df)} rows with {encoding}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading CSV: {str(e)}")
                continue
        raise RuntimeError("Failed to read CSV with any encoding")

    def prepare_documents(self, df: pd.DataFrame) -> list:
        """Convert DataFrame rows to text documents"""
        documents = []
        for _, row in df.iterrows():
            text = " ".join([f"{col}: {val}" for col, val in row.items()])
            documents.append(text)
        return documents

    def create_rag_pipeline(self, file_path: str):
        """Create RAG pipeline without chains"""
        # 1. Load CSV
        df = self.load_csv(file_path)
        
        # 2. Prepare documents
        docs = self.prepare_documents(df)
        
        # 3. Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        split_docs = text_splitter.create_documents(docs)
        
        # 4. Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # 5. Create vector store
        self.vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        # 6. Initialize LLM
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def query(self, question: str) -> tuple:
        """Query the RAG system directly without chains"""
        if not self.vectorstore or not self.llm:
            raise RuntimeError("RAG pipeline not initialized. Call create_rag_pipeline first.")
        
        # Retrieve relevant documents
        retrieved_docs = self.vectorstore.similarity_search(question, k=3)
        
        # Combine retrieved contexts
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Format prompt with context and question
        messages = self.prompt_template.format_messages(
            context=context,
            question=question
        )
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        
        return response.content, retrieved_docs

def main():
    # Initialize RAG system
    rag = CSVBasedRAG()
    
    # Create pipeline
    file_path = "./data/sample.csv"
    try:
        rag.create_rag_pipeline(file_path)
        
        # Example queries
        questions = [
            "which company does sheryl Baxter work for?",
            "provide names of all the customers located in chile",
        ]
        
        for question in questions:
            print(f"\nQuestion: {question}")
            answer, sources = rag.query(question)
            print(f"\nAnswer: {answer}")
            print("\nSources used:")
            for i, source in enumerate(sources[:2], 1):
                print(f"Source {i}: {source.page_content[:200]}...")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()