# Load blog
import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

# Index
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

from langchain.prompts import ChatPromptTemplate

# Multi Query Generation System
# This system generates multiple perspectives of the same question to improve retrieval quality.
# For example, if the original question is "What is task decomposition?", it might generate:
# 1. "Explain the concept of task decomposition in AI"
# 2. "How do LLM agents break down complex tasks?"
# 3. "What are the steps involved in task decomposition?"
# 4. "Define task decomposition and its importance"
# 5. "What is the role of task decomposition in agent systems?"

template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

from langchain.load import dumps, loads

def get_unique_union(documents: list[list]) -> list:
    """
    Removes duplicate documents from multiple retrieval results.
    
    This function is crucial for efficiency and non-redundancy because:
    1. Different query perspectives might retrieve the same documents
    2. We want to avoid showing the same information multiple times
    3. We need to maintain document metadata and structure
    
    Args:
        documents: A list of lists, where each inner list contains Document objects
                  retrieved from different query perspectives
    
    Returns:
        A list of unique Document objects, maintaining their original structure
    """
    # Flatten list of lists, and convert each Document to string for deduplication
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents using set operation
    unique_docs = list(set(flattened_docs))
    # Convert back to Document objects
    return [loads(doc) for doc in unique_docs]

from langchain_openai import ChatOpenAI

# Initialize LLM with temperature=0 for consistent, deterministic responses
llm = ChatOpenAI(temperature=0)

def generate_multiple_queries(question: str) -> list[str]:
    """
    Generates multiple variations of the input question to capture different perspectives.
    
    This function enhances retrieval by:
    1. Overcoming limitations of pure similarity search
    2. Capturing different aspects or phrasings of the same question
    3. Increasing the likelihood of finding relevant context
    
    Args:
        question: The original user question
        
    Returns:
        A list of rephrased questions, typically 5 variations
    """
    # Create the prompt for generating different perspectives
    prompt = prompt_perspectives.format(question=question)
    
    # Get response from LLM with variations
    response = llm.invoke(prompt)
    
    # Split into separate queries and clean up whitespace
    queries = [q.strip() for q in response.content.split('\n') if q.strip()]
    
    return queries

def get_rag_response(question: str) -> str:
    """
    Main RAG function that implements multi-query retrieval augmented generation.
    
    The process works in several steps:
    1. Generate multiple perspectives of the question
    2. Retrieve relevant documents for each perspective
    3. Remove duplicate documents
    4. Combine unique documents into context
    5. Generate a comprehensive answer using all gathered context
    
    This approach is particularly effective for:
    - Complex questions that can be viewed from multiple angles
    - Questions where simple semantic similarity might miss important context
    - Ensuring comprehensive coverage of relevant information
    
    Args:
        question: The user's original question
        
    Returns:
        A comprehensive answer based on retrieved context
    """
    # Step 1: Generate multiple perspectives of the question
    queries = generate_multiple_queries(question)
    
    # Step 2: Retrieve documents for each query perspective
    all_docs = []
    for query in queries:
        docs = retriever.get_relevant_documents(query)
        all_docs.append(docs)
    
    # Step 3: Remove duplicate documents while preserving structure
    unique_docs = get_unique_union(all_docs)
    
    # Step 4: Format all unique documents into a single context
    context = "\n\n".join(doc.page_content for doc in unique_docs)
    
    # Step 5: Create the final answer prompt with the comprehensive context
    answer_prompt = ChatPromptTemplate.from_template(
        """Answer the following question based on this context:
        
        Context: {context}
        
        Question: {question}
        Answer:"""
    )
    
    # Generate the final answer using all gathered context
    prompt = answer_prompt.format(context=context, question=question)
    response = llm.invoke(prompt)
    
    return response.content

# Example usage
question = "What is task decomposition for LLM agents?"
result = get_rag_response(question)
print("\nQuestion:", question)
print("\nAnswer:", result)