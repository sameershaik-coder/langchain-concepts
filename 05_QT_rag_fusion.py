#### INDEXING ####
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads

# Load blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50
)
splits = text_splitter.split_documents(blog_docs)

# Create vector store and retriever
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

def generate_search_queries(question: str, llm: ChatOpenAI) -> list:
    """Generate multiple search queries for a given question."""
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
    
    prompt = ChatPromptTemplate.from_template(template)
    messages = prompt.format_messages(question=question)
    response = llm.invoke(messages)
    return [q.strip() for q in response.content.split('\n') if q.strip()]

def reciprocal_rank_fusion(results: list[list], k=60):
    """Combine multiple ranked lists of documents using reciprocal rank fusion."""
    fused_scores = {}

    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

def get_rag_response(question: str, context: list, llm: ChatOpenAI) -> str:
    """Generate a response using RAG based on the provided context."""
    template = """Answer the following question based on this context:

{context}

Question: {question}
"""
    # Format context - using the top 3 documents
    formatted_context = "\n\n".join([doc.page_content for doc, _ in context[:3]])
    
    prompt = ChatPromptTemplate.from_template(template)
    messages = prompt.format_messages(
        context=formatted_context,
        question=question
    )
    response = llm.invoke(messages)
    return response.content

def main():
    question = "What is task decomposition for LLM agents?"
    llm = ChatOpenAI(temperature=0)
    
    # Generate multiple queries
    search_queries = generate_search_queries(question, llm)
    
    # Get results for each query
    all_results = []
    for query in search_queries:
        results = retriever.get_relevant_documents(query)
        all_results.append(results)
    
    # Apply reciprocal rank fusion
    fused_results = reciprocal_rank_fusion(all_results)
    
    # Get final response
    final_response = get_rag_response(question, fused_results, llm)
    print(final_response)

if __name__ == "__main__":
    main()