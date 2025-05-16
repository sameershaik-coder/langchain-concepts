import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed and create vectorstore
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

# Initialize retriever and LLM
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create prompt template
prompt_template = """Answer the question based on the following context:

Context: {context}

Question: {question}

Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_rag_response(question: str) -> str:
    # Get relevant documents
    retrieved_docs = retriever.get_relevant_documents(question)
    
    # Format documents
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    print("Retrieved context:", context)  # Optional: for debugging
    
    # Create prompt
    prompt = PROMPT.format(context=context, question=question)
    
    # Get response from LLM
    response = llm.invoke(prompt)
    
    return response.content

# Example usage
question = "What is Task Decomposition?"
result = get_rag_response(question)
print("\nAnswer:", result)