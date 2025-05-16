from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
import bs4
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# Load blog
web_loader = WebBaseLoader(
    web_paths=("https://arxiv.org/html/2412.05718v1",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("ltx_abstract", "ltx_section", "ltx_appendix")
        )
    ),
)
blog_docs = web_loader.load()

# csv_loader = CSVLoader(file_path="./data/sample.csv")
# csv_docs = csv_loader.load()

pdf_loader = PyPDFLoader(file_path="./data/sample_zero_shot_ai.pdf")
pdf_docs = pdf_loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(blog_docs + pdf_docs)
print(len(splits))

# Index

vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()



llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
questions = [
            "What is Zero-shot learning?",
        ]

for question in questions:
    print(f"\nQuestion: {question}")
    retrieved_docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print("Retrieved context:", context)
    prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context. 
            Use only the information from the context to answer questions. If you cannot find the answer 
            in the context, say "I cannot answer this based on the available information."
            
            Context: {context}"""),
            ("human", "{question}")
        ])
    messages = prompt_template.format_messages(
            context=context,
            question=question
        )
        
    # Get response from LLM
    response = llm.invoke(messages)
    
    answer = response.content
    sources = retrieved_docs
    print(f"\nAnswer: {answer}")
    print("\nSources used:")
    for i, source in enumerate(sources[:2], 1):
        print(f"Source {i}: {source.page_content[:200]}...")


