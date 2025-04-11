import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

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

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

# Index
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

# Initialize components
llm = ChatOpenAI(temperature=0)

# Create decomposition prompt template
template = """You are a helpful assistant that generates multiple sub-questions related to an input question.
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation.
Generate multiple search queries related to: {question}
Output (3 queries):"""

prompt_decomposition = ChatPromptTemplate.from_template(template)

def generate_sub_questions(question: str) -> list[str]:
    # Create the prompt
    prompt = prompt_decomposition.format(question=question)
    
    # Get response from LLM
    response = llm.invoke(prompt)
    
    # Split the response into separate questions
    sub_questions = [q.strip() for q in response.content.split('\n') if q.strip()]
    
    return sub_questions

def get_rag_responses(question: str) -> dict[str, str]:
    # Generate sub-questions
    sub_questions = generate_sub_questions(question)
    
    # Get answers for each sub-question
    answers = {}
    for sub_q in sub_questions:
        # Get relevant documents
        docs = retriever.get_relevant_documents(sub_q)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Create prompt for answering the sub-question
        answer_prompt = ChatPromptTemplate.from_template(
            """Based on the following context, answer the question:
            Context: {context}
            Question: {question}
            Answer:"""
        )
        
        # Get answer from LLM
        prompt = answer_prompt.format(context=context, question=sub_q)
        response = llm.invoke(prompt)
        
        answers[sub_q] = response.content
    
    return answers

# Example usage
question = "What are the main components of an LLM-powered autonomous agent system?"
sub_questions_and_answers = get_rag_responses(question)

# Print results
print("Original question:", question)
print("\nSub-questions and answers:")
for sub_q, answer in sub_questions_and_answers.items():
    print(f"\nSub-question: {sub_q}")
    print(f"Answer: {answer}")