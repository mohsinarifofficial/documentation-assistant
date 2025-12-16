from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import Client
from langchain_classic.hub import pull
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeEmbeddings
from typing import List, Dict, Any
import dotenv, os

dotenv.load_dotenv()
client=Client()



def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = PineconeEmbeddings(
        model="llama-text-embed-v2",
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )

    doc_search = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"), 
        embedding=embeddings
    )

    llm = ChatGroq(
        api_key=os.getenv("api_key"),
        model="meta-llama/llama-4-scout-17b-16e-instruct"
    )

    rephrase_prompt = pull("langchain-ai/chat-langchain-rephrase")
    
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=doc_search.as_retriever(),
        prompt=rephrase_prompt
    )

    retrieval_qa_chat_prompt = pull("langchain-ai/retrieval-qa-chat")
    
    stuff_doc = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    retrieval_qa_chain = create_retrieval_chain(
        retriever=history_aware_retriever,  # Use history-aware retriever
        combine_docs_chain=stuff_doc
    )
    formatted_chat_history = []
    for message in chat_history:
        if message["role"] == "user":
            formatted_chat_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            formatted_chat_history.append(AIMessage(content=message["content"]))

    response = retrieval_qa_chain.invoke({
        "input": query,
        "chat_history": formatted_chat_history
    })

    new_response = {
        "query": response["input"],
        "result": response["answer"],
        "source_document": response["context"]
    }
    return new_response

if __name__=="__main__":
    response=run_llm("What is the purpose of the LangChain library?")
    print(response["result"])



