from langchain_core.documents.base import Document


from configparser import MAX_INTERPOLATION_DEPTH
from langchain_groq import ChatGroq
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeEmbeddings
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors,log_error,log_header,log_info,log_success,log_warning)
from typing import Any, Dict,List
import asyncio
import ssl
import certifi
import dotenv, os

dotenv.load_dotenv()

ssl_context=ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"]=certifi.where()
os.environ["REQUEST_CA_BUNDLE"]=certifi.where()


llm = ChatGroq(
        api_key=os.getenv("api_key"),
        model="meta-llama/llama-4-scout-17b-16e-instruct"
    )
embeddings=PineconeEmbeddings( model="llama-text-embed-v2",
    pinecone_api_key=os.getenv("PINECONE_API_KEY"))

vectore_store=PineconeVectorStore(index_name=os.getenv("INDEX_NAME"),
pinecone_api_key=os.getenv("PINECONE_API_KEY"), embedding=embeddings)

tavily_crawl=TavilyCrawl()
tavilty_extract=TavilyExtract()
tavily_map=TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)


async def index_documents_async(documents:List[Document], batch_size:int=500):
    """Documents will processed in batches of 500 and indexed into the vector store."""
    log_header("Indexing documents")
    log_info(f"Indexing {len(documents)} documents in batches of {batch_size}")
    batches=[documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
    log_info(f"Created {len(batches)} batches of {batch_size} documents")

    async def index_batch_async(batch:List[Document]):
        try:
            vectore_store.add_documents(batch)
            log_success(f"Indexed {len(batch)} documents successfully")
        except Exception as e:
            log_error(f"Error indexing batch: {e}")
            raise
    
    tasks=[index_batch_async(batch) for i, batch in enumerate(batches)]
    results=await asyncio.gather(*tasks,return_exceptions=True)
    success_count=sum(1 for result in results if result is None)
    if success_count==len(batches):
        log_success("All documents indexed successfully")
    else:
        log_error(f"Failed to index {len(batches)-success_count} documents")
        raise Exception(f"Failed to index {len(batches)-success_count} documents")
    log_success(f"Successfully indexed {success_count} documents out of {len(batches)}")


async def main():
    """
    Main asyn function to orchestrate the entire process.
    """
    log_header("Documenetaion ingestion pipleine")

    log_info(
        "TavilyCrawl: started crawling the documenetaion from https://docs.langchain.com/oss/python/langchain/streaming"
    )
    
    res=tavily_crawl.invoke({
        "url": "https://docs.langchain.com/oss/python/langchain/streaming",
        "max_depth":1,
        "extract_Depth":"advanced",
        "instructions": "Content on AI agents"

    })
    all_docs=res["results"]
    all_docs=[Document(page_content=result["raw_content"], metadata={"source":result["url"]}) for result in res["results"]]
    log_success(    
        f"TavilyCrawl: Succesfully {all_docs} URLS from the langhcaon docuementaion site."
    )
    log_header("Document chunking")
    log_info(f"Text splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap")
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs=text_splitter.split_documents(all_docs)
    log_success(f"Text Splitter: created {len(splitted_docs)} chunks from {len(all_docs)}")


    await index_documents_async(splitted_docs, batch_size=500)
    log_success("Document indexing completed successfully")
    log_header("Document indexing completed successfully")
    log_info("Document indexing completed successfully")
    log_warning("Document indexing completed successfully")
    log_error("Document indexing completed successfully")
    log_success("Document indexing completed successfully")
    log_header("Document indexing completed successfully")
    log_info("Document indexing completed successfully")
    log_warning("Document indexing completed successfully")
    log_error("Document indexing completed successfully")






    


if __name__=="__main__":
    asyncio.run(main())