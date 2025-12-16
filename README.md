# documentation-assistant
LangChain Documentation Assistant with Conversational Memory

An intelligent Retrieval-Augmented Generation (RAG) chatbot that provides accurate answers about LangChain documentation with full conversational memory capabilities.
🌟 Features
Core Capabilities

📚 Documentation Ingestion Pipeline: Automatically crawls and indexes LangChain documentation using Tavily
🧠 Conversational Memory: Remembers chat history and rephrases follow-up questions for context-aware responses
🔍 Semantic Search: Uses Pinecone vector database with embeddings for intelligent document retrieval
⚡ Fast Inference: Powered by Groq's LLaMA 4 Scout model for lightning-fast responses
💬 Interactive Chat UI: Beautiful Streamlit interface with chat history and source citations

Technical Architecture

LLM: Groq Cloud (Meta's LLaMA 4 Scout 17B)
Embeddings: Pinecone's LLaMA Text Embed V2
Vector Store: Pinecone for scalable document storage and retrieval
Web Scraping: Tavily for intelligent documentation crawling
Memory: History-aware retriever with question rephrasing
Framework: LangChain for orchestrating the RAG pipeline
