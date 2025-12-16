import streamlit as st
from backend.core import run_llm

st.set_page_config(page_title="LangChain Assistant", page_icon="🤖")

st.title("🤖 LangChain Documentation Assistant")
st.caption("Ask questions about LangChain with conversational memory!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar with controls
with st.sidebar:
    st.header("💬 Chat Controls")
    
    # Display message count
    st.metric("Messages", len(st.session_state.messages))
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("This assistant remembers your conversation context!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available (only for assistant messages)
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.write(doc.page_content[:300] + "...")
                    st.caption(f"🔗 {doc.metadata.get('source', 'Unknown')}")
                    if i < len(message["sources"]):
                        st.divider()

# Chat input
if prompt := st.chat_input("Ask about LangChain..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response with chat history
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documentation..."):
            # Pass chat history (exclude current user message)
            response = run_llm(prompt, st.session_state.messages[:-1])
            answer = response["result"]
            sources = response["source_document"]
            
            st.markdown(answer)
            
            # Show sources in expandable section
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:**")
                    st.write(doc.page_content[:300] + "...")
                    st.caption(f"🔗 {doc.metadata.get('source', 'Unknown')}")
                    if i < len(sources):
                        st.divider()
    
    # Add assistant message with sources
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources
    })

# Show helpful message if no messages yet
if len(st.session_state.messages) == 0:
    st.info("👋 Welcome! Ask me anything about LangChain. I'll remember our conversation context!")
    
    # Example questions
    st.markdown("### 💡 Try asking:")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("What is LangChain?", use_container_width=True):
            st.session_state.example_query = "What is LangChain?"
            st.rerun()
        if st.button("How do I use agents?", use_container_width=True):
            st.session_state.example_query = "How do I use agents in LangChain?"
            st.rerun()
    
    with col2:
        if st.button("What are chains?", use_container_width=True):
            st.session_state.example_query = "What are chains in LangChain?"
            st.rerun()
        if st.button("How does memory work?", use_container_width=True):
            st.session_state.example_query = "How does memory work in LangChain?"
            st.rerun()

# Handle example query clicks
if "example_query" in st.session_state:
    example = st.session_state.example_query
    del st.session_state.example_query
    st.session_state.messages.append({"role": "user", "content": example})
    st.rerun()