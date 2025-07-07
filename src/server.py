import streamlit as st
from pdf_processor import PDFProcessor
from retriever import Retriever
from web_search import WebSearch
import os
import openai
from datetime import datetime

# Configure page
st.set_page_config(page_title="Research Assistant", layout="wide", page_icon="🔎")
st.title("🔎 AI Research Assistant")

# Get API keys from Streamlit secrets or environment variables
openai_api_key = st.secrets.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
serper_api_key = st.secrets.get("serper_api_key", os.getenv("SERPER_API_KEY", ""))

# Initialize OpenAI client
try:
    if openai_api_key:
        openai_client = openai.Client(api_key=openai_api_key)
    else:
        openai_client = None
        st.sidebar.warning("OpenAI API key not found. Response synthesis disabled.")
except Exception as e:
    st.sidebar.error(f"OpenAI client error: {e}")
    openai_client = None

# Initialize components on first run
if 'processor' not in st.session_state:
    st.session_state.processor = PDFProcessor()
if 'retriever' not in st.session_state:
    st.session_state.retriever = Retriever()
if 'web_search' not in st.session_state:
    st.session_state.web_search = WebSearch(api_key=serper_api_key)
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Function to generate citations
def generate_citations(sources):
    citations = []
    for i, source in enumerate(sources):
        if source['source'].startswith('web'):
            citations.append(f"[{i+1}] {source['title']} - {source['link']}")
        else:
            citations.append(f"[{i+1}] {source['source']}")
    return citations

# Function to synthesize response using OpenAI
def synthesize_response(query, sources):
    if not openai_client:
        return "To get AI-synthesized responses, add your OpenAI API key in .streamlit/secrets.toml"
    
    # Create context from sources
    context = ""
    for i, source in enumerate(sources):
        if 'snippet' in source:  # Web result
            context += f"\nSource {i+1} (Web - {source['title']}):\n{source['snippet']}\n"
        else:  # Document
            context += f"\nSource {i+1} (Document - {source['source']}):\n{source['text']}\n"
    
    try:
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a research assistant that provides accurate, factual responses based on the provided sources. Include citations like [1], [2] to link your statements to the provided sources. Be clear about what information comes from which source."},
                {"role": "user", "content": f"Query: {query}\n\nSources:\n{context}\n\nPlease synthesize an accurate response to the query using only the provided sources. Include citations."}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error synthesizing response: {str(e)}"

# Sidebar: PDF upload
with st.sidebar:
    st.header("📄 Upload PDF Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            docs = st.session_state.processor.extract_texts(uploaded_files)
            if docs:
                st.session_state.retriever.add_documents(docs)
                st.success(f"Processed {len(docs)} document chunks from {len(uploaded_files)} files.")
            else:
                st.error("No text could be extracted from the uploaded PDFs.")
                
    # Search settings
    st.header("⚙️ Search Settings")
    include_web = st.checkbox("Include web search", value=True)
    top_k = st.slider("Number of results", min_value=3, max_value=10, value=5)
    
    # Add info about API keys
    st.header("🔑 API Keys")
    if not openai_api_key:
        st.info("Add OpenAI API key in .streamlit/secrets.toml for response synthesis")
    if not serper_api_key and include_web:
        st.info("Add Serper API key in .streamlit/secrets.toml for real web search")

# Main: query input
def run_query():
    query = st.session_state['query_input']
    if not query:
        return
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save to history
    st.session_state.query_history.append({
        "query": query,
        "timestamp": timestamp
    })
    
    with st.spinner("Searching..."):
        # Document search
        docs = st.session_state.retriever.hybrid_search(query, top_k=top_k)
        
        # Web search
        web_results = []
        if include_web:
            web_results = st.session_state.web_search.search(query, top_k=top_k)
        
        # Combine results
        all_results = docs + web_results
        
    # Display results
    st.subheader(f"Query: {query}")
    st.caption(f"Searched at {timestamp}")
    
    # Document results
    if docs:
        st.subheader("📚 Document Results")
        for idx, r in enumerate(docs):
            with st.expander(f"Document {idx+1} - {r['source']} (Score: {r['score']:.2f}, Method: {r['method']})"):
                st.markdown(r['text'])
    elif not web_results:
        st.info("No document results found. Upload PDFs to search through them.")
            
    # Web results
    if web_results:
        st.subheader("🌐 Web Results")
        for idx, res in enumerate(web_results):
            with st.expander(f"{res['title']}"):
                st.markdown(f"**Source**: {res['link']}")
                st.markdown(res['snippet'])
    elif include_web:
        st.info("No web results found.")
    
    # AI synthesis
    if all_results and openai_client:
        st.subheader("🤖 AI Synthesis")
        with st.spinner("Generating synthesis..."):
            synthesis = synthesize_response(query, all_results)
            st.markdown(synthesis)
        
        # Display citations
        with st.expander("Sources & Citations"):
            citations = generate_citations(all_results)
            for citation in citations:
                st.markdown(citation)
    
# Query input field
st.text_input("Enter your research question:", key='query_input', on_change=run_query)

# Show recent queries
if st.session_state.query_history:
    with st.expander("Recent queries"):
        for item in reversed(st.session_state.query_history[-5:]):
            st.text(f"{item['timestamp']}: {item['query']}")
            
# Instructions
with st.expander("How to use this Research Assistant"):
    st.markdown("""
    ### Instructions:
    1. **Upload PDF documents** using the sidebar on the left
    2. **Enter a research question** in the text box above
    3. View results from your documents and the web
    4. Read the AI synthesis combining information from multiple sources
    
    ### Features:
    - **Hybrid search** combines keyword and semantic search
    - **Web search integration** for real-time information (if API key provided)
    - **Source verification** with citations
    - **Response synthesis** from multiple sources (if OpenAI API key provided)
    """)
