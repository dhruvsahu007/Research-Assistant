import streamlit as st
import os
import openai
from datetime import datetime
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import numpy as np

# Configure page
st.set_page_config(page_title="Research Assistant", layout="wide", page_icon="🔎")
st.title("🔎 AI Research Assistant")

# Get API keys from environment variables or .env file
openai_api_key = os.getenv("OPENAI_API_KEY", "")
if not openai_api_key and os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                openai_api_key = line.strip().split("=", 1)[1]
                break

# Initialize OpenAI client
try:
    if openai_api_key:
        openai_client = openai.OpenAI(api_key=openai_api_key)
        st.sidebar.success("OpenAI API key loaded successfully")
    else:
        openai_client = None
        st.sidebar.warning("OpenAI API key not found. Response synthesis disabled.")
except Exception as e:
    st.sidebar.error(f"OpenAI client error: {e}")
    openai_client = None

# Session state initialization
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    st.session_state.vectorizer_fitted = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# PDF Processing Functions
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    return chunks

# Search functions
def search_documents(query, docs, top_k=5):
    """Search documents using TF-IDF."""
    if not docs:
        return []
        
    # Extract texts for vectorization
    texts = [doc['text'] for doc in docs]
    
    # Fit vectorizer if needed
    if not st.session_state.vectorizer_fitted:
        st.session_state.vectorizer.fit(texts)
        st.session_state.vectorizer_fitted = True
    
    # Transform query and documents
    query_vec = st.session_state.vectorizer.transform([query])
    doc_vecs = st.session_state.vectorizer.transform(texts)
    
    # Calculate similarities
    similarities = (query_vec @ doc_vecs.T).toarray()[0]
    
    # Get top results
    results = []
    top_indices = similarities.argsort()[-top_k:][::-1]
    for idx in top_indices:
        if similarities[idx] > 0:  # Only include if there's some match
            doc = docs[idx]
            results.append({
                'text': doc['text'],
                'source': doc['source'],
                'score': float(similarities[idx]),
                'method': 'tfidf'
            })
    
    return results

def mock_web_search(query, top_k=2):
    """Generate mock web search results."""
    return [
        {
            'title': f'Web Search Result for "{query}" - 1',
            'link': 'https://example.com/result1',
            'snippet': f'This is a simulated search result for the query: "{query}". In a production environment, this would be a real result from a search API.',
            'source': 'web (simulated)',
            'score': 0.95
        },
        {
            'title': f'Web Search Result for "{query}" - 2',
            'link': 'https://example.com/result2',
            'snippet': f'A second simulated result for "{query}". Add a search API key to get real search results.',
            'source': 'web (simulated)',
            'score': 0.85
        }
    ]

# Function to generate citations
def generate_citations(sources):
    """Generate formatted citations for sources."""
    citations = []
    for i, source in enumerate(sources):
        if 'link' in source:  # Web result
            citations.append(f"[{i+1}] {source['title']} - {source['link']}")
        else:
            citations.append(f"[{i+1}] {source['source']}")
    return citations

# Function to synthesize response using OpenAI
def synthesize_response(query, sources):
    """Generate a response using OpenAI based on the sources."""
    if not openai_client:
        return "To get AI-synthesized responses, add your OpenAI API key in the .env file."
    
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
            for pdf_file in uploaded_files:
                # Extract text
                text = extract_text_from_pdf(pdf_file)
                if text:
                    # Chunk text
                    chunks = chunk_text(text)
                    # Add chunks to documents
                    for chunk in chunks:
                        st.session_state.documents.append({
                            'text': chunk,
                            'source': pdf_file.name
                        })
            
            st.success(f"Processed {len(uploaded_files)} PDF files, extracted {len(st.session_state.documents)} document chunks.")
                
    # Search settings
    st.header("⚙️ Search Settings")
    include_web = st.checkbox("Include web search", value=True)
    top_k = st.slider("Number of results", min_value=3, max_value=10, value=5)
    
    # Display document count
    if st.session_state.documents:
        st.info(f"Total document chunks: {len(st.session_state.documents)}")

# Main: query input
def run_query():
    query = st.session_state.get('query_input')
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
        docs = search_documents(query, st.session_state.documents, top_k=top_k)
        
        # Web search
        web_results = []
        if include_web:
            web_results = mock_web_search(query, top_k=min(3, top_k))
        
        # Combine results
        all_results = docs + web_results
        
    # Display results
    st.subheader(f"Query: {query}")
    st.caption(f"Searched at {timestamp}")
    
    # Document results
    if docs:
        st.subheader("📚 Document Results")
        for idx, r in enumerate(docs):
            with st.expander(f"Document {idx+1} - {r['source']} (Score: {r['score']:.2f})"):
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
    - **PDF document processing** with chunking for better retrieval
    - **TF-IDF keyword search** for finding relevant information
    - **Web search integration** (simulated in this demo)
    - **Source verification** with citations
    - **Response synthesis** from multiple sources using OpenAI
    """)
