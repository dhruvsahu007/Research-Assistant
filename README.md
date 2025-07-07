# Research Assistant

A Streamlit-based AI research assistant that supports hybrid retrieval (PDF + web search) with source verification and citation.

## Features

- PDF document upload and processing
- Real-time web search integration
- Hybrid retrieval combining document embeddings and web results
- Response synthesis with citations using OpenAI API

## Setup

1. Clone the repository.
2. Navigate to the project folder and create a virtual environment:

   ```powershell
   python -m venv venv
   .\\venv\\Scripts\\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Set your OpenAI API key in Streamlit secrets:

   Create a file `.streamlit/secrets.toml` with:

   ```toml
   [general]
   openai_api_key = "YOUR_OPENAI_API_KEY"
   ```

5. Run the app:

   ```powershell
   streamlit run src/server.py
   ```

## Project Structure

```
├── requirements.txt
├── README.md
└── src
    ├── server.py
    ├── pdf_processor.py
    ├── retriever.py
    ├── web_search.py
    └── utils.py
```
