from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

class PDFProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2', chunk_size=1000, overlap=200):
        # PdfReader doesn't need initialization
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_texts(self, files):
        texts = []
        for f in files:
            pdf = PdfReader(f)
            full_text = "".join([p.extract_text() for p in pdf.pages])
            # chunk text
            for i in range(0, len(full_text), self.chunk_size - self.overlap):
                chunk = full_text[i:i+self.chunk_size]
                if chunk:  # Only add non-empty chunks
                    texts.append({
                        'text': chunk,
                        'source': f.name
                    })
        return texts
