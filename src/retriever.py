from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class Retriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dense_index = None
        self.sparse_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        self.documents = []
        
    def add_documents(self, docs):
        if not docs:
            return
            
        # Store documents
        self.documents.extend(docs)
        
        # Extract just the text for embeddings
        texts = [doc['text'] for doc in docs]
        
        # Dense index (semantic search)
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        d = embeddings.shape[1]
        if self.dense_index is None:
            self.dense_index = faiss.IndexFlatL2(d)
        self.dense_index.add(embeddings)
        
        # Sparse index (keyword search)
        if len(self.documents) == len(docs):  # First time adding docs
            self.sparse_vectorizer.fit(texts)
        
        print(f"Added {len(docs)} documents to retriever")

    def hybrid_search(self, query, top_k=5, alpha=0.5):
        if not self.documents:
            return []
            
        results = []
        
        # Dense search
        if self.dense_index:
            q_emb = self.model.encode([query], convert_to_numpy=True)
            D, I = self.dense_index.search(q_emb, min(top_k, len(self.documents)))
            
            for i, (score, idx) in enumerate(zip(D[0], I[0])):
                if idx < len(self.documents):  # Safety check
                    doc = self.documents[idx]
                    results.append({
                        'id': idx,
                        'text': doc['text'],
                        'source': doc['source'],
                        'score': float(1/(1+score)),  # Convert distance to similarity score
                        'method': 'dense'
                    })
        
        # Sparse search (TF-IDF)
        if len(self.documents) > 0:
            try:
                query_vec = self.sparse_vectorizer.transform([query])
                doc_vecs = self.sparse_vectorizer.transform([doc['text'] for doc in self.documents])
                
                # Calculate cosine similarity
                similarities = (query_vec @ doc_vecs.T).toarray()[0]
                
                # Get top results
                top_indices = similarities.argsort()[-top_k:][::-1]
                for idx in top_indices:
                    if similarities[idx] > 0:  # Only include if there's some match
                        doc = self.documents[idx]
                        results.append({
                            'id': idx,
                            'text': doc['text'],
                            'source': doc['source'],
                            'score': float(similarities[idx]),
                            'method': 'sparse'
                        })
            except Exception as e:
                print(f"Sparse search error: {e}")
        
        # Combine results (removing duplicates, favoring higher scores)
        combined = {}
        for r in results:
            doc_id = r['id']
            if doc_id not in combined or r['score'] > combined[doc_id]['score']:
                combined[doc_id] = r
                
        # Return sorted results
        return sorted(combined.values(), key=lambda x: x['score'], reverse=True)[:top_k]
