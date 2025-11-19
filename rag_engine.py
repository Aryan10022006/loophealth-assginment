"""
Enhanced RAG Engine with:
- TF-IDF semantic embeddings (lightweight, fast)
- Hybrid search (keyword + semantic)
- Caching for instant startup
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

load_dotenv()

class HospitalRAG:
    def __init__(self, csv_path="hospital.csv", use_cache=True):
        self.csv_path = csv_path
        self.use_cache = use_cache
        self.cache_dir = "rag_cache"
        
        print("🔄 Initializing RAG engine...")
        
        self.hospitals_df = None
        self.documents = []
        self.metadata = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
        self._load_and_index_data()
    
    def _load_and_index_data(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, "tfidf_index.pkl")
        
        if self.use_cache and os.path.exists(cache_file):
            try:
                print("🔄 Loading cached index...")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.hospitals_df = cache_data['hospitals_df']
                    self.documents = cache_data['documents']
                    self.metadata = cache_data['metadata']
                    self.vectorizer = cache_data['vectorizer']
                    self.tfidf_matrix = cache_data['tfidf_matrix']
                print(f"✅ Loaded {len(self.documents)} hospitals from cache")
                return
            except Exception as e:
                print(f"⚠️ Cache failed: {e}. Rebuilding...")
        
        try:
            print(f"🔄 Loading {self.csv_path}...")
            self.hospitals_df = pd.read_csv(self.csv_path, encoding='utf-8', on_bad_lines='skip')
            self.hospitals_df.columns = self.hospitals_df.columns.str.strip().str.lower()
            
            required_columns = ['hospital name', 'address', 'city']
            if not all(col in self.hospitals_df.columns for col in required_columns):
                self._create_dummy_data()
                return
            
            self.hospitals_df = self.hospitals_df.fillna("").drop_duplicates(subset=['hospital name', 'city'])
            print(f"✅ Loaded {len(self.hospitals_df)} hospitals")
            
            print("🔄 Creating semantic documents...")
            self.documents = []
            self.metadata = []
            
            for idx, row in self.hospitals_df.iterrows():
                name = str(row['hospital name']).strip()
                city = str(row['city']).strip()
                address = str(row['address']).strip()
                
                doc = f"{name} {city} {address} hospital healthcare facility medical center"
                self.documents.append(doc)
                self.metadata.append({'name': name, 'city': city, 'address': address, 'index': idx})
            
            print(f"🔄 Building TF-IDF index...")
            self.vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.8,
                sublinear_tf=True
            )
            
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            print(f"✅ TF-IDF index created ({self.tfidf_matrix.shape[0]} vectors)")
            
            if self.use_cache:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump({
                            'hospitals_df': self.hospitals_df,
                            'documents': self.documents,
                            'metadata': self.metadata,
                            'vectorizer': self.vectorizer,
                            'tfidf_matrix': self.tfidf_matrix
                        }, f)
                    print("✅ Index cached")
                except: pass
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        dummy_data = [
            {"hospital name": "Apollo Hospital", "address": "123 Main St", "city": "Bangalore"},
            {"hospital name": "Manipal Hospital", "address": "456 Park Ave", "city": "Bangalore"},
            {"hospital name": "Fortis Hospital", "address": "789 Lake Rd", "city": "Delhi"},
        ]
        self.hospitals_df = pd.DataFrame(dummy_data)
        
        self.documents = []
        self.metadata = []
        for idx, row in self.hospitals_df.iterrows():
            doc = f"{row['hospital name']} {row['city']} {row['address']} hospital"
            self.documents.append(doc)
            self.metadata.append({'name': row['hospital name'], 'city': row['city'], 'address': row['address'], 'index': idx})
        
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        print("✅ Dummy index created")
    
    def search_hospitals(self, query: str, k: int = 5, score_threshold: float = 0.1):
        if not self.vectorizer or self.tfidf_matrix is None:
            return []
        
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = np.argsort(similarities)[::-1][:k]
            
            hospitals = []
            for idx in top_indices:
                if idx < len(self.metadata) and similarities[idx] >= score_threshold:
                    meta = self.metadata[idx]
                    score = float(similarities[idx])
                    hospitals.append({
                        "name": meta['name'],
                        "address": meta['address'],
                        "city": meta['city'],
                        "full_text": self.documents[idx],
                        "score": score,
                        "relevance": "high" if score > 0.5 else "medium" if score > 0.2 else "low"
                    })
            return hospitals
        except:
            return []
    
    def search_by_name_and_city(self, name: str, city: str = None, k: int = 5):
        if self.hospitals_df is None:
            return []
        
        try:
            mask = self.hospitals_df['hospital name'].str.lower().str.contains(name.lower(), na=False, regex=False)
            if city:
                mask = mask & self.hospitals_df['city'].str.lower().str.contains(city.lower(), na=False, regex=False)
            
            text_results = self.hospitals_df[mask]
            hospitals = []
            for _, row in text_results.head(k).iterrows():
                hospitals.append({"name": row['hospital name'], "address": row['address'], "city": row['city'], "score": 1.0, "relevance": "high"})
            
            if len(hospitals) < k:
                query = f"{name} hospital in {city}" if city else f"{name} hospital"
                semantic_results = self.search_hospitals(query, k=k)
                existing_names = {h['name'].lower() for h in hospitals}
                for result in semantic_results:
                    if result['name'].lower() not in existing_names:
                        hospitals.append(result)
                        if len(hospitals) >= k:
                            break
            
            return hospitals[:k]
        except:
            query = f"{name} hospital in {city}" if city else f"{name} hospital"
            return self.search_hospitals(query, k=k)


rag_engine = None

def get_rag_engine():
    global rag_engine
    if rag_engine is None:
        rag_engine = HospitalRAG()
    return rag_engine
