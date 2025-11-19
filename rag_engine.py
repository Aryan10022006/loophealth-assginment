import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class HospitalRAG:
    def __init__(self, csv_path="hospital.csv"):
        """Initialize the RAG engine with hospital data - using TF-IDF (100% FREE, no API!)"""
        self.csv_path = csv_path
        self.vectorizer = None
        self.tfidf_matrix = None
        self.hospitals_df = None
        self.documents = []
        self._load_and_index_data()
    
    def _load_and_index_data(self):
        """Load CSV and create TF-IDF index."""
        try:
            # Load CSV
            self.hospitals_df = pd.read_csv(self.csv_path)
            
            # Clean column names (strip whitespace, lowercase)
            self.hospitals_df.columns = self.hospitals_df.columns.str.strip().str.lower()
            
            # Check if required columns exist
            required_columns = ['hospital name', 'address', 'city']
            missing_columns = [col for col in required_columns if col not in self.hospitals_df.columns]
            
            if missing_columns:
                print(f"Warning: Missing columns {missing_columns}. Creating dummy data.")
                self._create_dummy_data()
                return
            
            # Clean data - remove NaN values
            self.hospitals_df = self.hospitals_df.fillna("")
            
            # Create text documents for search
            self.documents = []
            for idx, row in self.hospitals_df.iterrows():
                # Concatenate important columns into searchable text
                text = f"{row['hospital name']} {row['city']} {row['address']}"
                self.documents.append(text)
            
            print(f"Loaded {len(self.documents)} hospitals from CSV")
            
            # Create TF-IDF vectorizer (FREE - runs instantly, no downloads!)
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)  # Use unigrams and bigrams
            )
            
            # Fit and transform documents
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            print("TF-IDF index created successfully")
            
        except FileNotFoundError:
            print(f"Warning: CSV file '{self.csv_path}' not found. Creating dummy data.")
            self._create_dummy_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data if CSV is missing."""
        dummy_data = [
            {"hospital name": "Apollo Hospital", "address": "123 Main St", "city": "Bangalore"},
            {"hospital name": "Manipal Hospital", "address": "456 Park Ave", "city": "Bangalore"},
            {"hospital name": "Fortis Hospital", "address": "789 Lake Rd", "city": "Delhi"},
        ]
        self.hospitals_df = pd.DataFrame(dummy_data)
        
        # Create documents
        self.documents = []
        for idx, row in self.hospitals_df.iterrows():
            text = f"{row['hospital name']} {row['city']} {row['address']}"
            self.documents.append(text)
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        print("Created dummy TF-IDF index")
    
    def search_hospitals(self, query: str, k: int = 3):
        """
        Search for hospitals based on query using TF-IDF similarity.
        
        Args:
            query: User's search query
            k: Number of top results to return (default: 3)
            
        Returns:
            List of dictionaries containing hospital information
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []
        
        try:
            # Transform query using the same vectorizer
            query_vec = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:k]
            
            # Format results
            hospitals = []
            for idx in top_indices:
                if idx < len(self.hospitals_df):
                    row = self.hospitals_df.iloc[idx]
                    hospital_info = {
                        "name": row['hospital name'],
                        "address": row['address'],
                        "city": row['city'],
                        "full_text": self.documents[idx],
                        "score": float(similarities[idx])
                    }
                    hospitals.append(hospital_info)
            
            return hospitals
            
        except Exception as e:
            print(f"Error searching hospitals: {e}")
            return []
    
    def search_by_name_and_city(self, name: str, city: str = None):
        """
        Search for a specific hospital by name and optionally city.
        
        Args:
            name: Hospital name to search for
            city: Optional city to narrow down search
            
        Returns:
            List of matching hospitals
        """
        if self.hospitals_df is None:
            return []
        
        try:
            # Filter by name (case-insensitive partial match)
            mask = self.hospitals_df['hospital name'].str.lower().str.contains(name.lower(), na=False)
            
            # If city is provided, add city filter
            if city:
                mask = mask & self.hospitals_df['city'].str.lower().str.contains(city.lower(), na=False)
            
            results = self.hospitals_df[mask]
            
            # Convert to list of dictionaries
            hospitals = []
            for _, row in results.iterrows():
                hospital_info = {
                    "name": row['hospital name'],
                    "address": row['address'],
                    "city": row['city']
                }
                hospitals.append(hospital_info)
            
            return hospitals
            
        except Exception as e:
            print(f"Error in exact search: {e}")
            return []


# Initialize the RAG engine (singleton pattern)
rag_engine = None

def get_rag_engine():
    """Get or create the RAG engine instance."""
    global rag_engine
    if rag_engine is None:
        rag_engine = HospitalRAG()
    return rag_engine
