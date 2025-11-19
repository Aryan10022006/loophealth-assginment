import os
from dotenv import load_dotenv
from rag_engine import get_rag_engine
import re
import ollama

load_dotenv()

class LoopAIAgent:
    def __init__(self):
        """Initialize the Loop AI Agent with Ollama."""
        self.rag_engine = get_rag_engine()
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama2')
        self.system_prompt = self._create_system_prompt()
        
        # Verify Ollama connection
        try:
            ollama.list()
            print(f"✅ Connected to Ollama at {self.ollama_host}")
            print(f"✅ Using model: {self.ollama_model}")
        except Exception as e:
            print(f"⚠️ Warning: Could not connect to Ollama: {e}")
            print("Please ensure Ollama is running: ollama serve")
    
    def _create_system_prompt(self):
        """Create the system prompt for Loop AI."""
        return """You are "Loop AI", a helpful and friendly hospital network assistant for Loop Health.

Your PRIMARY ROLE:
- Help users find hospitals in the Loop Health network
- Provide accurate information about hospital names, addresses, and locations
- Answer queries about specific hospitals and their availability in the network

CRITICAL RULES:
1. ONLY answer hospital-related queries (finding hospitals, confirming if a hospital is in network, hospital addresses, locations)
2. If asked ANYTHING else (weather, coding, general knowledge, math, jokes, etc.), respond EXACTLY with:
   "I'm sorry, I can't help with that. I am forwarding this to a human agent."
   Then STOP. Do not provide any additional information.

3. If multiple hospitals match a generic query (e.g., "Apollo Hospital"), ask clarifying questions:
   "I found several hospitals with that name. In which city are you looking for [Hospital Name]?"

4. Keep answers CONCISE and NATURAL for voice conversation (2-3 sentences maximum)
5. Be friendly and professional
6. If a hospital is not found in the provided context, say: "I couldn't find that hospital in our network. Could you provide more details like the city or full hospital name?"

CONTEXT USAGE:
- You will be provided with relevant hospital information from the database
- Use this context to answer the user's question accurately
- Always mention the city when listing hospitals

Remember: You are ONLY a hospital assistant. Politely decline any non-hospital queries."""

    def _is_hospital_related(self, query: str) -> bool:
        """
        Check if the query is hospital-related.
        
        Args:
            query: User's query
            
        Returns:
            True if hospital-related, False otherwise
        """
        # Keywords that indicate hospital-related queries
        hospital_keywords = [
            'hospital', 'clinic', 'medical', 'health', 'doctor', 'healthcare',
            'network', 'manipal', 'apollo', 'fortis', 'bangalore', 'delhi',
            'location', 'address', 'find', 'near', 'around', 'city',
            'confirm', 'check', 'available', 'list', 'tell me'
        ]
        
        query_lower = query.lower()
        
        # Check if any hospital keyword is in the query
        for keyword in hospital_keywords:
            if keyword in query_lower:
                return True
        
        # If no keywords found, it's likely not hospital-related
        return False
    
    def _extract_city_from_query(self, query: str) -> str:
        """Extract city name from query if present."""
        query_lower = query.lower()
        
        # Common cities
        cities = ['bangalore', 'bengaluru', 'delhi', 'mumbai', 'chennai', 'hyderabad', 
                  'pune', 'kolkata', 'gurugram', 'gurgaon', 'noida', 'faridabad']
        
        for city in cities:
            if city in query_lower:
                return city.capitalize()
        
        return None
    
    def _extract_hospital_name(self, query: str) -> str:
        """Extract hospital name from query."""
        query_lower = query.lower()
        
        # Common hospital names
        hospital_names = ['manipal', 'apollo', 'fortis', 'max', 'medanta', 
                         'artemis', 'columbia', 'cloudnine', 'motherhood']
        
        for name in hospital_names:
            if name in query_lower:
                return name.capitalize()
        
        return None
    
    def process_query(self, user_query: str, is_first_message: bool = False) -> str:
        """
        Process user query and generate response.
        
        Args:
            user_query: User's question
            is_first_message: Whether this is the first message (for introduction)
            
        Returns:
            AI response text
        """
        try:
            # Add introduction if first message
            if is_first_message:
                intro = "Hello! I'm Loop AI, your hospital network assistant. "
            else:
                intro = ""
            
            # Check if query is hospital-related
            if not self._is_hospital_related(user_query):
                return "I'm sorry, I can't help with that. I am forwarding this to a human agent."
            
            # Check for specific hospital confirmation query
            hospital_name = self._extract_hospital_name(user_query)
            city = self._extract_city_from_query(user_query)
            
            if ('confirm' in user_query.lower() or 'check' in user_query.lower()) and hospital_name:
                # Use exact search for confirmation queries
                results = self.rag_engine.search_by_name_and_city(hospital_name, city)
            else:
                # Use semantic search for general queries
                results = self.rag_engine.search_hospitals(user_query, k=3)
            
            # Create CONCISE context from search results (RAG - only relevant data)
            if results:
                # Limit to top 3 results for concise voice response
                top_results = results[:3]
                context = f"Found {len(top_results)} relevant hospital(s):\\n\\n"
                for i, hospital in enumerate(top_results, 1):
                    context += f"{i}. {hospital['name']} - {hospital['city']}\\n"
                    context += f"   Address: {hospital['address']}\\n"
            else:
                context = "No hospitals found matching this query in our network database."
            
            # Create prompt for Ollama
            prompt = f"Context from hospital database:\n{context}\n\nUser query: {user_query}\n\nProvide a concise, natural response suitable for voice conversation (2-3 sentences max)."
            
            # Get response from Ollama
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': prompt}
                ]
            )
            answer = response['message']['content'].strip()
            
            return intro + answer
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return "I'm having trouble processing your request. Please try again."


# Initialize the agent (singleton pattern)
agent = None

def get_agent():
    """Get or create the agent instance."""
    global agent
    if agent is None:
        agent = LoopAIAgent()
    return agent
