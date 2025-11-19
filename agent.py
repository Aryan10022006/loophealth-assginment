"""
Enhanced Loop AI Agent with:
- Conversational memory for follow-ups
- Multi-attempt reasoning with internal prompts
- Full RAG pipeline integration
"""

import os
from dotenv import load_dotenv
from rag_engine import get_rag_engine
import re
import ollama
from typing import List, Dict, Optional

load_dotenv()

class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.history = []
        self.max_history = max_history
        self.context = {}
    
    def add_interaction(self, user_query: str, ai_response: str, hospitals: List[Dict]):
        self.history.append({'user': user_query, 'assistant': ai_response, 'hospitals': hospitals})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        self._update_context(user_query, hospitals)
    
    def _update_context(self, query: str, hospitals: List[Dict]):
        if hospitals:
            self.context['last_cities'] = list(set([h['city'] for h in hospitals]))
            self.context['last_hospital_names'] = [h['name'] for h in hospitals]
        
        query_lower = query.lower()
        if any(w in query_lower for w in ['confirm', 'check', 'is']):
            self.context['last_intent'] = 'confirmation'
        elif any(w in query_lower for w in ['find', 'show', 'list', 'tell']):
            self.context['last_intent'] = 'search'
        elif any(w in query_lower for w in ['more', 'other', 'additional']):
            self.context['last_intent'] = 'followup'
    
    def get_conversation_context(self) -> str:
        if not self.history:
            return ""
        recent = self.history[-3:]
        context_str = "Previous conversation:\\n"
        for interaction in recent:
            context_str += f"User: {interaction['user']}\\nAssistant: {interaction['assistant']}\\n"
        return context_str
    
    def get_last_city(self) -> Optional[str]:
        return self.context.get('last_cities', [None])[0] if 'last_cities' in self.context else None
    
    def clear(self):
        self.history = []
        self.context = {}


class LoopAIAgent:
    def __init__(self):
        self.rag_engine = get_rag_engine()
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2')
        self.memory = ConversationMemory()
        self.session_started = False
        
        self.system_prompt = """You are Loop AI Health Assistant - a professional, empathetic hospital locator.

CORE IDENTITY:
- Name: Loop AI Health Assistant
- Purpose: Help users find hospitals in the Loop Health network
- Tone: Warm, professional, efficient

RULES:
1. ONLY answer hospital/healthcare facility queries
2. For non-hospital topics: "I'm sorry, I can only assist with hospital queries. Let me connect you with a human agent."
3. Use conversation history intelligently for follow-ups
4. Keep responses CONCISE (2-4 sentences for voice clarity)
5. Always mention city names with hospitals
6. NO newlines or special characters in responses
7. Be empathetic - healthcare decisions are important

TOKEN EFFICIENCY:
- Limit response to 60 tokens maximum
- Use bullet points mentally but speak naturally
- Prioritize essential information"""
        
        self.greeting_message = """Hello! I'm Loop AI Health Assistant, your dedicated guide to finding hospitals in the Loop Health network. I can help you locate hospitals by city, verify if specific facilities are in your network, and answer questions about our healthcare partners. How may I assist you today?"""
        
        try:
            models = ollama.list()
            hospital_count = len(self.rag_engine.hospitals_df) if hasattr(self.rag_engine, 'hospitals_df') and self.rag_engine.hospitals_df is not None else 'unknown'
            print(f"✅ Loop AI Health Assistant initialized")
            print(f"   Model: {self.ollama_model}")
            print(f"   RAG Database: {hospital_count} hospitals loaded")
            print(f"   Available Ollama models: {len(models.get('models', []))}")
        except Exception as e:
            print(f"⚠️ Status check issue: {str(e)}")
            print(f"   Agent initialized successfully with {self.ollama_model}")
    
    def _is_hospital_related(self, query: str) -> bool:
        keywords = ['hospital', 'clinic', 'medical', 'health', 'doctor', 'healthcare', 'network', 
                    'manipal', 'apollo', 'fortis', 'bangalore', 'delhi', 'location', 'address',
                    'find', 'near', 'around', 'city', 'confirm', 'check', 'available', 'list', 'tell']
        return any(k in query.lower() for k in keywords)
    
    def _extract_intent(self, query: str) -> Dict:
        query_lower = query.lower()
        intent = {'type': 'search', 'city': None, 'hospital_name': None, 'count': 3}
        
        if any(w in query_lower for w in ['confirm', 'is', 'check', 'verify']):
            intent['type'] = 'confirmation'
        elif any(w in query_lower for w in ['more', 'other', 'additional']):
            intent['type'] = 'followup'
        
        cities = ['bangalore', 'bengaluru', 'delhi', 'mumbai', 'chennai', 'hyderabad', 'pune', 'kolkata']
        for city in cities:
            if city in query_lower:
                intent['city'] = city.capitalize()
                break
        
        if not intent['city'] and intent['type'] == 'followup':
            intent['city'] = self.memory.get_last_city()
        
        hospitals = ['manipal', 'apollo', 'fortis', 'max', 'medanta', 'artemis']
        for hosp in hospitals:
            if hosp in query_lower:
                intent['hospital_name'] = hosp
                break
        
        numbers = re.findall(r'\\b(\\d+|three|five|ten)\\b', query_lower)
        if numbers:
            num_map = {'three': 3, 'five': 5, 'ten': 10}
            intent['count'] = num_map.get(numbers[0], int(numbers[0])) if numbers[0].isdigit() else num_map.get(numbers[0], 3)
        
        return intent
    
    def process_query(self, user_query: str, is_first_message: bool = False) -> str:
        try:
            # Handle first interaction or greeting
            if is_first_message or not self.session_started:
                self.session_started = True
                if user_query.strip().lower() in ['hi', 'hello', 'hey', 'start', 'help']:
                    return self.greeting_message
                # For first real query, add brief intro
                intro = "Hello! I'm Loop AI Health Assistant. "
            else:
                intro = ""
            
            if not self._is_hospital_related(user_query):
                return "I'm sorry, I can only assist with hospital queries. Let me connect you with a human agent."
            
            print("\\n" + "="*50)
            print("🧠 REASONING PIPELINE")
            print("="*50)
            
            # Step 1: Intent analysis
            print("\\n📋 Step 1: Intent Analysis")
            intent = self._extract_intent(user_query)
            print(f"Intent: {intent['type']}, City: {intent.get('city')}, Hospital: {intent.get('hospital_name')}")
            
            # Step 2: RAG retrieval
            print("\\n🔍 Step 2: RAG Retrieval")
            if intent['type'] == 'confirmation' and intent['hospital_name']:
                hospitals = self.rag_engine.search_by_name_and_city(intent['hospital_name'], intent['city'], k=intent['count'])
            else:
                hospitals = self.rag_engine.search_hospitals(user_query, k=intent['count'])
            
            print(f"Retrieved {len(hospitals)} hospitals")
            
            # Step 3: Response generation
            print("\\n💬 Step 3: Response Generation")
            if hospitals:
                context = f"Found {len(hospitals)} hospital(s):\\n"
                for i, h in enumerate(hospitals[:3], 1):
                    context += f"{i}. {h['name']} in {h['city']}\\n   Address: {h['address']}\\n"
            else:
                context = "No hospitals found."
            
            conversation_context = self.memory.get_conversation_context()
            prompt = f"Context: {context}\\n{conversation_context}\\nUser: {user_query}\\n\\nProvide a concise voice response (2-3 sentences, NO newlines)."
            
            # Multi-attempt reasoning with token optimization
            for attempt in range(1, 4):
                try:
                    print(f"🤔 LLM Attempt {attempt}/3")
                    response = ollama.chat(
                        model=self.ollama_model,
                        messages=[
                            {'role': 'system', 'content': self.system_prompt},
                            {'role': 'user', 'content': prompt}
                        ],
                        options={
                            'temperature': 0.7,
                            'num_predict': 100,  # Token limit for efficiency
                            'top_p': 0.9,
                            'repeat_penalty': 1.1
                        }
                    )
                    answer = response['message']['content'].strip().replace('\\n', ' ').replace('\\r', ' ')
                    
                    # Validate response quality
                    if 20 < len(answer) < 500 and not answer.startswith('I apologize'):
                        tokens_used = response.get('eval_count', 0)
                        print(f"✅ Generated response (attempt {attempt}, ~{tokens_used} tokens)")
                        self.memory.add_interaction(user_query, answer, hospitals)
                        print("="*50 + "\\n")
                        return intro + answer
                    else:
                        print(f"⚠️ Response quality check failed (attempt {attempt})")
                except Exception as e:
                    print(f"⚠️ Attempt {attempt} failed: {e}")
                    if attempt == 3:
                        raise
            
            # Fallback
            if hospitals:
                answer = f"I found {len(hospitals)} hospitals: {', '.join([h['name'] for h in hospitals[:3]])}."
            else:
                answer = "I couldn't find that hospital. Could you provide more details?"
            
            self.memory.add_interaction(user_query, answer, hospitals)
            print("="*50 + "\\n")
            return intro + answer
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return "I'm having trouble processing your request. Please try again."


agent = None

def get_agent():
    global agent
    if agent is None:
        agent = LoopAIAgent()
    return agent
