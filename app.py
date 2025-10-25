from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import random
import datetime
import re
from vector_db import ITUVectorDatabase
import os
import openai
from typing import List, Dict

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class Chatbot:
    def __init__(self):
        self.conversation_history = []
        self.vector_db = None
        self.openai_client = None
        self.load_vector_database()
        self.setup_openai()
    
    def load_vector_database(self):
        """Load the vector database if it exists"""
        try:
            if os.path.exists("itu_vector_index.faiss") and os.path.exists("itu_metadata.pkl"):
                self.vector_db = ITUVectorDatabase()
                self.vector_db.load_database()
                print("✅ Vector database loaded successfully")
            else:
                print("⚠️ Vector database not found. Run scraper and vector_db scripts first.")
        except Exception as e:
            print(f"❌ Error loading vector database: {e}")
    
    def setup_openai(self):
        """Setup OpenAI client"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                print("✅ OpenAI client initialized")
            else:
                print("⚠️ OPENAI_API_KEY not found. LLM responses will be disabled.")
        except Exception as e:
            print(f"❌ Error setting up OpenAI: {e}")
    
    def search_knowledge_base(self, query: str, k: int = 3) -> list:
        """Search the ITU knowledge base for relevant information"""
        if not self.vector_db:
            return []
        
        try:
            results = self.vector_db.search(query, k=k)
            return results
        except Exception as e:
            print(f"Error searching knowledge base: {e}")
            return []
    
    def generate_llm_response(self, user_message: str, search_results: List[Dict]) -> str:
        """Generate natural language response using LLM or local processing"""
        if not search_results:
            return None
        
        # Try OpenAI first if available
        if self.openai_client:
            try:
                # Prepare context from search results
                context_parts = []
                for result in search_results:
                    context_parts.append(f"Title: {result['title']}\nContent: {result['text']}\nURL: {result['url']}")
                
                context = "\n\n".join(context_parts)
                
                prompt = f"""You are a helpful ITU (IT University of Copenhagen) assistant. Based on the following information from the ITU website, provide a natural, conversational response to the user's question.

User's question: {user_message}

ITU Website Content:
{context}

Instructions:
- Answer the user's question in a friendly, conversational tone
- Use the provided ITU content to give accurate information
- If the content doesn't fully answer the question, acknowledge this and provide what information is available
- Keep the response concise but informative
- Don't repeat the raw content - synthesize it into a natural response
- If there are specific procedures or deadlines mentioned, highlight them clearly

Response:"""

                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for ITU students and prospective students."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"Error generating OpenAI response: {e}")
        
        # Fallback: Generate natural response locally
        return self.generate_local_response(user_message, search_results)
    
    def generate_local_response(self, user_message: str, search_results: List[Dict]) -> str:
        """Generate natural response using local text processing"""
        if not search_results:
            return None
        
        # Get the most relevant result
        best_result = search_results[0]
        title = best_result['title']
        content = best_result['text']
        url = best_result['url']
        
        # Extract key information patterns
        message_lower = user_message.lower()
        
        # Generate contextual responses based on content type
        if any(word in message_lower for word in ['exam', 'examination', 'test', 'assessment']):
            if 'illness' in message_lower or 'sick' in message_lower:
                return f"Regarding illness during exams: {content[:400]}...\n\nYou can find more detailed information about exam procedures and medical certificates on the ITU website."
            elif 're-exam' in message_lower or 'retake' in message_lower:
                return f"About re-examinations: {content[:400]}...\n\nFor specific re-exam procedures and deadlines, please check the complete guidelines on the ITU website."
            else:
                return f"Here's information about exams at ITU: {content[:400]}...\n\nFor complete exam guidelines and procedures, visit the ITU student portal."
        
        elif any(word in message_lower for word in ['program', 'course', 'study', 'curriculum']):
            return f"Here's what I found about ITU programs: {content[:400]}...\n\nFor detailed program information and course descriptions, check the ITU website or contact the study administration."
        
        elif any(word in message_lower for word in ['application', 'admission', 'apply', 'enroll']):
            return f"Regarding admissions: {content[:400]}...\n\nFor complete admission requirements and application procedures, please visit the ITU admissions page."
        
        elif any(word in message_lower for word in ['project', 'thesis', 'bachelor', 'master']):
            return f"About project work: {content[:400]}...\n\nFor detailed project guidelines and supervision information, consult the ITU project handbook."
        
        else:
            # Generic response
            return f"Based on ITU information: {content[:300]}...\n\nThis information comes from the ITU website. For more details, you can visit the source page or contact ITU directly."
    
    def generate_response(self, user_message):
        """Generate a response based on the user's message"""
        message_lower = user_message.lower()
        
        # Store the conversation
        self.conversation_history.append({
            'user': user_message,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # First, try to find relevant information in the knowledge base
        knowledge_results = self.search_knowledge_base(user_message, k=3)
        
        if knowledge_results:
            # Try to generate LLM response first
            llm_response = self.generate_llm_response(user_message, knowledge_results)
            if llm_response:
                return llm_response
            
            # Fallback to structured response if LLM fails
            response_parts = []
            for i, result in enumerate(knowledge_results, 1):
                if result['similarity_score'] > 0.2:  # Lower threshold to get more results
                    similarity_pct = result['similarity_score'] * 100
                    response_parts.append(f"{i}. **{result['title']}** (Relevance: {similarity_pct:.1f}%)\n   {result['text'][:300]}...")
            
            if response_parts:
                sources = "\n".join([f"• {result['url']}" for result in knowledge_results[:2]])
                return f"Based on ITU website content:\n\n" + "\n\n".join(response_parts) + f"\n\n**Sources:**\n{sources}"
        
        # Fall back to simple response patterns if no relevant knowledge found
        if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return "Hello! Nice to meet you. How can I assist you today?"
        
        if 'help' in message_lower:
            return "I'm here to help! You can ask me questions, and I'll do my best to provide useful information. What would you like to know?"
        
        if any(phrase in message_lower for phrase in ['how are you', 'how do you do', 'how\'s it going']):
            return "I'm doing great, thank you for asking! I'm here and ready to help with any questions you might have."
        
        if 'name' in message_lower and ('what' in message_lower or 'your' in message_lower):
            return "I'm the ITU Chatbot! I'm designed to help answer questions and provide assistance. What can I help you with?"
        
        if any(phrase in message_lower for phrase in ['thank', 'thanks', 'appreciate']):
            return "You're very welcome! I'm happy to help. Is there anything else you'd like to know?"
        
        if any(phrase in message_lower for phrase in ['bye', 'goodbye', 'see you', 'farewell']):
            return "Goodbye! It was nice chatting with you. Feel free to come back anytime if you have more questions!"
        
        if any(word in message_lower for word in ['time', 'date', 'clock']):
            now = datetime.datetime.now()
            return f"The current time is {now.strftime('%H:%M:%S')} and the date is {now.strftime('%B %d, %Y')}."
        
        if 'weather' in message_lower:
            return "I don't have access to real-time weather data, but I'd recommend checking a weather app or website for the most current conditions in your area."
        
        if any(word in message_lower for word in ['joke', 'funny', 'laugh', 'humor']):
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything!",
                "Why did the chatbot go to therapy? It had too many issues to process!",
                "What do you call a fake noodle? An impasta!",
                "Why don't eggs tell jokes? They'd crack each other up!",
                "What do you call a bear with no teeth? A gummy bear!",
                "Why did the Python programmer prefer dark mode? Because light attracts bugs!"
            ]
            return random.choice(jokes)
        
        if 'python' in message_lower:
            return "Python is a great programming language! It's known for its simplicity and readability. Are you working on any Python projects?"
        
        if 'itu' in message_lower:
            return "ITU (International Telecommunication Union) is a specialized agency of the United Nations. Are you interested in telecommunications or IT standards?"
        
        if any(word in message_lower for word in ['code', 'programming', 'coding', 'develop']):
            return "Programming is fascinating! What programming language are you working with? I'd be happy to help with coding questions."
        
        if any(word in message_lower for word in ['ai', 'artificial intelligence', 'machine learning']):
            return "Artificial Intelligence is an exciting field! I'm a simple chatbot, but AI has many applications in various industries. What aspect of AI interests you?"
        
        # Default responses
        default_responses = [
            "That's an interesting question! Let me think about that for a moment.",
            "I understand what you're asking. Could you provide a bit more detail?",
            "Thanks for sharing that with me. What would you like to know more about?",
            "I'm processing your message. Could you rephrase that in a different way?",
            "That's a great point! I'd be happy to discuss this further with you.",
            "I appreciate your question. Let me help you with that.",
            "Interesting! I'd like to learn more about your perspective on this.",
            "I'm here to help! Could you tell me more about what you're looking for?",
            "That's a thoughtful question. Let me see how I can assist you with that.",
            "I'm listening! What else would you like to explore?"
        ]
        
        return random.choice(default_responses)

# Initialize the chatbot
chatbot = Chatbot()

@app.route('/')
def index():
    """Serve the main chatbot page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        if len(user_message) > 500:
            return jsonify({'error': 'Message too long (max 500 characters)'}), 400
        
        # Generate bot response
        bot_response = chatbot.generate_response(user_message)
        
        return jsonify({
            'response': bot_response,
            'timestamp': datetime.datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': 'An error occurred processing your message'}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'chatbot': 'ITU Chatbot v1.0'
    })

@app.route('/api/history')
def get_history():
    """Get conversation history"""
    return jsonify({
        'history': chatbot.conversation_history[-10:],  # Last 10 messages
        'total_messages': len(chatbot.conversation_history)
    })

@app.route('/api/search', methods=['POST'])
def search_knowledge():
    """Search the knowledge base"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        results = chatbot.search_knowledge_base(query, k=5)
        
        return jsonify({
            'query': query,
            'results': results,
            'total_results': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': 'An error occurred during search'}), 500

@app.route('/api/database/stats')
def get_database_stats():
    """Get vector database statistics"""
    if chatbot.vector_db:
        stats = chatbot.vector_db.get_database_stats()
        return jsonify(stats)
    else:
        return jsonify({'error': 'Vector database not loaded'}), 404

if __name__ == '__main__':
    print("🤖 ITU Chatbot starting...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔧 API endpoints available at: http://localhost:5000/api/")
    app.run(debug=True, host='0.0.0.0', port=5000)
