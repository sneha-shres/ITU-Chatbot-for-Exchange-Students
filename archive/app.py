from flask import Flask, render_template, request, jsonify
# Load environment variables from .env (if present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; if not installed the app still works using exported env vars
    pass
from flask_cors import CORS
import random
import datetime
import re
import logging
from src.database.vector_db import ITUVectorDatabase
from src.database.course_db import CourseDatabase
from src.core.rag_pipeline import RAGPipeline  # Use the working pipeline
import os
import requests
from typing import List, Dict

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class Chatbot:
    """Chatbot for answering ITU questions using RAG pipeline."""
    
    def __init__(self):
        self.conversation_history = []
        self.vector_db = None
        self.course_db = None
        self.rag_pipeline = None
        self.ollama_url = os.getenv('OLLAMA_URL')
        self.ollama_api_key = os.getenv('OLLAMA_API_KEY')
        self.load_databases()
        self.setup_ollama()
        self.initialize_rag_pipeline()
    
    def load_databases(self):
        """Load both vector and course databases."""
        try:
            index_path = os.path.join('data', 'vectors', 'itu_vector_index.faiss')
            metadata_path = os.path.join('data', 'vectors', 'itu_metadata.pkl')
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.vector_db = ITUVectorDatabase()
                self.vector_db.load_database(index_path=index_path, metadata_path=metadata_path)
                print("✅ Vector database loaded successfully")
            else:
                print("⚠️ Vector database not found.")
        except Exception as e:
            print(f"❌ Error loading vector database: {e}")
        
        try:
            # Prefer explicit data path for course DB after reorg
            course_db_path = os.path.join('data', 'courses', 'courses.db')
            self.course_db = CourseDatabase(db_path=course_db_path)
            if self.course_db.db_path:
                print("✅ Course database loaded successfully")
            else:
                print("⚠️ Course database not found.")
        except Exception as e:
            print(f"❌ Error loading course database: {e}")
    
    def setup_ollama(self):
        """Configure Ollama as the LLM backend and test connectivity."""
        if not self.ollama_url:
            print("⚠️  OLLAMA_URL not set. LLM responses disabled (will use template fallback).")
            return
        
        try:
            # Test if Ollama server is reachable
            response = requests.get(f"{self.ollama_url.rstrip('/')}/api/tags", timeout=2)
            if response.status_code == 200:
                model = os.getenv('OLLAMA_MODEL', 'unknown')
                print(f"✅ Ollama online at {self.ollama_url}")
                print(f"   Model: {model}")
                print(f"   LLM responses: ENABLED ✨")
            else:
                print(f"⚠️  Ollama not responding (status {response.status_code}). LLM disabled.")
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to Ollama at {self.ollama_url}")
            print(f"   Start Ollama with: ollama serve")
            print(f"   LLM responses: DISABLED (using template fallback)")
        except requests.exceptions.Timeout:
            print(f"⚠️  Ollama connection timeout. LLM responses disabled.")
        except Exception as e:
            print(f"⚠️  Error checking Ollama: {e}. LLM responses may be disabled.")
    
    def initialize_rag_pipeline(self):
        """Initialize the RAG pipeline with loaded databases."""
        try:
            self.rag_pipeline = RAGPipeline(
                vector_db=self.vector_db,
                course_db=self.course_db,
            )
            print("✅ RAG pipeline initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing RAG pipeline: {e}")
    
    def generate_response(self, user_message: str) -> Dict:
        """Generate a response using the RAG pipeline."""
        if not self.rag_pipeline:
            return {"text": "RAG pipeline not initialized.", "llm_used": False}
        
        try:
            # Use RAG pipeline to generate response
            retrieval_result = self.rag_pipeline.retrieve(user_message)
            response = self.rag_pipeline.generate_response(
                user_message,
                retrieval_result,
                use_llm=True
            )
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"text": "An error occurred. Please try again.", "llm_used": False}

# Lazily initialize the chatbot to avoid heavy work at import time
chatbot = None

def _init_chatbot_once():
    """Create the chatbot instance once. Safe to call multiple times."""
    global chatbot
    if chatbot is None:
        chatbot = Chatbot()

# Ensure the chatbot is created before handling requests when running under Flask.
# Flask 3 removed `before_first_request`; use `before_request` which is available
# and idempotently initializes the chatbot on the first actual request.
@app.before_request
def _ensure_chatbot_initialized():
    _init_chatbot_once()

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

        # Normalize to structured response with llm_used flag
        if isinstance(bot_response, dict):
            response_text = bot_response.get('text')
            llm_used = bool(bot_response.get('llm_used', False))
        else:
            response_text = str(bot_response)
            llm_used = False

        return jsonify({
            'response': response_text,
            'llm_used': llm_used,
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
    """Get database statistics for both vector and course databases"""
    stats = {}
    
    # Vector database stats
    if chatbot.vector_db:
        stats['vector_db'] = chatbot.vector_db.get_database_stats()
    else:
        stats['vector_db'] = {'error': 'Vector database not loaded'}
    
    # Course database stats
    if chatbot.course_db and chatbot.course_db.db_path:
        stats['course_db'] = chatbot.course_db.get_database_stats()
    else:
        stats['course_db'] = {'error': 'Course database not loaded'}
    
    # RAG pipeline status
    stats['rag_pipeline'] = {
        'initialized': chatbot.rag_pipeline is not None,
        'ollama_available': bool(getattr(chatbot, 'ollama_url', None))
    }
    
    return jsonify(stats)

@app.route('/api/courses/search', methods=['POST'])
def search_courses():
    """Search courses using the course database"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        course_code = data.get('course_code', '').strip()
        semester = data.get('semester', '').strip()
        level = data.get('level', '').strip()
        offered_exchange = data.get('offered_exchange')
        limit = data.get('limit', 10)
        
        if not chatbot.course_db or not chatbot.course_db.db_path:
            return jsonify({'error': 'Course database not available'}), 404
        
        # Build search parameters
        search_params = {'limit': limit}
        
        if query:
            search_params['query'] = query
        if course_code:
            search_params['course_code'] = course_code
        if semester:
            search_params['semester'] = semester
        if level:
            search_params['level'] = level
        if offered_exchange is not None:
            search_params['offered_exchange'] = bool(offered_exchange)
        
        courses = chatbot.course_db.search_courses(**search_params)
        
        return jsonify({
            'query': query,
            'courses': courses,
            'total_results': len(courses)
        })
    
    except Exception as e:
        return jsonify({'error': f'An error occurred during course search: {str(e)}'}), 500

@app.route('/api/courses/exchange', methods=['GET'])
def get_exchange_courses():
    """Get all courses available for exchange students"""
    try:
        semester = request.args.get('semester', '').strip()
        limit = int(request.args.get('limit', 20))
        
        if not chatbot.course_db or not chatbot.course_db.db_path:
            return jsonify({'error': 'Course database not available'}), 404
        
        courses = chatbot.course_db.get_exchange_courses(
            semester=semester if semester else None,
            limit=limit
        )
        
        return jsonify({
            'semester': semester,
            'courses': courses,
            'total_results': len(courses)
        })
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/rag/classify', methods=['POST'])
def classify_query():
    """Classify a query to determine which data sources to use"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if not chatbot.rag_pipeline:
            return jsonify({'error': 'RAG pipeline not initialized'}), 404
        
        query_type, metadata = chatbot.rag_pipeline.classify_query(query)
        
        return jsonify({
            'query': query,
            'query_type': query_type.value,
            'metadata': metadata
        })
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


@app.route('/api/rag/retrieve', methods=['POST'])
def rag_retrieve():
    """Retrieve merged RAG context for a query (diagnostic endpoint)."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        if not chatbot.rag_pipeline:
            return jsonify({'error': 'RAG pipeline not initialized'}), 404

        # Allow overriding retrieval pagination via request body or environment defaults
        sql_k = data.get('sql_k')
        sql_offset = data.get('sql_offset', 0)
        vector_k = data.get('vector_k')
        vector_offset = data.get('vector_offset', 0)

        # Convert to ints if provided
        try:
            sql_k = int(sql_k) if sql_k is not None else None
        except Exception:
            sql_k = None
        try:
            sql_offset = int(sql_offset)
        except Exception:
            sql_offset = 0
        try:
            vector_k = int(vector_k) if vector_k is not None else None
        except Exception:
            vector_k = None
        try:
            vector_offset = int(vector_offset)
        except Exception:
            vector_offset = 0

        merged = chatbot.rag_pipeline.retrieve(query, sql_k=sql_k, sql_offset=sql_offset, vector_k=vector_k, vector_offset=vector_offset)

        return jsonify({
            'query': query,
            'merged': merged
        })

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 ITU Chatbot starting...")
    print("="*60)
    
    # Initialize chatbot to show startup diagnostics
    _init_chatbot_once()
    
    # Check LLM status
    if chatbot and chatbot.ollama_url:
        print("\n LLM STATUS: ENABLED")
    else:
        print("\n LLM STATUS: DISABLED (template responses will be used)")
    
    print("\n" + "="*60)
    print("Open your browser and go to: http://localhost:5000")
    print("API endpoints available at: http://localhost:5000/api/")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
