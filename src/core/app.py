"""
ITU Chatbot Flask Application

A web-based chatbot for answering IT University of Copenhagen (ITU) student questions
using a Retrieval-Augmented Generation (RAG) pipeline with:
- Vector similarity search over ITU documentation
- LLM generation via Ollama for natural language responses
- Reasoning layer for complex queries

"""

from flask import Flask, render_template, request, jsonify
import os
import re
import logging
import datetime
import requests
from typing import List, Dict

try:
    from dotenv import load_dotenv
    from pathlib import Path
    # Load .env from config/ directory
    env_path = Path(__file__).parent.parent.parent / "config" / ".env"
    load_dotenv(dotenv_path=env_path)
except Exception:
    pass

from flask_cors import CORS

# Project modules
from database.vector_db import ITUVectorDatabase
from core.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
repo_root = Path(__file__).parent.parent.parent
app = Flask(
    __name__,
    template_folder=str(repo_root / "templates"),
    static_folder=str(repo_root / "static")
)
CORS(app)  # Enable CORS for all routes

class Chatbot:
    
    def __init__(self):
        self.conversation_history = []
        self.vector_db = None
        self.rag_pipeline = None
        self.ollama_url = os.getenv('OLLAMA_URL')
        self.ollama_api_key = os.getenv('OLLAMA_API_KEY')
        
        # Load data sources
        self.load_databases()
        self.setup_ollama()
        self.initialize_rag_pipeline()
    
    def load_databases(self):
        """Load vector database."""
        try:
            index_path = os.path.join('data', 'vectors', 'itu_vector_index.faiss')
            metadata_path = os.path.join('data', 'vectors', 'itu_metadata.pkl')
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.vector_db = ITUVectorDatabase()
                self.vector_db.load_database(index_path=index_path, metadata_path=metadata_path)
                print("Vector database loaded successfully")
            else:
                print("Vector database not found.")
        except Exception as e:
            print(f"Error loading vector database: {e}")
    
    def setup_ollama(self):
        if not self.ollama_url:
            print("OLLAMA_URL not set. LLM responses disabled (will use template fallback).")
            return
        
        try:
            response = requests.get(f"{self.ollama_url.rstrip('/')}/api/tags", timeout=2)
            if response.status_code == 200:
                model = os.getenv('OLLAMA_MODEL', 'unknown')
                print(f"Ollama online at {self.ollama_url}")
                print(f"   Model: {model}")
                print(f"   LLM responses: ENABLED ✨")
            else:
                print(f"Ollama not responding (status {response.status_code}). LLM disabled.")
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to Ollama at {self.ollama_url}")
            print(f" Start Ollama with: ollama serve")
            print(f"LLM responses: DISABLED (using template fallback)")
        except requests.exceptions.Timeout:
            print(f"Ollama connection timeout. LLM responses disabled.")
        except Exception as e:
            print(f"Error checking Ollama: {e}. LLM responses may be disabled.")
    
    def initialize_rag_pipeline(self):
        try:
            self.rag_pipeline = RAGPipeline(
                vector_db=self.vector_db,
            )
            print("RAG pipeline initialized successfully")
        except Exception as e:
            print(f"Error initializing RAG pipeline: {e}")
    
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

chatbot = None


def _init_chatbot_once():
    global chatbot
    if chatbot is None:
        chatbot = Chatbot()


@app.before_request
def _ensure_chatbot_initialized():
    _init_chatbot_once()



@app.route('/')
def index():
    """Serve the main chatbot UI page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():

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
        
        if not chatbot.vector_db:
            return jsonify({'error': 'Vector database not available'}), 404
        
        results = chatbot.vector_db.search(query, k=5)
        
        return jsonify({
            'query': query,
            'results': results,
            'total_results': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': 'An error occurred during search'}), 500

@app.route('/api/database/stats')
def get_database_stats():
    """Get database statistics for vector database.
    """
    stats = {}
    
    # Vector database stats
    if chatbot.vector_db:
        stats['vector_db'] = chatbot.vector_db.get_database_stats()
    else:
        stats['vector_db'] = {'error': 'Vector database not loaded'}
    
    # RAG pipeline status
    stats['rag_pipeline'] = {
        'initialized': chatbot.rag_pipeline is not None,
        'ollama_available': bool(getattr(chatbot, 'ollama_url', None))
    }
    
    return jsonify(stats)

@app.route('/api/rag/classify', methods=['POST'])
def classify_query():
    """Classify a query to determine which data sources to use.
    """
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
    """Retrieve merged RAG context for a query (diagnostic endpoint).

    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        if not chatbot.rag_pipeline:
            return jsonify({'error': 'RAG pipeline not initialized'}), 404

        # Allow overriding retrieval pagination via request body or environment defaults
        vector_k = data.get('vector_k')
        vector_offset = data.get('vector_offset', 0)

        # Convert to ints if provided
        try:
            vector_k = int(vector_k) if vector_k is not None else None
        except Exception:
            vector_k = None
        try:
            vector_offset = int(vector_offset)
        except Exception:
            vector_offset = 0

        merged = chatbot.rag_pipeline.retrieve(query, vector_k=vector_k, vector_offset=vector_offset)

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
