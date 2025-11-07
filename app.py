from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import random
import datetime
import re
from vector_db import ITUVectorDatabase
from course_db import CourseDatabase
from rag_pipeline import RAGPipeline
import os
import openai
from typing import List, Dict

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class Chatbot:
    def __init__(self):
        self.conversation_history = []
        self.vector_db = None
        self.course_db = None
        self.rag_pipeline = None
        self.openai_client = None
        self.load_databases()
        self.setup_openai()
        self.initialize_rag_pipeline()
    
    def load_databases(self):
        """Load both vector and course databases"""
        # Load vector database
        try:
            if os.path.exists("itu_vector_index.faiss") and os.path.exists("itu_metadata.pkl"):
                self.vector_db = ITUVectorDatabase()
                self.vector_db.load_database()
                print("✅ Vector database loaded successfully")
            else:
                print("⚠️ Vector database not found. Run scraper and vector_db scripts first.")
        except Exception as e:
            print(f"❌ Error loading vector database: {e}")
        
        # Load course database
        try:
            self.course_db = CourseDatabase()
            if self.course_db.db_path:
                print("✅ Course database loaded successfully")
            else:
                print("⚠️ Course database not found. Some course queries may not work.")
        except Exception as e:
            print(f"❌ Error loading course database: {e}")
    
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
    
    def initialize_rag_pipeline(self):
        """Initialize the RAG pipeline with loaded databases"""
        try:
            self.rag_pipeline = RAGPipeline(
                vector_db=self.vector_db,
                course_db=self.course_db,
                openai_client=self.openai_client
            )
            print("✅ RAG pipeline initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing RAG pipeline: {e}")
    
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
                for i, result in enumerate(search_results, start=1):
                    title = result.get('title') or result.get('doc_id') or f'Source {i}'
                    url = result.get('url', '')
                    text = result.get('text', '')
                    context_parts.append(f"Source {i}: {title}\nURL: {url}\nContent: {text}\n")

                context = "\n\n".join(context_parts)

                model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
                temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
                max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '500'))

                system_instructions = (
                    "You are an assistant for exchange students at the IT University of Copenhagen (ITU). "
                    "Answer using ONLY the information provided in the Context. Do not hallucinate. "
                    "If the answer cannot be found in the Context, reply: 'I couldn't find that in the available ITU data.'" 
                    "When reporting course information, use the following compact course-card format for each course: "
                    "Title (Course code) - X ECTS - Semester - Language - Instructors (if available). "
                    "After each asserted fact, include a short citation using the source index or URL from the Context, e.g. (Source 1) or (URL). "
                )

                user_prompt = f"User question: {user_message}\n\nContext:\n{context}\n\nPlease answer concisely and cite sources as described above."

                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
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
        """Generate a response using the RAG pipeline"""
        message_lower = user_message.lower()
        
        # Store the conversation
        self.conversation_history.append({
            'user': user_message,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Handle greetings and simple queries first (match whole words to avoid false positives like 'which')
        import re
        if re.search(r"\b(hello|hi|hey|good morning|good afternoon|good evening)\b", message_lower):
            return "Hello! I'm the ITU Chatbot, here to help exchange students with questions about courses, programs, and ITU services. How can I assist you today?"
        
        if 'help' in message_lower:
            return "I'm here to help! You can ask me about:\n- ITU courses (codes, ECTS, semesters, instructors)\n- Exchange student information\n- Admission and application processes\n- Campus facilities and services\n- General ITU information\n\nWhat would you like to know?"
        
        if any(phrase in message_lower for phrase in ['how are you', 'how do you do', 'how\'s it going']):
            return "I'm doing great, thank you for asking! I'm here and ready to help with any questions about ITU. What can I help you with?"
        
        if 'name' in message_lower and ('what' in message_lower or 'your' in message_lower):
            return "I'm the ITU Chatbot! I'm designed to help exchange students and prospective students with questions about ITU courses, programs, and services. What can I help you with?"
        
        if any(phrase in message_lower for phrase in ['thank', 'thanks', 'appreciate']):
            return "You're very welcome! I'm happy to help. Is there anything else you'd like to know about ITU?"
        
        if any(phrase in message_lower for phrase in ['bye', 'goodbye', 'see you', 'farewell']):
            return "Goodbye! It was nice chatting with you. Feel free to come back anytime if you have more questions about ITU!"
        
        # Use RAG pipeline for substantive queries
        if self.rag_pipeline:
            try:
                # Retrieve relevant context
                context = self.rag_pipeline.retrieve(user_message, sql_k=5, vector_k=3)
                
                # Generate response using RAG
                response = self.rag_pipeline.generate_response(
                    user_message,
                    context,
                    use_llm=True
                )
                
                if response:
                    return response
            except Exception as e:
                print(f"Error in RAG pipeline: {e}")
                # Fall through to fallback responses
        
        # Fallback: Try old vector search method if RAG fails
        knowledge_results = self.search_knowledge_base(user_message, k=3)
        if knowledge_results:
            llm_response = self.generate_llm_response(user_message, knowledge_results)
            if llm_response:
                return llm_response
        
        # Final fallback responses
        if any(word in message_lower for word in ['time', 'date', 'clock']):
            now = datetime.datetime.now()
            return f"The current time is {now.strftime('%H:%M:%S')} and the date is {now.strftime('%B %d, %Y')}."
        
        if 'weather' in message_lower:
            return "I don't have access to real-time weather data, but I'd recommend checking a weather app or website for the most current conditions in your area."
        
        # Default response
        return "I'm here to help with questions about ITU courses, programs, and services. Could you please rephrase your question or provide more details? For example:\n- 'What AI courses are available for exchange students?'\n- 'Tell me about admission requirements'\n- 'Which courses are offered in Spring 2026?'"

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
        'openai_available': chatbot.openai_client is not None
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
    print("🤖 ITU Chatbot starting...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔧 API endpoints available at: http://localhost:5000/api/")
    app.run(debug=True, host='0.0.0.0', port=5000)
