"""
RAG Pipeline for Chatbot_ITU

Implements a Retrieval-Augmented Generation pipeline that uses
vector similarity search over ITU documentation to provide
comprehensive responses to student queries.
"""

import re
from typing import List, Dict, Optional, Tuple, Literal
from enum import Enum
import logging

import os
from database.vector_db import ITUVectorDatabase
from core.reasoning_layer import CourseCombinationReasoner
import requests
# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)

# Define constants at module level for reusability
STOP_WORDS = {"the", "and", "or", "but", "for", "with", "from", "about", "what", "which", "when", "where", "how", "why"}
# Define stopwords / noisy n-grams
# NOISY_NGRAMS = {
#     "are", "is", "was", "were", "be", "been", "being",
#     "which", "what", "where", "when", "how",
#     "at", "the", "and", "for", "with", "that", "this", "of", "in", "on"
# }
NOISY_NGRAMS = {"language", "ects", "show", "give", "available", "courses", "course", "in", "the", "me", "how", "what", "when", "where", "which", "why", "are", "does", "offer", "itu", "all", "there", "list", "tell"}
COURSE_KEYWORDS = ['course', 'courses', 'ects', 'syllabus', 'course code', 'module', 'curriculum', 'study', 'class']
EXCHANGE_KEYWORDS = ['exchange', 'exchange student', 'exchange students', 'erasmus']
VECTOR_KEYWORDS = ['apply', 'application', 'admission', 'housing', 'deadline', 'campus', 'research', 'life', 'accommodation', 'enroll', 'enrol']
MIN_ECTS_KEYWORDS = ['minimum', 'least', 'lowest', 'fewest', 'smallest']



class QueryType(Enum):
    """Query classification types."""
    VECTOR = "vector"  # General ITU info queries
    REASONING = "reasoning"  # Queries requiring combinatorial reasoning (e.g., course combinations)



class RAGPipeline:
    """Main RAG pipeline orchestrator."""
    
    def __init__(
        self,
        vector_db: ITUVectorDatabase = None,
    ):
        """Initialize the RAG pipeline.
        
        Args:
            vector_db: Initialized ITUVectorDatabase instance
        """
        self.vector_db = vector_db

        # Initialize reasoning layer for combination queries
        self.reasoner = CourseCombinationReasoner(vector_db=vector_db)

        # Ollama support: optional local or remote Ollama server URL (e.g. http://localhost:11434)
        self.ollama_url = os.getenv('OLLAMA_URL')
        self.ollama_api_key = os.getenv('OLLAMA_API_KEY')

        # Configurable defaults from environment
        self.default_vector_k = int(os.getenv('RAG_VECTOR_K', '3'))
        self.max_context_chars = int(os.getenv('RAG_MAX_CONTEXT_CHARS', '1000'))
        self.max_vector_items = int(os.getenv('RAG_MAX_VECTOR_ITEMS', '3'))
    
    def classify_query(self, query: str) -> Tuple[QueryType, Dict]:
        """Classify a query to determine whether to use Vector or Reasoning retrieval.
        Returns: (QueryType, metadata)
        """
        q = (query or "").strip()
        ql = q.lower()

        # Check for combination/reasoning queries (e.g., "which combinations sum to 30 ECTS")
        is_combo_query, combo_ects = self.reasoner.is_combination_query(query)
        
        if is_combo_query and combo_ects is not None:
            logger.debug(f"Detected combination query with target ECTS: {combo_ects}")
            meta = {'reasoning_type': 'combination', 'target_ects': combo_ects}
            
            # Extract filters if present
            if 'autumn' in ql or 'fall' in ql:
                meta['semester'] = 'Autumn'
            elif 'spring' in ql:
                meta['semester'] = 'Spring'
            
            if 'english' in ql:
                meta['language'] = 'English'
            elif 'danish' in ql or 'dansk' in ql:
                meta['language'] = 'Danish'
            
            return QueryType.REASONING, meta

        # Default to VECTOR for all other queries
        logger.debug("Classifying as VECTOR")
        return QueryType.VECTOR, {}
    
    def query_vector(self, query: str, k: int = 5) -> List[Dict]:
        """Query the FAISS vector database.
        
        Args:
            query: User query string
            k: Maximum number of results
            
        Returns:
            List of vector search results
        """
        if not self.vector_db:
            return []
        
        try:
            results = self.vector_db.search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error querying vector database: {e}")
            return []

    def query_vector_paginated(self, query: str, k: int = 5, offset: int = 0) -> List[Dict]:
        """Query vector DB with simple pagination by requesting k+offset and slicing results."""
        if not self.vector_db:
            return []
        try:
            results = self.vector_db.search(query, k=offset + k)
            # slice safely
            return results[offset: offset + k]
        except Exception as e:
            logger.error(f"Error querying vector database (paginated): {e}")
            return []
        
    def merge_results(
        self,
        vector_results: List[Dict],
        query_type: QueryType
    ) -> Dict:
        """Merge vector results into a unified context.
        
        Args:
            vector_results: Results from vector database
            query_type: Type of query
            
        Returns:
            Dictionary with merged context and metadata
        """
        # Apply context truncation to prevent token overflow
        vector_results = self._truncate_context(vector_results, max_items=self.max_vector_items)
        
        merged = {
            "vector_results": vector_results,
            "query_type": query_type.value,
            "total_results": len(vector_results)
        }
        
        # Format context
        if query_type == QueryType.VECTOR:
            merged["primary_source"] = "vector"
            merged["context"] = self._format_vector_context(vector_results)
        elif query_type == QueryType.REASONING:
            merged["primary_source"] = "reasoning"
            merged["context"] = ""  # Will be set by _handle_reasoning_query
        
        return merged
    
    def _truncate_context(self, results: List[Dict], max_items: int = 5) -> List[Dict]:
        """Truncate context to prevent token overflow.
        
        Args:
            results: List of result dictionaries
            max_items: Maximum number of items to keep
            
        Returns:
            Truncated list of results
        """
        if len(results) <= max_items:
            return results
        
        # Keep top N results by score
        return results[:max_items]
    
    def _format_vector_context(self, results: List[Dict]) -> str:
        """Format vector search results into context string.
        
        Args:
            results: List of vector search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            text = result.get("text", "")
            url = result.get("url", "")
            
            context_parts.append(
                f"Source {i}: {title}\n"
                f"Content: {text[:self.max_context_chars]}\n"
                f"URL: {url}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def retrieve(self, query: str, vector_k: int = None, vector_offset: int = 0) -> Dict:
        """Main retrieval function that orchestrates the RAG pipeline.
        
        Args:
            query: User query string
            vector_k: Number of vector results to retrieve
            
        Returns:
            Dictionary with merged results and context
        """

        # Classify query
        query_type, metadata = self.classify_query(query)

        # Resolve defaults
        if vector_k is None:
            vector_k = self.default_vector_k

        # Retrieve from appropriate sources
        vector_results = []

        # Handle reasoning queries separately
        if query_type == QueryType.REASONING:
            return self._handle_reasoning_query(query, metadata)

        # Query vector DB
        if query_type == QueryType.VECTOR:
            vector_results = self.query_vector_paginated(query, k=vector_k, offset=vector_offset)
        
        # Merge results
        merged = self.merge_results(vector_results, query_type)

        # Add pagination totals
        merged_meta = merged.copy()
        try:
            if self.vector_db:
                vd_stats = self.vector_db.get_database_stats()
                merged_meta['vector_total'] = vd_stats.get('total_vectors') if vd_stats else len(self.vector_db.metadata)
        except Exception:
            merged_meta['vector_total'] = len(vector_results)

        return merged_meta

    def _handle_reasoning_query(self, query: str, metadata: Dict) -> Dict:
        """Handle reasoning queries (e.g., course combinations).
        
        Args:
            query: User query string
            metadata: Query metadata with reasoning information
            
        Returns:
            Dictionary with reasoning results formatted for context
        """
        target_ects = metadata.get('target_ects')
        if not target_ects:
            return {
                'sql_results': [],
                'vector_results': [],
                'query_type': 'reasoning',
                'total_results': 0,
                'primary_source': 'reasoning',
                'context': 'Could not determine target ECTS value for combination query.',
                'reasoning_results': None
            }
        
        # Build filters from metadata
        filters = {}
        if metadata.get('language'):
            filters['language'] = metadata['language']
        if metadata.get('semester'):
            filters['semester'] = metadata['semester']
        if metadata.get('level'):
            filters['level'] = metadata['level']
        if metadata.get('exchange_related'):
            filters['offered_exchange'] = metadata['exchange_related']
        
        # Use reasoning layer to find combinations
        reasoning_result = self.reasoner.reason_about_query(query, target_ects, filters)
        
        # Format context for LLM
        context = reasoning_result.get('formatted_context', 'No combinations found.')
        
        return {
            'vector_results': [],
            'query_type': 'reasoning',
            'total_results': len(reasoning_result.get('combinations', [])),
            'primary_source': 'reasoning',
            'context': context,
            'reasoning_results': reasoning_result,
            'target_ects': target_ects,
            'courses_used': reasoning_result.get('courses_used', [])
        }

    
    def generate_response(
        self,
        query: str,
        context: Dict,
        use_llm: bool = True
    ) -> str:
        """Generate a natural language response using the retrieved context.
        
        Args:
            query: User's original query
            context: Merged context dictionary from retrieve()
            use_llm: Whether to use LLM (if available) or fallback
            
        Returns:
            Generated response string
        """
        # If no context, return a helpful fallback
        if not context.get("context"):
            return {"text": "I couldn't find relevant information to answer your question. Could you please rephrase it or provide more details?", "llm_used": False}

        # Try LLM if available and requested. Support Ollama HTTP API.
        if use_llm and self.ollama_url:
            try:
                llm_text = self._generate_llm_response(query, context)
                return {"text": llm_text, "llm_used": True}
            except Exception as e:
                logger.error(f"Error generating LLM response: {e}")
                # Fall through to template-based response

        # Fallback to template-based response
        tpl = self._generate_template_response(query, context)
        return {"text": tpl, "llm_used": False}
    
    def _generate_llm_response(self, query: str, context: Dict) -> str:
        print("generate llm repsonse")
        print(context)
        
        """Generate response using Ollama LLM.
        
        Args:
            query: User query
            context: Merged context dictionary
            
        Returns:
            LLM-generated response
        """
        # Use environment-configurable model parameters (prefer OLLAMA_* vars)
        model = os.getenv('OLLAMA_MODEL', os.getenv('OPENAI_MODEL', 'gpt-oss'))
        temperature = float(os.getenv('OLLAMA_TEMPERATURE', os.getenv('OPENAI_TEMPERATURE', '0.1')))
        max_tokens = int(os.getenv('OLLAMA_MAX_TOKENS', os.getenv('OPENAI_MAX_TOKENS', '800')))

        system_prompt = (
            "You are a concise, strict-format assistant for exchange students at the IT University of Copenhagen (ITU).\n"
            "Important rules (follow exactly):\n"
            "1) Always use ONLY the provided Context. Do NOT invent facts or add information not present in Context. If something is not in Context, reply: 'I couldn't find that in the available ITU data.'\n"
            "2) FORMATTING FOR COURSES:\n"
            "   - When answering about courses, list them as a NUMBERED LIST with each course on its own line\n"
            "   - Format: 'N. Title (Course Code) - X ECTS - Semester'\n"
            "   - Example:\n"
            "     1. Advanced Machine Learning (KSAMLDS2KU) - 7.5 ECTS - Spring 2026\n"
            "     2. Machine Learning (BSMALEA1KU) - 15.0 ECTS - Autumn 2026\n"
            "   - Each course must be on its own line with proper numbering\n"
            "   - If query doesn't have any conditions (e.g. 15 ECTs, English language etc.) and the question is more general example: 'Which courses does itu offeer', use all the courses to answer\n"
            "3) FORMATTING FOR HYBRID QUERIES (courses + general info):\n"
            "   - Start with 'Relevant Courses:' section with numbered list (one course per line, formatted as above)\n"
            "   - Add blank line\n"
            "   - Then 'General Information:' section with facts about the topic (one paragraph)\n"
            "4) FORMATTING FOR GENERAL INFORMATION:\n"
            "   - Use only vector data from Context\n"
            "   - Write concisely in one or two short paragraphs\n"
            "   - Do NOT include citations or source references\n"
            "5) Be concise and well-formatted: clear section headers, proper spacing between sections, numbered lists for courses\n"
            "6) Do not include any intermediate JSON or streaming fragments — output a single clean text response only.\n"
            "7) If the query is unrelated or Context is empty, respond with: 'I couldn't find relevant information to answer your question.'\n"
        )

        user_prompt = f"User Question: {query}\n\nContext:\n{context.get('context', 'No context available')}\n\nProvide a concise, factual answer and include citations as requested."

        # Ollama is the only supported LLM backend now. Require OLLAMA_URL to be set.
        if self.ollama_url:
            try:
                return self._generate_ollama_response(system_prompt, user_prompt, model=model, temperature=temperature, max_tokens=max_tokens)
            except Exception as e:
                logger.error(f"Error calling Ollama API: {e}")
                raise

        # If we reach here, no Ollama URL configured — raise a clear error so callers fall back to templates
        raise RuntimeError("OLLAMA_URL is not configured; cannot call LLM")

    def _generate_ollama_response(self, system_prompt: str, user_prompt: str, model: str = None, temperature: float = 0.1, max_tokens: int = 800) -> str:
        """Call an Ollama server via its HTTP API and return the generated text.

        Expects an environment variable OLLAMA_URL (e.g. http://localhost:11434).
        Optionally uses OLLAMA_API_KEY for authorization if set.
        """
        print("generating ollama epsonsse")
        if not self.ollama_url:
            raise RuntimeError("OLLAMA_URL is not configured")

        payload = {
            # Ollama typically accepts a `model` and `prompt`/`instruction` field; place system+user into the prompt
            "model": model or os.getenv('OLLAMA_MODEL', os.getenv('OPENAI_MODEL', 'gpt-oss')),
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "temperature": float(temperature),
            # max_tokens may not be supported by all Ollama models; include as a hint
            "max_tokens": int(max_tokens)
        }

        headers = {"Content-Type": "application/json"}
        if self.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"

        url = self.ollama_url.rstrip('/') + '/api/generate'
        resp = requests.post(url, json=payload, headers=headers, timeout=30, stream=True)

        # If endpoint returned a non-200 code, try to show a helpful error
        if resp.status_code != 200:
            try:
                body = resp.text
            except Exception:
                body = '<unreadable body>'
            logger.error(f"Ollama returned status {resp.status_code}: {body}")
            raise RuntimeError(f"Ollama API error: {resp.status_code}")

        # The Ollama server may stream incremental JSON objects (SSE-like or newline-delimited JSON).
        # We'll iterate the response lines and accumulate text fragments from common fields
        accumulated = []
        final_text = None

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                line = raw_line.strip()
                # handle SSE 'data: {...}' lines
                if line.startswith('data:'):
                    line = line[len('data:'):].strip()

                # Try to parse JSON from the line
                try:
                    part = None
                    j = None
                    # Some servers send multiple JSON objects per line; try json.loads
                    import json
                    j = json.loads(line)

                    # prefer 'response' then 'output' then nested choices/results
                    # IMPORTANT: ignore 'thinking' fragments (internal chain-of-thought)
                    if isinstance(j, dict):
                        # If server sends chunk with 'response' (final) or 'thinking' (partial)
                        if 'response' in j and isinstance(j['response'], str) and j['response']:
                            part = j['response']
                        elif 'output' in j and isinstance(j['output'], str) and j['output']:
                            part = j['output']
                        elif 'results' in j and isinstance(j['results'], list) and j['results']:
                            first = j['results'][0]
                            if isinstance(first, dict):
                                for key in ('text', 'output'):
                                    if key in first and isinstance(first[key], str) and first[key]:
                                        part = first[key]
                                        break
                            elif isinstance(first, str):
                                part = first
                        elif 'choices' in j and isinstance(j['choices'], list) and j['choices']:
                            ch = j['choices'][0]
                            if isinstance(ch, dict):
                                if 'message' in ch and isinstance(ch['message'], dict) and 'content' in ch['message']:
                                    part = str(ch['message']['content'])
                                else:
                                    for key in ('text', 'output'):
                                        if key in ch and isinstance(ch[key], str):
                                            part = ch[key]
                                            break
                        # If there's a 'done' flag and it's true, mark final_text
                        if j.get('done') is True:
                            # Only append if this chunk contained a non-thinking response/output
                            if part:
                                accumulated.append(part)
                            final_text = ''.join(accumulated).strip()
                            break

                    # Only append selected parts (response/output/text) -- ignore 'thinking' fragments
                    if part:
                        accumulated.append(part)
                        # continue reading until done=True is seen
                        continue
                except Exception:
                    # Not a JSON line -- ignore to avoid leaking chain-of-thought or protocol noise
                    continue

            # After streaming, if we have a final_text from done flag use it
            if final_text is not None:
                return final_text

            # Otherwise, join accumulated parts and return
            if accumulated:
                return ''.join(accumulated).strip()

            # Fallback: return raw text
            return resp.text.strip()
        finally:
            try:
                resp.close()
            except Exception:
                pass
    
    def _generate_template_response(self, query: str, context: Dict) -> str:
        """Generate response using template-based approach (fallback).
        
        Args:
            query: User query
            context: Merged context dictionary
            
        Returns:
            Template-based response
        """
        query_type = context.get("query_type", "vector")
        vector_results = context.get("vector_results", [])
        
        response_parts = []
        
        if query_type == "vector" and vector_results:
            best_result = vector_results[0]
            response_parts.append(best_result.get("text", "")[:self.max_context_chars])
        elif query_type == "reasoning":
            # Use the context from reasoning layer
            reasoning_context = context.get("context", "")
            if reasoning_context:
                response_parts.append(reasoning_context)
        
        if not response_parts:
            return "I couldn't find specific information to answer your question. Could you try rephrasing it?"
        
        return "\n".join(response_parts)

# """
# RAG Pipeline for Chatbot_ITU

# Implements a Retrieval-Augmented Generation pipeline that intelligently combines
# structured SQL course data and unstructured FAISS vector data to provide
# comprehensive responses to student queries.
# """

# import re
# from typing import List, Dict, Optional, Tuple, Literal
# from enum import Enum
# import logging

# import os
# from database.vector_db import ITUVectorDatabase
# from database.course_db import CourseDatabase
# import requests
# # Load environment variables from .env if present
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except Exception:
#     pass

# logger = logging.getLogger(__name__)

# # Define constants at module level for reusability
# STOP_WORDS = {"the", "and", "or", "but", "for", "with", "from", "about", "what", "which", "when", "where", "how", "why", "are", "does", "can", "all", "there", "list", "show", "tell"}
# NOISY_NGRAMS = {"language", "ects", "show", "give", "available", "courses", "course", "in", "the", "me", "how", "what", "when", "where", "which", "why", "are", "does", "offer", "itu", "all", "there", "list", "tell"}
# COURSE_KEYWORDS = ['course', 'courses', 'ects', 'syllabus', 'course code', 'module', 'curriculum', 'study', 'class']
# EXCHANGE_KEYWORDS = ['exchange', 'exchange student', 'exchange students', 'erasmus']
# VECTOR_KEYWORDS = ['apply', 'application', 'admission', 'housing', 'deadline', 'campus', 'research', 'life', 'accommodation', 'enroll', 'enrol']
# MIN_ECTS_KEYWORDS = ['minimum', 'least', 'lowest', 'fewest', 'smallest']


# class QueryType(Enum):
#     """Query classification types."""
#     SQL = "sql"  # Course-specific, structured queries
#     VECTOR = "vector"  # General ITU info queries
#     HYBRID = "hybrid"  # Queries spanning both domains


# class RAGPipeline:
#     """Main RAG pipeline orchestrator."""
    
#     def __init__(
#         self,
#         vector_db: ITUVectorDatabase = None,
#         course_db: CourseDatabase = None,
#     ):
#         """Initialize the RAG pipeline.
        
#         Args:
#             vector_db: Initialized ITUVectorDatabase instance
#             course_db: Initialized CourseDatabase instance
#         """
#         self.vector_db = vector_db
#         self.course_db = course_db
#     # No OpenAI client — Ollama HTTP is used when configured
#         # Ollama support: optional local or remote Ollama server URL (e.g. http://localhost:11434)
#         self.ollama_url = os.getenv('OLLAMA_URL')
#         self.ollama_api_key = os.getenv('OLLAMA_API_KEY')

#         # Configurable defaults from environment
#         self.default_sql_k = int(os.getenv('RAG_SQL_K', '5'))
#         self.default_vector_k = int(os.getenv('RAG_VECTOR_K', '3'))
#         self.max_context_chars = int(os.getenv('RAG_MAX_CONTEXT_CHARS', '1000'))
#         self.max_sql_items = int(os.getenv('RAG_MAX_SQL_ITEMS', '5'))
#         self.max_vector_items = int(os.getenv('RAG_MAX_VECTOR_ITEMS', '3'))
    
#     def classify_query(self, query: str) -> Tuple[QueryType, Dict]:
#         """Classify a query to determine whether to use SQL, Vector, or Hybrid retrieval.
#         Tries a quick SQL probe when course-like patterns or course keywords are present.
#         Returns: (QueryType, metadata)
#         """
#         q = (query or "").strip()
#         ql = q.lower()

#         # Patterns
#         course_code_pattern = r'\b([A-Z]{2,}[A-Z0-9]*)\b'  # e.g. BDSA, BDSA101, KSADAPS1KU

#         # Extract explicit course code if present (highest priority)
#         course_code_match = re.search(course_code_pattern, query)
#         extracted_course_code = course_code_match.group(1) if course_code_match else None
        
#         # Exclude common non-course-code uppercase words
#         excluded_codes = {'ECTS', 'IT', 'ITU', 'MSC', 'BSC', 'AI', 'ML', 'NLP', 'CV'}
#         if extracted_course_code and extracted_course_code.upper() in excluded_codes:
#             extracted_course_code = None
        
#         has_course_kw = any(kw in ql for kw in COURSE_KEYWORDS)
#         has_exchange_kw = any(kw in ql for kw in EXCHANGE_KEYWORDS)
#         has_vector_kw = any(kw in ql for kw in VECTOR_KEYWORDS)

#         # Detect explicit ECTS mention (e.g., '15 ects' or '7,5 ECTS') and semesters
#         ects_value = None
#         ects_match = re.search(r"\b(\d{1,2}(?:[\.,]\d)?)\s*(?:ects)\b", ql)
#         if ects_match:
#             raw = ects_match.group(1).replace(',', '.')
#             try:
#                 ects_value = float(raw)
#             except Exception:
#                 ects_value = None
        
#         semester_match = None
#         if 'autumn' in ql or 'fall' in ql:
#             semester_match = 'Autumn'
#         elif 'spring' in ql:
#             semester_match = 'Spring'

#         # Detect language mentions (e.g., 'english courses')
#         language_match = None
#         if 'english' in ql:
#             language_match = 'English'
#         elif 'danish' in ql or 'dansk' in ql:
#             language_match = 'Danish'

#         # Detect minimum/least/lowest ECTS keywords
#         has_min_ects = any(kw in ql for kw in MIN_ECTS_KEYWORDS)

#         logger.debug(f"Classify query: '{q}' | course_code={extracted_course_code} | course_kw={has_course_kw} | min_ects={has_min_ects}")

#         # HIGHEST PRIORITY: If explicit course code found, look up directly (SQL only, ignore vector keywords)
#         if extracted_course_code:
#             logger.debug(f"Found explicit course code: {extracted_course_code}")
#             return QueryType.SQL, {'course_code': extracted_course_code}

#         # Special case: if user asks for min/least ECTS, handle it as a special SQL query
#         if has_min_ects:
#             qtype = QueryType.HYBRID if has_exchange_kw or has_vector_kw else QueryType.SQL
#             meta = {'sql_matches': 0, 'min_ects': True}
#             if language_match:
#                 meta['language'] = language_match
#             logger.debug(f"Detected minimum ECTS query; classifying as {qtype} with min_ects sorting")
#             return qtype, meta

#         if ects_value is not None:
#             if self.course_db:
#                 try:
#                     ects_probe = self.course_db.search_courses(ects=ects_value, limit=5)
#                     if ects_probe:
#                         qtype = QueryType.HYBRID if has_exchange_kw or has_vector_kw else QueryType.SQL
#                         meta = {'sql_matches': len(ects_probe), 'ects': ects_value}
#                         if language_match:
#                             meta['language'] = language_match
#                         logger.debug(f"ECTS probe matched {len(ects_probe)} rows; classifying as {qtype}")
#                         return qtype, meta
#                 except Exception as e:
#                     logger.debug(f"ECTS probe failed in classify_query: {e}")

#             # If ECTS was mentioned but no matches found, still treat as SQL intent with ects hint
#             meta = {'sql_matches': 0, 'ects': ects_value}
#             if language_match:
#                 meta['language'] = language_match
#             return QueryType.SQL, meta

#         if has_course_kw or semester_match:
#             if self.course_db:
#                 try:
#                     # First try a direct probe across searchable fields
#                     probe_results = self.course_db.search_courses(query=q, limit=5)
#                     if probe_results:
#                         qtype = QueryType.HYBRID if has_exchange_kw or has_vector_kw else QueryType.SQL
#                         meta = {'sql_matches': len(probe_results)}
#                         if language_match:
#                             meta['language'] = language_match
#                         if ects_value is not None:
#                             meta['ects'] = ects_value
#                         logger.debug(f"SQL direct probe matched {len(probe_results)} rows; classifying as {qtype}")
#                         return qtype, meta

#                     # If direct probe failed, try n-gram title probes (more robust for partial titles)
#                     tokens = [t for t in re.findall(r"\w+", ql) if len(t) > 2]
#                     # Only try n-grams of 2+ words (single words are too noisy)
#                     for n in range(min(4, len(tokens)), 1, -1):  # Changed: start from 2-word phrases, skip single words
#                         for i in range(len(tokens) - n + 1):    
#                             ngram = ' '.join(tokens[i:i+n])
#                             # Skip noisy n-grams that are unlikely to be useful as titles
#                             if ngram.strip().lower() in NOISY_NGRAMS:
#                                 continue
#                             # Skip if ngram contains only common words
#                             ngram_words = ngram.lower().split()
#                             if all(w in STOP_WORDS or w in NOISY_NGRAMS for w in ngram_words):
#                                 continue
#                             try:
#                                 title_matches = self.course_db.search_courses(course_title=ngram, limit=3)
#                                 if title_matches:
#                                     qtype = QueryType.HYBRID if has_exchange_kw or has_vector_kw else QueryType.SQL
#                                     meta = {'sql_matches': len(title_matches), 'probe_ngram': ngram}
#                                     if language_match:
#                                         meta['language'] = language_match
#                                     if ects_value is not None:
#                                         meta['ects'] = ects_value
#                                     logger.debug(f"Title n-gram probe '{ngram}' matched {len(title_matches)} rows; classifying as {qtype}")
#                                     return qtype, meta
#                             except Exception:
#                                 # ignore individual probe errors and continue
#                                 continue
#                 except Exception as e:
#                     logger.debug(f"SQL probe failed in classify_query: {e}")

#             # If it looked like a course query but no SQL DB or matches, still favor SQL intent
#             if has_course_kw or semester_match:
#                 meta = {'sql_matches': 0}
#                 if language_match:
#                     meta['language'] = language_match
#                 if ects_value is not None:
#                     meta['ects'] = ects_value
#                 logger.debug("No SQL matches found but query looks course-related; returning SQL intent (best-effort)")
#                 return QueryType.SQL, meta

#         # If vector-specific keywords or exchange-related keywords appear, pick vector
#         if has_exchange_kw or has_vector_kw:
#             logger.debug("Classifying as VECTOR based on exchange/vector keywords")
#             return QueryType.VECTOR, {}

#         # Ambiguous case: prefer Hybrid when both course and vector cues exist
#         if has_course_kw and has_vector_kw:
#             logger.debug("Classifying as HYBRID (both course and vector cues present)")
#             return QueryType.HYBRID, {}

#         # Default to VECTOR
#         logger.debug("Defaulting to VECTOR classification")
#         return QueryType.VECTOR, {}
    
#     def query_sql(self, query: str, metadata: Dict, k: int = 5, offset: int = 0) -> List[Dict]:
#         """Query the SQL course database.
        
#         Args:
#             query: User query string
#             metadata: Query metadata from classification
#             k: Maximum number of results
            
#         Returns:
#             List of course dictionaries
#         """
#         if not self.course_db:
#             return []
        
#         try:
#             # Extract keywords from query
#             keywords = re.findall(r'\b\w{3,}\b', query.lower())
#             keywords = [kw for kw in keywords if kw not in STOP_WORDS]
            
#             # Special handling for minimum ECTS queries
#             if metadata.get("min_ects"):
#                 all_courses = self.course_db.search_courses(limit=1000)
#                 valid = [c for c in all_courses if c.get('ects') is not None and c.get('ects') > 0]
#                 valid.sort(key=lambda x: x.get('ects', float('inf')))
#                 return valid[:k]
            
#             # Build search parameters
#             search_params = {"limit": k, "offset": offset}
            
#             # Check if this is a general "list all courses" type query
#             is_general_list = (
#                 metadata.get("sql_matches", 0) == 0 and
#                 not metadata.get("course_code") and
#                 not metadata.get("probe_ngram") and
#                 not metadata.get("ects") and
#                 not metadata.get("semester") and
#                 not metadata.get("language")
#             )
#             print("in general list")
#             print(is_general_list)
            
#             # Apply metadata hints to search parameters (in priority order)
#             if metadata.get("course_code"):
#                 search_params["course_code"] = metadata["course_code"]
#             elif metadata.get("probe_ngram"):
#                 # Only use probe_ngram if it's meaningful (not a filter-only query)
#                 # Skip if query is primarily about language/semester/ECTS filtering
#                 if not (metadata.get("language") and metadata.get("sql_matches", 0) < 10):
#                     search_params["course_title"] = metadata.get("probe_ngram")
#                 else:
#                     # Language-focused query, use free-text search instead
#                     search_params["query"] = query
#             elif metadata.get("course_specific"):
#                 search_params["course_title"] = query
#             elif is_general_list:
#                 # General "show all courses" query - don't pass query param to get all courses
#                 # Will return all courses sorted by relevance/date
#                 pass
#             else:
#                 # Default: use free-text search across title, description, abstract
#                 search_params["query"] = query

#             # Apply optional filters
#             if metadata.get("language"):
#                 search_params["language"] = metadata.get("language")
#             if metadata.get("semester"):
#                 search_params["semester"] = metadata["semester"]
#             if metadata.get("level"):
#                 search_params["level"] = metadata["level"]
#             if metadata.get("exchange_related"):
#                 search_params["offered_exchange"] = True

#             # Handle ECTS filter
#             if metadata.get("ects") is not None:
#                 try:
#                     search_params["ects"] = float(metadata.get("ects"))
#                 except Exception:
#                     search_params["ects"] = metadata.get("ects")
#                 # If only filtering by ECTS (no specific course), remove free-text query
#                 if not search_params.get("course_code") and not search_params.get("course_title"):
#                     search_params.pop('query', None)
            
#             # Remove free-text query if only using specific filters
#             if metadata.get("language") and not search_params.get("course_code") and not search_params.get("course_title"):
#                 search_params.pop('query', None)

#             # Perform search
#             logger.debug(f"SQL search params: {search_params}")
#             courses = self.course_db.search_courses(**search_params)
            
#             # Add relevance scores and sort
#             for course in courses:
#                 course["relevance_score"] = self._calculate_sql_relevance(course, query, keywords)
#             courses.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
#             return courses[:k]
            
#         except Exception as e:
#             logger.error(f"Error querying SQL database: {e}")
#             return []
    
#     def query_vector(self, query: str, k: int = 5) -> List[Dict]:
#         """Query the FAISS vector database.
        
#         Args:
#             query: User query string
#             k: Maximum number of results
            
#         Returns:
#             List of vector search results
#         """
#         if not self.vector_db:
#             return []
        
#         try:
#             results = self.vector_db.search(query, k=k)
#             return results
#         except Exception as e:
#             logger.error(f"Error querying vector database: {e}")
#             return []

#     def query_vector_paginated(self, query: str, k: int = 5, offset: int = 0) -> List[Dict]:
#         """Query vector DB with simple pagination by requesting k+offset and slicing results."""
#         if not self.vector_db:
#             return []
#         try:
#             results = self.vector_db.search(query, k=offset + k)
#             # slice safely
#             return results[offset: offset + k]
#         except Exception as e:
#             logger.error(f"Error querying vector database (paginated): {e}")
#             return []
    
#     def _calculate_sql_relevance(self, course: Dict, query: str, keywords: List[str]) -> float:
#         """Calculate relevance score for a course based on query keywords.
#         Searches title, description, abstract, and course code.
        
#         Args:
#             course: Course dictionary
#             query: Original query string
#             keywords: Extracted keywords
            
#         Returns:
#             Relevance score (0.0 to 1.0)
#         """
#         score = 0.0
#         query_lower = query.lower()
        
#         # Check course code match (exact or partial) — highest priority
#         course_code = (course.get("course_code") or "").lower()
#         if course_code and course_code in query_lower:
#             score += 0.5
        
#         # Check title match
#         title = (course.get("course_title") or "").lower()
#         if title:
#             if any(kw in title for kw in keywords):
#                 score += 0.4
#             if query_lower in title or title in query_lower:
#                 score += 0.3
        
#         # Check description AND abstract (search both fields comprehensively)
#         description = (course.get("description") or "").lower()
#         abstract = (course.get("abstract") or "").lower()
#         combined_text = description + " " + abstract
        
#         if combined_text:
#             # Count keyword matches in description/abstract
#             keyword_matches = sum(1 for kw in keywords if kw in combined_text)
#             score += min(0.2, keyword_matches * 0.05)
        
#         return min(1.0, score)
    
#     def merge_results(
#         self,
#         sql_results: List[Dict],
#         vector_results: List[Dict],
#         query_type: QueryType
#     ) -> Dict:
#         """Merge SQL and vector results into a unified context.
        
#         Args:
#             sql_results: Results from SQL database
#             vector_results: Results from vector database
#             query_type: Type of query (determines merge strategy)
            
#         Returns:
#             Dictionary with merged context and metadata
#         """
#         # Apply hybrid scoring and ranking for hybrid queries
#         if query_type == QueryType.HYBRID:
#             sql_results, vector_results = self._rank_hybrid_results(
#                 sql_results, vector_results
#             )
        
#         # Apply context truncation to prevent token overflow
#         sql_results = self._truncate_context(sql_results, max_items=self.max_sql_items)
#         vector_results = self._truncate_context(vector_results, max_items=self.max_vector_items)
        
#         merged = {
#             "sql_results": sql_results,
#             "vector_results": vector_results,
#             "query_type": query_type.value,
#             "total_results": len(sql_results) + len(vector_results)
#         }
        
#         # Prioritize SQL results for factual course data
#         if query_type == QueryType.SQL:
#             merged["primary_source"] = "sql"
#             merged["context"] = self._format_sql_context(sql_results)
#         elif query_type == QueryType.VECTOR:
#             merged["primary_source"] = "vector"
#             merged["context"] = self._format_vector_context(vector_results)
#         else:  # HYBRID
#             merged["primary_source"] = "hybrid"
#             merged["context"] = self._format_hybrid_context(sql_results, vector_results)
        
#         return merged
    
#     def _rank_hybrid_results(
#         self,
#         sql_results: List[Dict],
#         vector_results: List[Dict]
#     ) -> Tuple[List[Dict], List[Dict]]:
#         """Rank and score hybrid results using combined scoring.
        
#         Args:
#             sql_results: SQL course results
#             vector_results: Vector search results
            
#         Returns:
#             Tuple of (ranked_sql_results, ranked_vector_results)
#         """
#         sql_weight = 0.6
#         vector_weight = 0.4
        
#         # Normalize and score SQL results
#         for result in sql_results:
#             score = result.get("relevance_score", 0.0)
#             result["hybrid_score"] = min(1.0, max(0.0, score)) * sql_weight
        
#         # Normalize and score vector results
#         for result in vector_results:
#             score = result.get("similarity_score", 0.0)
#             result["hybrid_score"] = min(1.0, max(0.0, score)) * vector_weight
        
#         # Sort by hybrid scores
#         sql_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
#         vector_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        
#         return sql_results, vector_results
    
#     def _truncate_context(self, results: List[Dict], max_items: int = 5) -> List[Dict]:
#         """Truncate context to prevent token overflow.
        
#         Args:
#             results: List of result dictionaries
#             max_items: Maximum number of items to keep
            
#         Returns:
#             Truncated list of results
#         """
#         if len(results) <= max_items:
#             return results
        
#         # Keep top N results by score
#         return results[:max_items]
    
#     def _format_sql_context(self, courses: List[Dict]) -> str:
#         """Format SQL course results into context string.
        
#         Args:
#             courses: List of course dictionaries
            
#         Returns:
#             Formatted context string
#         """
#         if not courses:
#             return ""
        
#         context_parts = []
#         for i, course in enumerate(courses, 1):
#             parts = []
            
#             # Course title and code
#             title = course.get("course_title", "Unknown Course")
#             code = course.get("course_code", "")
#             if code:
#                 parts.append(f"Course {i}: {title} ({code})")
#             else:
#                 parts.append(f"Course {i}: {title}")
            
#             # ECTS
#             ects = course.get("ects")
#             if ects:
#                 parts.append(f"ECTS: {ects}")
            
#             # Semester
#             semester = course.get("semester")
#             if semester:
#                 parts.append(f"Semester: {semester}")
            
#             # Level
#             level = course.get("level")
#             if level:
#                 parts.append(f"Level: {level}")
            
#             # Exchange availability
#             exchange = course.get("offered_exchange")
#             if exchange == "yes":
#                 parts.append("Available for exchange students: Yes")
            
#             # Description/Abstract
#             description = course.get("description") or course.get("abstract")
#             if description:
#                 parts.append(f"Description: {description[:self.max_context_chars]}")
            
#             # Teachers
#             teachers = course.get("teachers")
#             if teachers:
#                 parts.append(f"Instructors: {teachers}")
            
#             # Prerequisites
#             prereqs = course.get("formal_prerequisites")
#             if prereqs:
#                 parts.append(f"Prerequisites: {prereqs}")
            
#             context_parts.append("\n".join(parts))
        
#         return "\n\n---\n\n".join(context_parts)
    
#     def _format_vector_context(self, results: List[Dict]) -> str:
#         """Format vector search results into context string.
        
#         Args:
#             results: List of vector search results
            
#         Returns:
#             Formatted context string
#         """
#         if not results:
#             return ""
        
#         context_parts = []
#         for i, result in enumerate(results, 1):
#             title = result.get("title", "Untitled")
#             text = result.get("text", "")
#             url = result.get("url", "")
            
#             context_parts.append(
#                 f"Source {i}: {title}\n"
#                 f"Content: {text[:self.max_context_chars]}\n"
#                 f"URL: {url}"
#             )
        
#         return "\n\n---\n\n".join(context_parts)
    
#     def _format_hybrid_context(self, sql_results: List[Dict], vector_results: List[Dict]) -> str:
#         """Format hybrid results (both SQL and vector) into context string.
        
#         Args:
#             sql_results: SQL course results
#             vector_results: Vector search results
            
#         Returns:
#             Formatted context string
#         """
#         parts = []
        
#         if sql_results:
#             parts.append("=== COURSE INFORMATION ===")
#             parts.append(self._format_sql_context(sql_results))
        
#         if vector_results:
#             if parts:
#                 parts.append("\n\n=== GENERAL ITU INFORMATION ===")
#             else:
#                 parts.append("=== GENERAL ITU INFORMATION ===")
#             parts.append(self._format_vector_context(vector_results))
        
#         return "\n\n".join(parts)
    
#     def retrieve(self, query: str, sql_k: int = None, sql_offset: int = 0, vector_k: int = None, vector_offset: int = 0) -> Dict:
#         """Main retrieval function that orchestrates the RAG pipeline.
        
#         Args:
#             query: User query string
#             sql_k: Number of SQL results to retrieve
#             vector_k: Number of vector results to retrieve
            
#         Returns:
#             Dictionary with merged results and context
#         """

#         # Classify query
#         query_type, metadata = self.classify_query(query)

#         # Resolve defaults
#         if sql_k is None:
#             sql_k = self.default_sql_k
#         if vector_k is None:
#             vector_k = self.default_vector_k

#         # Retrieve from appropriate sources
#         sql_results = []
#         vector_results = []

#         if query_type in [QueryType.SQL, QueryType.HYBRID]:
#             sql_results = self.query_sql(query, metadata, k=sql_k, offset=sql_offset)

#         if query_type in [QueryType.VECTOR, QueryType.HYBRID]:
#             vector_results = self.query_vector_paginated(query, k=vector_k, offset=vector_offset)
        
#         # Merge results
#         merged = self.merge_results(sql_results, vector_results, query_type)

#         # Add pagination totals where available
#         merged_meta = merged.copy()
#         # SQL total: attempt a count using the same metadata
#         try:
#             if self.course_db:
#                 count_params = {
#                     'query': None,
#                     'course_code': metadata.get('course_code'),
#                     'course_title': metadata.get('probe_ngram') or (query if metadata.get('course_specific') else None),
#                     'semester': metadata.get('semester'),
#                     'level': metadata.get('level'),
#                     'ects': metadata.get('ects'),
#                     'offered_exchange': metadata.get('exchange_related'),
#                     'programme': None,
#                     'language': metadata.get('language')
#                 }
#                 sql_total = self.course_db.count_courses(**count_params)
#                 merged_meta['sql_total'] = sql_total
#         except Exception:
#             merged_meta['sql_total'] = len(sql_results)

#         # Vector approximate total
#         try:
#             if self.vector_db:
#                 vd_stats = self.vector_db.get_database_stats()
#                 merged_meta['vector_total'] = vd_stats.get('total_vectors') if vd_stats else len(self.vector_db.metadata)
#         except Exception:
#             merged_meta['vector_total'] = len(vector_results)

#         return merged_meta
    
#     def generate_response(
#         self,
#         query: str,
#         context: Dict,
#         use_llm: bool = True
#     ) -> str:
#         """Generate a natural language response using the retrieved context.
        
#         Args:
#             query: User's original query
#             context: Merged context dictionary from retrieve()
#             use_llm: Whether to use LLM (if available) or fallback
            
#         Returns:
#             Generated response string
#         """
#         # If no context, return a helpful fallback
#         if not context.get("context"):
#             return {"text": "I couldn't find relevant information to answer your question. Could you please rephrase it or provide more details?", "llm_used": False}

#         # Try LLM if available and requested. Support Ollama HTTP API.
#         if use_llm and self.ollama_url:
#             try:
#                 llm_text = self._generate_llm_response(query, context)
#                 return {"text": llm_text, "llm_used": True}
#             except Exception as e:
#                 logger.error(f"Error generating LLM response: {e}")
#                 # Fall through to template-based response

#         # Fallback to template-based response
#         tpl = self._generate_template_response(query, context)
#         return {"text": tpl, "llm_used": False}
    
#     def _generate_llm_response(self, query: str, context: Dict) -> str:
#         """Generate response using Ollama LLM.
        
#         Args:
#             query: User query
#             context: Merged context dictionary
            
#         Returns:
#             LLM-generated response
#         """
#         # Use environment-configurable model parameters (prefer OLLAMA_* vars)
#         model = os.getenv('OLLAMA_MODEL', os.getenv('OPENAI_MODEL', 'gpt-oss'))
#         temperature = float(os.getenv('OLLAMA_TEMPERATURE', os.getenv('OPENAI_TEMPERATURE', '0.1')))
#         max_tokens = int(os.getenv('OLLAMA_MAX_TOKENS', os.getenv('OPENAI_MAX_TOKENS', '800')))

#         system_prompt = (
#             "You are a concise, strict-format assistant for exchange students at the IT University of Copenhagen (ITU).\n"
#             "Important rules (follow exactly):\n"
#             "1) Always use ONLY the provided Context. Do NOT invent facts or add information not present in Context. If something is not in Context, reply: 'I couldn't find that in the available ITU data.'\n"
#             "2) FORMATTING FOR COURSES:\n"
#             "   - When answering about courses, list them as a NUMBERED LIST with each course on its own line\n"
#             "   - Format: 'N. Title (Course Code) - X ECTS - Semester'\n"
#             "   - Example:\n"
#             "     1. Advanced Machine Learning (KSAMLDS2KU) - 7.5 ECTS - Spring 2026\n"
#             "     2. Machine Learning (BSMALEA1KU) - 15.0 ECTS - Autumn 2026\n"
#             "   - Each course must be on its own line with proper numbering\n"
#             "3) FORMATTING FOR HYBRID QUERIES (courses + general info):\n"
#             "   - Start with 'Relevant Courses:' section with numbered list (one course per line, formatted as above)\n"
#             "   - Add blank line\n"
#             "   - Then 'General Information:' section with facts about the topic (one paragraph)\n"
#             "4) FORMATTING FOR GENERAL INFORMATION:\n"
#             "   - Use only vector data from Context\n"
#             "   - Write concisely in one or two short paragraphs\n"
#             "   - Do NOT include citations or source references\n"
#             "5) Be concise and well-formatted: clear section headers, proper spacing between sections, numbered lists for courses\n"
#             "6) Do not include any intermediate JSON or streaming fragments — output a single clean text response only.\n"
#             "7) If the query is unrelated or Context is empty, respond with: 'I couldn't find relevant information to answer your question.'\n"
#         )

#         user_prompt = f"User Question: {query}\n\nContext:\n{context.get('context', 'No context available')}\n\nProvide a concise, factual answer and include citations as requested."

#         # Ollama is the only supported LLM backend now. Require OLLAMA_URL to be set.
#         if self.ollama_url:
#             try:
#                 return self._generate_ollama_response(system_prompt, user_prompt, model=model, temperature=temperature, max_tokens=max_tokens)
#             except Exception as e:
#                 logger.error(f"Error calling Ollama API: {e}")
#                 raise

#         # If we reach here, no Ollama URL configured — raise a clear error so callers fall back to templates
#         raise RuntimeError("OLLAMA_URL is not configured; cannot call LLM")

#     def _generate_ollama_response(self, system_prompt: str, user_prompt: str, model: str = None, temperature: float = 0.1, max_tokens: int = 800) -> str:
#         """Call an Ollama server via its HTTP API and return the generated text.

#         Expects an environment variable OLLAMA_URL (e.g. http://localhost:11434).
#         Optionally uses OLLAMA_API_KEY for authorization if set.
#         """
#         if not self.ollama_url:
#             raise RuntimeError("OLLAMA_URL is not configured")

#         payload = {
#             # Ollama typically accepts a model and prompt`/instruction` field; place system+user into the prompt
#             "model": model or os.getenv('OLLAMA_MODEL', os.getenv('OPENAI_MODEL', 'gpt-oss')),
#             "prompt": f"{system_prompt}\n\n{user_prompt}",
#             "temperature": float(temperature),
#             # max_tokens may not be supported by all Ollama models; include as a hint
#             "max_tokens": int(max_tokens)
#         }

#         headers = {"Content-Type": "application/json"}
#         if self.ollama_api_key:
#             headers["Authorization"] = f"Bearer {self.ollama_api_key}"

#         url = self.ollama_url.rstrip('/') + '/api/generate'
#         resp = requests.post(url, json=payload, headers=headers, timeout=30, stream=True)

#         # If endpoint returned a non-200 code, try to show a helpful error
#         if resp.status_code != 200:
#             try:
#                 body = resp.text
#             except Exception:
#                 body = '<unreadable body>'
#             logger.error(f"Ollama returned status {resp.status_code}: {body}")
#             raise RuntimeError(f"Ollama API error: {resp.status_code}")

#         # The Ollama server may stream incremental JSON objects (SSE-like or newline-delimited JSON).
#         # We'll iterate the response lines and accumulate text fragments from common fields
#         accumulated = []
#         final_text = None

#         try:
#             for raw_line in resp.iter_lines(decode_unicode=True):
#                 if not raw_line:
#                     continue

#                 line = raw_line.strip()
#                 # handle SSE 'data: {...}' lines
#                 if line.startswith('data:'):
#                     line = line[len('data:'):].strip()

#                 # Try to parse JSON from the line
#                 try:
#                     part = None
#                     j = None
#                     # Some servers send multiple JSON objects per line; try json.loads
#                     import json
#                     j = json.loads(line)

#                     # prefer 'response' then 'output' then nested choices/results
#                     # IMPORTANT: ignore 'thinking' fragments (internal chain-of-thought)
#                     if isinstance(j, dict):
#                         # If server sends chunk with 'response' (final) or 'thinking' (partial)
#                         if 'response' in j and isinstance(j['response'], str) and j['response']:
#                             part = j['response']
#                         elif 'output' in j and isinstance(j['output'], str) and j['output']:
#                             part = j['output']
#                         elif 'results' in j and isinstance(j['results'], list) and j['results']:
#                             first = j['results'][0]
#                             if isinstance(first, dict):
#                                 for key in ('text', 'output'):
#                                     if key in first and isinstance(first[key], str) and first[key]:
#                                         part = first[key]
#                                         break
#                             elif isinstance(first, str):
#                                 part = first
#                         elif 'choices' in j and isinstance(j['choices'], list) and j['choices']:
#                             ch = j['choices'][0]
#                             if isinstance(ch, dict):
#                                 if 'message' in ch and isinstance(ch['message'], dict) and 'content' in ch['message']:
#                                     part = str(ch['message']['content'])
#                                 else:
#                                     for key in ('text', 'output'):
#                                         if key in ch and isinstance(ch[key], str):
#                                             part = ch[key]
#                                             break
#                         # If there's a 'done' flag and it's true, mark final_text
#                         if j.get('done') is True:
#                             # Only append if this chunk contained a non-thinking response/output
#                             if part:
#                                 accumulated.append(part)
#                             final_text = ''.join(accumulated).strip()
#                             break

#                     # Only append selected parts (response/output/text) -- ignore 'thinking' fragments
#                     if part:
#                         accumulated.append(part)
#                         # continue reading until done=True is seen
#                         continue
#                 except Exception:
#                     # Not a JSON line -- ignore to avoid leaking chain-of-thought or protocol noise
#                     continue

#             # After streaming, if we have a final_text from done flag use it
#             if final_text is not None:
#                 return final_text

#             # Otherwise, join accumulated parts and return
#             if accumulated:
#                 return ''.join(accumulated).strip()

#             # Fallback: return raw text
#             return resp.text.strip()
#         finally:
#             try:
#                 resp.close()
#             except Exception:
#                 pass
    
#     def _generate_template_response(self, query: str, context: Dict) -> str:
#         """Generate response using template-based approach (fallback).
        
#         Args:
#             query: User query
#             context: Merged context dictionary
            
#         Returns:
#             Template-based response
#         """
#         query_type = context.get("query_type", "hybrid")
#         sql_results = context.get("sql_results", [])
#         vector_results = context.get("vector_results", [])
        
#         response_parts = []
        
#         if query_type == "sql" and sql_results:
#             for i, course in enumerate(sql_results[:self.max_sql_items], 1):
#                 title = course.get("course_title", "Unknown")
#                 code = course.get("course_code", "")
#                 ects = course.get("ects", "")
#                 semester = course.get("semester", "")
                
#                 course_info = f"{i}. {title}"
#                 if code:
#                     course_info += f" ({code})"
#                 if ects:
#                     course_info += f" - {ects} ECTS"
#                 if semester:
#                     course_info += f" - {semester}"
                
#                 response_parts.append(course_info)
        
#         elif query_type == "vector" and vector_results:
#             best_result = vector_results[0]
#             response_parts.append(best_result.get("text", "")[:self.max_context_chars])
        
#         elif query_type == "hybrid":
#             if sql_results:
#                 response_parts.append("Relevant Courses:")
#                 for i, course in enumerate(sql_results[:self.max_sql_items], 1):
#                     title = course.get("course_title", "Unknown")
#                     code = course.get("course_code", "")
#                     ects = course.get("ects", "")
#                     semester = course.get("semester", "")
                    
#                     course_info = f"{i}. {title}"
#                     if code:
#                         course_info += f" ({code})"
#                     if ects:
#                         course_info += f" - {ects} ECTS"
#                     if semester:
#                         course_info += f" - {semester}"
                    
#                     response_parts.append(course_info)
                
#                 response_parts.append("")  # Blank line separator
            
#             if vector_results:
#                 response_parts.append("General Information:")
#                 response_parts.append(vector_results[0].get("text", "")[:self.max_context_chars])
        
#         if not response_parts:
#             return "I couldn't find specific information to answer your question. Could you try rephrasing it?"
        
#         return "\n".join(response_parts)