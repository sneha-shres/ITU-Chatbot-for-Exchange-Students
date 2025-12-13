"""
RAG Pipeline for Chatbot_ITU
"""

import re
from typing import List, Dict, Optional, Tuple, Literal
from enum import Enum
import logging

import os
from database.vector_db import ITUVectorDatabase
from core.reasoning_layer import CourseCombinationReasoner
import requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)

STOP_WORDS = {"the", "and", "or", "but", "for", "with", "from", "about", "what", "which", "when", "where", "how", "why"}

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
        """
        if len(results) <= max_items:
            return results
        
        # Keep top N results by score
        return results[:max_items]
    
    def _format_vector_context(self, results: List[Dict]) -> str:
        """Format vector search results into context string.
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
            "model": model or os.getenv('OLLAMA_MODEL', os.getenv('OPENAI_MODEL', 'gpt-oss')),
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "temperature": float(temperature),
            "max_tokens": int(max_tokens)
        }

        headers = {"Content-Type": "application/json"}
        if self.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"

        url = self.ollama_url.rstrip('/') + '/api/generate'
        resp = requests.post(url, json=payload, headers=headers, timeout=30, stream=True)

        if resp.status_code != 200:
            try:
                body = resp.text
            except Exception:
                body = '<unreadable body>'
            logger.error(f"Ollama returned status {resp.status_code}: {body}")
            raise RuntimeError(f"Ollama API error: {resp.status_code}")

        accumulated = []
        final_text = None

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue


                # Normalize to string safely (some servers may yield bytes)
                if isinstance(raw_line, bytes):
                    try:
                        line = raw_line.decode('utf-8', errors='replace').strip()
                    except Exception:
                        line = str(raw_line).strip()
                else:
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