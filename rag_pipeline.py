"""
RAG Pipeline for Chatbot_ITU

Implements a Retrieval-Augmented Generation pipeline that intelligently combines
structured SQL course data and unstructured FAISS vector data to provide
comprehensive responses to student queries.
"""

import re
from typing import List, Dict, Optional, Tuple, Literal
from enum import Enum
import logging

import os
from vector_db import ITUVectorDatabase
from course_db import CourseDatabase

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query classification types."""
    SQL = "sql"  # Course-specific, structured queries
    VECTOR = "vector"  # General ITU info queries
    HYBRID = "hybrid"  # Queries spanning both domains


class RAGPipeline:
    """Main RAG pipeline orchestrator."""
    
    def __init__(
        self,
        vector_db: ITUVectorDatabase = None,
        course_db: CourseDatabase = None,
        openai_client = None
    ):
        """Initialize the RAG pipeline.
        
        Args:
            vector_db: Initialized ITUVectorDatabase instance
            course_db: Initialized CourseDatabase instance
            openai_client: OpenAI client instance (optional)
        """
        self.vector_db = vector_db
        self.course_db = course_db
        self.openai_client = openai_client
        # Configurable defaults from environment
        self.default_sql_k = int(os.getenv('RAG_SQL_K', '5'))
        self.default_vector_k = int(os.getenv('RAG_VECTOR_K', '3'))
        self.max_context_chars = int(os.getenv('RAG_MAX_CONTEXT_CHARS', '1000'))
        self.max_sql_items = int(os.getenv('RAG_MAX_SQL_ITEMS', '5'))
        self.max_vector_items = int(os.getenv('RAG_MAX_VECTOR_ITEMS', '3'))
    
    def classify_query(self, query: str) -> Tuple[QueryType, Dict]:
        """Classify a query to determine whether to use SQL, Vector, or Hybrid retrieval.
        Tries a quick SQL probe when course-like patterns or course keywords are present.
        Returns: (QueryType, metadata)
        """
        q = (query or "").strip()
        ql = q.lower()

        # Patterns & keywords
        course_code_pattern = r'\b[A-Z]{2,5}\s*\d{0,4}\b'  # e.g. BDSA, BDSA101
        course_keywords = ['course', 'courses', 'ects', 'syllabus', 'course code', 'module', 'curriculum', 'study', 'class']
        exchange_keywords = ['exchange', 'exchange student', 'exchange students', 'erasmus']
        vector_keywords = ['apply', 'application', 'admission', 'housing', 'deadline', 'campus', 'research', 'life', 'accommodation', 'enroll', 'enrol']

        # Basic detections
        is_course_code = bool(re.search(course_code_pattern, query))
        has_course_kw = any(kw in ql for kw in course_keywords)
        has_exchange_kw = any(kw in ql for kw in exchange_keywords)
        has_vector_kw = any(kw in ql for kw in vector_keywords)

        # Detect explicit ECTS mention (e.g., '15 ects' or '7,5 ECTS') and semesters
        ects_value = None
        ects_match = re.search(r"\b(\d{1,2}(?:[\.,]\d)?)\s*(?:ects)\b", ql)
        if ects_match:
            raw = ects_match.group(1).replace(',', '.')
            try:
                ects_value = float(raw)
            except Exception:
                ects_value = None
        semester_match = None
        if 'autumn' in ql or 'fall' in ql:
            semester_match = 'Autumn'
        elif 'spring' in ql:
            semester_match = 'Spring'

        # Detect language mentions (e.g., 'english courses')
        language_match = None
        if 'english' in ql:
            language_match = 'English'
        elif 'danish' in ql or 'dansk' in ql:
            language_match = 'Danish'

        logger.debug(f"Classify query: '{q}' | course_code={is_course_code} | course_kw={has_course_kw} | exchange_kw={has_exchange_kw} | vector_kw={has_vector_kw} | ects={ects_value} | semester={semester_match} | language={language_match}")

        # If it looks course-related (code, keyword, ects or semester), perform SQL probes
        # Special-case: if ECTS explicitly mentioned, prefer an ECTS-based probe and skip n-gram title probing
        if ects_value is not None:
            if self.course_db:
                try:
                    ects_probe = self.course_db.search_courses(ects=ects_value, limit=5)
                    if ects_probe:
                        qtype = QueryType.HYBRID if has_exchange_kw or has_vector_kw else QueryType.SQL
                        meta = {'sql_matches': len(ects_probe), 'ects': ects_value}
                        if language_match:
                            meta['language'] = language_match
                        logger.debug(f"ECTS probe matched {len(ects_probe)} rows; classifying as {qtype}")
                        return qtype, meta
                except Exception as e:
                    logger.debug(f"ECTS probe failed in classify_query: {e}")

            # If ECTS was mentioned but no matches found, still treat as SQL intent with ects hint
            meta = {'sql_matches': 0, 'ects': ects_value}
            if language_match:
                meta['language'] = language_match
            return QueryType.SQL, meta

        if is_course_code or has_course_kw or semester_match:
            if self.course_db:
                try:
                    # First try a direct probe across searchable fields
                    probe_results = self.course_db.search_courses(query=q, limit=5)
                    if probe_results:
                        qtype = QueryType.HYBRID if has_exchange_kw or has_vector_kw else QueryType.SQL
                        meta = {'sql_matches': len(probe_results)}
                        if language_match:
                            meta['language'] = language_match
                        if ects_value is not None:
                            meta['ects'] = ects_value
                        logger.debug(f"SQL direct probe matched {len(probe_results)} rows; classifying as {qtype}")
                        return qtype, meta

                    # If direct probe failed, try n-gram title probes (more robust for partial titles)
                    tokens = [t for t in re.findall(r"\w+", ql) if len(t) > 2]
                    for n in range(min(4, len(tokens)), 0, -1):
                        for i in range(len(tokens) - n + 1):
                            ngram = ' '.join(tokens[i:i+n])
                            try:
                                # Skip noisy n-grams that are unlikely to be useful as titles
                                noisy = {"language", "ects", "show", "give", "available", "courses", "course", "in", "the", "me"}
                                if ngram.strip().lower() in noisy:
                                    continue
                                title_matches = self.course_db.search_courses(course_title=ngram, limit=3)
                                if title_matches:
                                    qtype = QueryType.HYBRID if has_exchange_kw or has_vector_kw else QueryType.SQL
                                    meta = {'sql_matches': len(title_matches), 'probe_ngram': ngram}
                                    if language_match:
                                        meta['language'] = language_match
                                    if ects_value is not None:
                                        meta['ects'] = ects_value
                                    logger.debug(f"Title n-gram probe '{ngram}' matched {len(title_matches)} rows; classifying as {qtype}")
                                    return qtype, meta
                            except Exception:
                                # ignore individual probe errors and continue
                                continue
                except Exception as e:
                    logger.debug(f"SQL probe failed in classify_query: {e}")

            # If it looked like a course query but no SQL DB or matches, still favor SQL intent
            if is_course_code or has_course_kw or ects_match or semester_match:
                meta = {'sql_matches': 0}
                if language_match:
                    meta['language'] = language_match
                if ects_value is not None:
                    meta['ects'] = ects_value
                logger.debug("No SQL matches found but query looks course-related; returning SQL intent (best-effort)")
                return QueryType.SQL, meta

        # If vector-specific keywords or exchange-related keywords appear, pick vector
        if has_exchange_kw or has_vector_kw:
            logger.debug("Classifying as VECTOR based on exchange/vector keywords")
            return QueryType.VECTOR, {}

        # Ambiguous case: prefer Hybrid when both course and vector cues exist
        if has_course_kw and has_vector_kw:
            logger.debug("Classifying as HYBRID (both course and vector cues present)")
            return QueryType.HYBRID, {}

        # Default to VECTOR
        logger.debug("Defaulting to VECTOR classification")
        return QueryType.VECTOR, {}
    
    def query_sql(self, query: str, metadata: Dict, k: int = 5, offset: int = 0) -> List[Dict]:
        """Query the SQL course database.
        
        Args:
            query: User query string
            metadata: Query metadata from classification
            k: Maximum number of results
            
        Returns:
            List of course dictionaries
        """
        if not self.course_db:
            return []
        
        try:
            # Extract keywords from query
            keywords = re.findall(r'\b\w{3,}\b', query.lower())
            # Filter out common stop words
            stop_words = {"the", "and", "or", "but", "for", "with", "from", "about", "what", "which", "when", "where", "how", "why"}
            keywords = [kw for kw in keywords if kw not in stop_words]
            
            # Build search parameters
            search_params = {
                "limit": k,
                "offset": offset
            }
            
            if metadata.get("course_code"):
                search_params["course_code"] = metadata["course_code"]
            elif metadata.get("course_specific"):
                search_params["course_title"] = query
            # If classifier provided a probe n-gram (title fragment), prefer searching by title
            elif metadata.get("probe_ngram"):
                search_params["course_title"] = metadata.get("probe_ngram")
            else:
                search_params["query"] = query

            # If classifier provided a language hint, include it (do not overwrite query/course_title)
            if metadata.get("language"):
                search_params["language"] = metadata.get("language")
            
            if metadata.get("semester"):
                search_params["semester"] = metadata["semester"]
            
            if metadata.get("level"):
                search_params["level"] = metadata["level"]
            
            if metadata.get("exchange_related"):
                search_params["offered_exchange"] = True

            # If classifier provided an ECTS hint, include it and avoid combining with raw free-text query
            if metadata.get("ects") is not None:
                try:
                    # ensure numeric
                    search_params["ects"] = float(metadata.get("ects"))
                except Exception:
                    search_params["ects"] = metadata.get("ects")
                # If we only want to filter by ECTS (no specific title/code), remove free-text query
                if not search_params.get("course_code") and not search_params.get("course_title"):
                    search_params.pop('query', None)
            
            # If language was provided but no specific course identifier/title, prefer language-only search
            if metadata.get("language") and not search_params.get("course_code") and not search_params.get("course_title"):
                # remove the free-text query since it may prevent matches when combined with language
                search_params.pop('query', None)

            # Perform search
            logger.debug(f"SQL search params: {search_params}")
            courses = self.course_db.search_courses(**search_params)
            
            # Add relevance scores (simple keyword matching)
            for course in courses:
                course["relevance_score"] = self._calculate_sql_relevance(course, query, keywords)
            
            # Sort by relevance
            courses.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            return courses[:k]
            
        except Exception as e:
            logger.error(f"Error querying SQL database: {e}")
            return []
    
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
    
    def _calculate_sql_relevance(self, course: Dict, query: str, keywords: List[str]) -> float:
        """Calculate relevance score for a course based on query keywords.
        
        Args:
            course: Course dictionary
            query: Original query string
            keywords: Extracted keywords
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0
        query_lower = query.lower()
        
        # Check title match
        title = (course.get("course_title") or "").lower()
        if title:
            if any(kw in title for kw in keywords):
                score += 0.4
            if query_lower in title or title in query_lower:
                score += 0.3
        
        # Check description/abstract match
        description = (course.get("description") or course.get("abstract") or "").lower()
        if description:
            keyword_matches = sum(1 for kw in keywords if kw in description)
            score += min(0.2, keyword_matches * 0.05)
        
        # Check course code match
        course_code = (course.get("course_code") or "").lower()
        if course_code and course_code in query_lower:
            score += 0.3
        
        return min(1.0, score)
    
    def merge_results(
        self,
        sql_results: List[Dict],
        vector_results: List[Dict],
        query_type: QueryType
    ) -> Dict:
        """Merge SQL and vector results into a unified context.
        
        Args:
            sql_results: Results from SQL database
            vector_results: Results from vector database
            query_type: Type of query (determines merge strategy)
            
        Returns:
            Dictionary with merged context and metadata
        """
        # Apply hybrid scoring and ranking for hybrid queries
        if query_type == QueryType.HYBRID:
            sql_results, vector_results = self._rank_hybrid_results(
                sql_results, vector_results
            )
        
        # Apply context truncation to prevent token overflow
        sql_results = self._truncate_context(sql_results, max_items=self.max_sql_items)
        vector_results = self._truncate_context(vector_results, max_items=self.max_vector_items)
        
        merged = {
            "sql_results": sql_results,
            "vector_results": vector_results,
            "query_type": query_type.value,
            "total_results": len(sql_results) + len(vector_results)
        }
        
        # Prioritize SQL results for factual course data
        if query_type == QueryType.SQL:
            merged["primary_source"] = "sql"
            merged["context"] = self._format_sql_context(sql_results)
        elif query_type == QueryType.VECTOR:
            merged["primary_source"] = "vector"
            merged["context"] = self._format_vector_context(vector_results)
        else:  # HYBRID
            merged["primary_source"] = "hybrid"
            merged["context"] = self._format_hybrid_context(sql_results, vector_results)
        
        return merged
    
    def _rank_hybrid_results(
        self,
        sql_results: List[Dict],
        vector_results: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Rank and score hybrid results using combined scoring.
        
        Args:
            sql_results: SQL course results
            vector_results: Vector search results
            
        Returns:
            Tuple of (ranked_sql_results, ranked_vector_results)
        """
        # Normalize scores for SQL results (0.0 to 1.0)
        for result in sql_results:
            score = result.get("relevance_score", 0.0)
            result["normalized_score"] = min(1.0, max(0.0, score))
        
        # Normalize scores for vector results (similarity scores are already 0.0 to 1.0)
        for result in vector_results:
            score = result.get("similarity_score", 0.0)
            result["normalized_score"] = min(1.0, max(0.0, score))
        
        # Apply hybrid scoring: combine SQL relevance with vector similarity
        # For hybrid queries, we want to balance both sources
        # SQL results get slight priority for factual course data
        sql_weight = 0.6
        vector_weight = 0.4
        
        # Re-rank SQL results with hybrid consideration
        for result in sql_results:
            base_score = result.get("normalized_score", 0.0)
            # Boost if it has high relevance
            result["hybrid_score"] = base_score * sql_weight
        
        # Re-rank vector results
        for result in vector_results:
            base_score = result.get("normalized_score", 0.0)
            result["hybrid_score"] = base_score * vector_weight
        
        # Sort by hybrid scores
        sql_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        vector_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        
        return sql_results, vector_results
    
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
    
    def _format_sql_context(self, courses: List[Dict]) -> str:
        """Format SQL course results into context string.
        
        Args:
            courses: List of course dictionaries
            
        Returns:
            Formatted context string
        """
        if not courses:
            return ""
        
        context_parts = []
        for i, course in enumerate(courses, 1):
            parts = []
            
            # Course title and code
            title = course.get("course_title", "Unknown Course")
            code = course.get("course_code", "")
            if code:
                parts.append(f"Course {i}: {title} ({code})")
            else:
                parts.append(f"Course {i}: {title}")
            
            # ECTS
            ects = course.get("ects")
            if ects:
                parts.append(f"ECTS: {ects}")
            
            # Semester
            semester = course.get("semester")
            if semester:
                parts.append(f"Semester: {semester}")
            
            # Level
            level = course.get("level")
            if level:
                parts.append(f"Level: {level}")
            
            # Exchange availability
            exchange = course.get("offered_exchange")
            if exchange == "yes":
                parts.append("Available for exchange students: Yes")
            
            # Description/Abstract
            description = course.get("description") or course.get("abstract")
            if description:
                parts.append(f"Description: {description[:self.max_context_chars]}")
            
            # Teachers
            teachers = course.get("teachers")
            if teachers:
                parts.append(f"Instructors: {teachers}")
            
            # Prerequisites
            prereqs = course.get("formal_prerequisites")
            if prereqs:
                parts.append(f"Prerequisites: {prereqs}")
            
            context_parts.append("\n".join(parts))
        
        return "\n\n---\n\n".join(context_parts)
    
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
    
    def _format_hybrid_context(self, sql_results: List[Dict], vector_results: List[Dict]) -> str:
        """Format hybrid results (both SQL and vector) into context string.
        
        Args:
            sql_results: SQL course results
            vector_results: Vector search results
            
        Returns:
            Formatted context string
        """
        parts = []
        
        if sql_results:
            parts.append("=== COURSE INFORMATION ===")
            parts.append(self._format_sql_context(sql_results))
        
        if vector_results:
            if parts:
                parts.append("\n\n=== GENERAL ITU INFORMATION ===")
            else:
                parts.append("=== GENERAL ITU INFORMATION ===")
            parts.append(self._format_vector_context(vector_results))
        
        return "\n\n".join(parts)
    
    def retrieve(self, query: str, sql_k: int = None, sql_offset: int = 0, vector_k: int = None, vector_offset: int = 0) -> Dict:
        """Main retrieval function that orchestrates the RAG pipeline.
        
        Args:
            query: User query string
            sql_k: Number of SQL results to retrieve
            vector_k: Number of vector results to retrieve
            
        Returns:
            Dictionary with merged results and context
        """

        # Classify query
        query_type, metadata = self.classify_query(query)

        # Resolve defaults
        if sql_k is None:
            sql_k = self.default_sql_k
        if vector_k is None:
            vector_k = self.default_vector_k

        # Retrieve from appropriate sources
        sql_results = []
        vector_results = []

        if query_type in [QueryType.SQL, QueryType.HYBRID]:
            sql_results = self.query_sql(query, metadata, k=sql_k, offset=sql_offset)

        if query_type in [QueryType.VECTOR, QueryType.HYBRID]:
            vector_results = self.query_vector_paginated(query, k=vector_k, offset=vector_offset)
        
        # Merge results
        merged = self.merge_results(sql_results, vector_results, query_type)

        # Add pagination totals where available
        merged_meta = merged.copy()
        # SQL total: attempt a count using the same metadata
        try:
            if self.course_db:
                count_params = {
                    'query': None,
                    'course_code': metadata.get('course_code'),
                    'course_title': metadata.get('probe_ngram') or (query if metadata.get('course_specific') else None),
                    'semester': metadata.get('semester'),
                    'level': metadata.get('level'),
                    'ects': metadata.get('ects'),
                    'offered_exchange': metadata.get('exchange_related'),
                    'programme': None,
                    'language': metadata.get('language')
                }
                sql_total = self.course_db.count_courses(**count_params)
                merged_meta['sql_total'] = sql_total
        except Exception:
            merged_meta['sql_total'] = len(sql_results)

        # Vector approximate total
        try:
            if self.vector_db:
                vd_stats = self.vector_db.get_database_stats()
                merged_meta['vector_total'] = vd_stats.get('total_vectors') if vd_stats else len(self.vector_db.metadata)
        except Exception:
            merged_meta['vector_total'] = len(vector_results)

        return merged_meta
    
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
        if not context.get("context"):
            return "I couldn't find relevant information to answer your question. Could you please rephrase it or provide more details?"
        
        # Try LLM if available and requested
        if use_llm and self.openai_client:
            try:
                return self._generate_llm_response(query, context)
            except Exception as e:
                logger.error(f"Error generating LLM response: {e}")
                # Fall through to template-based response
        
        # Fallback to template-based response
        return self._generate_template_response(query, context)
    
    def _generate_llm_response(self, query: str, context: Dict) -> str:
        """Generate response using OpenAI LLM.
        
        Args:
            query: User query
            context: Merged context dictionary
            
        Returns:
            LLM-generated response
        """
        # Use environment-configurable model parameters
        model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
        max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '800'))

        system_prompt = (
            "You are a helpful assistant for exchange students at the IT University of Copenhagen (ITU). "
            "Answer using ONLY the provided Context. Do not invent facts. If the requested information is not in the Context, reply: 'I couldn't find that in the available ITU data.' "
            "Prioritize structured SQL course data when present. For each reported course, use this compact course-card format: "
            "Title (Course code) - X ECTS - Semester - Language - Instructors (if available). "
            "Cite sources inline using the context source index or URL, e.g. (Source 1) or (https://...). "
        )

        user_prompt = f"User Question: {query}\n\nContext:\n{context.get('context', 'No context available')}\n\nProvide a concise, factual answer and include citations as requested."

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content.strip()
    
    def _generate_template_response(self, query: str, context: Dict) -> str:
        """Generate response using template-based approach (fallback).
        
        Args:
            query: User query
            context: Merged context dictionary
            
        Returns:
            Template-based response
        """
        query_type = context.get("query_type", "hybrid")
        sql_results = context.get("sql_results", [])
        vector_results = context.get("vector_results", [])
        
        response_parts = []
        
        if query_type == "sql" and sql_results:
            response_parts.append("Here are the courses that match your query:\n\n")
            for i, course in enumerate(sql_results[:self.max_sql_items], 1):
                title = course.get("course_title", "Unknown")
                code = course.get("course_code", "")
                ects = course.get("ects", "")
                semester = course.get("semester", "")
                
                course_info = f"{i}. {title}"
                if code:
                    course_info += f" ({code})"
                if ects:
                    course_info += f" - {ects} ECTS"
                if semester:
                    course_info += f" - {semester}"
                
                response_parts.append(course_info)
        
        elif query_type == "vector" and vector_results:
            response_parts.append("Based on ITU information:\n\n")
            best_result = vector_results[0]
            response_parts.append(best_result.get("text", "")[:self.max_context_chars])
        
        elif query_type == "hybrid":
            if sql_results:
                response_parts.append("**Course Information:**\n")
                for course in sql_results[:self.max_sql_items]:
                    title = course.get("course_title", "Unknown")
                    code = course.get("course_code", "")
                    if code:
                        response_parts.append(f"- {title} ({code})")
                    else:
                        response_parts.append(f"- {title}")
                response_parts.append("")
            
            if vector_results:
                response_parts.append("**General Information:**\n")
                response_parts.append(vector_results[0].get("text", "")[:self.max_context_chars])
        
        if not response_parts:
            return "I couldn't find specific information to answer your question. Could you try rephrasing it?"
        
        return "\n".join(response_parts)

