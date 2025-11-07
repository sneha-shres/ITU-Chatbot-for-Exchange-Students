# RAG Pipeline Architecture

## Overview

The Chatbot_ITU implements a **Retrieval-Augmented Generation (RAG)** pipeline that intelligently combines structured SQL course data and unstructured FAISS vector data to provide comprehensive responses to student queries.

## Architecture Components

### 1. Data Sources

#### SQL Database (`Courses/output/courses.db`)
- **Purpose**: Structured course information
- **Schema**: Contains course details including:
  - Course codes, titles, descriptions
  - ECTS credits, semesters, levels (BSc/MSc)
  - Instructors, prerequisites
  - Exchange student availability
  - Assessment methods, exam types
- **Interface**: `CourseDatabase` class in `course_db.py`

#### FAISS Vector Database
- **Purpose**: Unstructured ITU website content
- **Content**: General ITU information including:
  - Admission and application processes
  - Housing and accommodation
  - Campus facilities and services
  - Exchange student policies
  - Research opportunities
- **Interface**: `ITUVectorDatabase` class in `vector_db.py`

### 2. RAG Pipeline (`rag_pipeline.py`)

The `RAGPipeline` class orchestrates the entire retrieval and generation process:

#### Query Classification
- **Method**: `classify_query(query: str) -> Tuple[QueryType, Dict]`
- **Types**:
  - `SQL`: Course-specific queries (e.g., "What AI courses are available?")
  - `VECTOR`: General ITU info queries (e.g., "How do I apply for housing?")
  - `HYBRID`: Queries spanning both domains (e.g., "Which AI courses are suitable for exchange students?")

#### Retrieval Methods
- **SQL Retrieval**: `query_sql(query, metadata, k=5)`
  - Searches course database using keyword matching
  - Extracts course codes, semesters, levels from query
  - Calculates relevance scores based on keyword matches
- **Vector Retrieval**: `query_vector(query, k=5)`
  - Uses FAISS similarity search
  - Returns top-k most similar content chunks

#### Result Merging
- **Method**: `merge_results(sql_results, vector_results, query_type)`
- **Features**:
  - Hybrid scoring for HYBRID queries (60% SQL weight, 40% vector weight)
  - Context truncation to prevent token overflow
  - Intelligent ranking based on relevance scores
  - Formatting for LLM consumption

#### Response Generation
- **LLM-based**: Uses OpenAI GPT-3.5-turbo with carefully designed prompts
- **Template-based**: Fallback when LLM is unavailable
- **Prompts**: System and user prompts designed for exchange students

### 3. Integration (`app.py`)

The Flask application integrates the RAG pipeline:

```python
chatbot = Chatbot()
# Initializes:
# - Vector database
# - Course database
# - RAG pipeline
# - OpenAI client (if API key available)
```

#### Main Flow
1. User sends message via `/api/chat`
2. `Chatbot.generate_response()` is called
3. RAG pipeline retrieves relevant context
4. Response is generated using LLM or templates
5. Response is returned to user

## Query Flow Example

### Example 1: Course-Specific Query
**Query**: "What machine learning courses are available in Spring 2026?"

1. **Classification**: `QueryType.SQL` (course-specific keywords detected)
2. **Retrieval**: 
   - SQL: Search courses with "machine learning" + semester="Spring 2026"
   - Vector: Skip (not needed)
3. **Merging**: Format SQL results only
4. **Generation**: LLM synthesizes course list with details

### Example 2: General ITU Query
**Query**: "How do I apply for student housing?"

1. **Classification**: `QueryType.VECTOR` (general ITU info)
2. **Retrieval**:
   - SQL: Skip (not course-related)
   - Vector: Search for housing/admission content
3. **Merging**: Format vector results only
4. **Generation**: LLM provides step-by-step guidance

### Example 3: Hybrid Query
**Query**: "Which AI courses are suitable for exchange students?"

1. **Classification**: `QueryType.HYBRID` (course + exchange student keywords)
2. **Retrieval**:
   - SQL: Search courses with "AI" + `offered_exchange=True`
   - Vector: Search for exchange student policies
3. **Merging**: 
   - Apply hybrid scoring (60% SQL, 40% vector)
   - Rank and truncate results
   - Format both sources
4. **Generation**: LLM combines course list with exchange student context

## API Endpoints

### Chat Endpoint
- **POST** `/api/chat`
- **Body**: `{"message": "user query"}`
- **Returns**: `{"response": "generated response", "timestamp": "..."}`

### Course Search
- **POST** `/api/courses/search`
- **Body**: `{"query": "...", "semester": "...", "level": "...", "offered_exchange": true}`
- **Returns**: List of matching courses

### Exchange Courses
- **GET** `/api/courses/exchange?semester=Spring 2026&limit=20`
- **Returns**: Courses available for exchange students

### Query Classification
- **POST** `/api/rag/classify`
- **Body**: `{"query": "..."}`
- **Returns**: Query type and extracted metadata

### Database Statistics
- **GET** `/api/database/stats`
- **Returns**: Stats for both SQL and vector databases

## Prompt Templates

### System Prompt
```
You are a helpful assistant for exchange students at the IT University of Copenhagen (ITU). 
Your role is to provide clear, accurate, and student-friendly information about ITU courses, programs, and services.

Guidelines:
- Answer questions based ONLY on the provided context
- For course information, prioritize structured data (SQL results) over general descriptions
- Be specific: mention course codes, ECTS credits, semesters, and instructors when available
- For exchange students, clearly indicate which courses are available for exchange
- If information is not in the context, acknowledge this honestly
- Use a friendly, conversational tone suitable for international students
- Cite specific sources when mentioning courses or policies
- Format course information clearly (course code, title, ECTS, semester)
- If multiple courses match, present them in a clear list format
```

### User Prompt Template
```
User Question: {query}

Context Information:
{formatted_context}

Please provide a helpful, accurate response to the user's question based on the context above. 
If the context contains course information, format it clearly with course codes and key details.
If the context contains general ITU information, synthesize it into a clear answer.
```

## Scoring and Ranking

### SQL Relevance Scoring
- Title match: 0.4 points
- Description/abstract keyword matches: 0.2 points
- Course code match: 0.3 points
- Total: Normalized to 0.0-1.0

### Vector Similarity Scoring
- FAISS cosine similarity: Already 0.0-1.0
- Direct use of similarity scores

### Hybrid Scoring
- SQL weight: 0.6 (prioritize factual course data)
- Vector weight: 0.4 (supplement with general info)
- Combined score used for ranking

## Context Truncation

To prevent token overflow:
- SQL results: Max 5 courses
- Vector results: Max 3 chunks
- Results sorted by relevance before truncation

## Error Handling

- **Database not found**: Graceful degradation (warns but continues)
- **LLM unavailable**: Falls back to template-based responses
- **Empty results**: Returns helpful message asking for clarification
- **Query classification errors**: Defaults to HYBRID type

## Future Enhancements

Potential improvements:
1. **Conversation context**: Track previous queries for follow-up questions
2. **Multi-turn dialogue**: Maintain conversation state
3. **Citation tracking**: Link responses to specific sources
4. **Confidence scores**: Indicate certainty of responses
5. **Query expansion**: Use synonyms and related terms
6. **Re-ranking**: Apply more sophisticated ranking algorithms
7. **Feedback loop**: Learn from user interactions

## Testing

To test the RAG pipeline:

```python
from rag_pipeline import RAGPipeline
from vector_db import ITUVectorDatabase
from course_db import CourseDatabase

# Initialize
vector_db = ITUVectorDatabase()
vector_db.load_database()
course_db = CourseDatabase()
rag = RAGPipeline(vector_db, course_db)

# Test query
query = "Which AI courses are available for exchange students?"
context = rag.retrieve(query)
response = rag.generate_response(query, context)
print(response)
```

## Dependencies

- `faiss-cpu`: Vector similarity search
- `sentence-transformers`: Embedding generation
- `sqlite3`: Course database
- `openai`: LLM responses (optional)
- `flask`: Web framework

