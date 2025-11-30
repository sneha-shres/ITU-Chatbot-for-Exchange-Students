from src.core.rag_pipeline import RAGPipeline, QueryType
from src.database.course_db import CourseDatabase

# SQL-only queries (course-specific)
sql_examples = [
    "give me 15 ects courses",
    "how many credits/ects do i get if i enroll in the course KSADAPS1KU", 
    "what options in courses do i have if i want to specialize in machine learning",
    "what is the least ects course available" 
]

# Vector-only queries (general ITU information)
vector_examples = [
    "How do I apply for admission?",
    "Tell me about campus life and student housing",
    "What research opportunities are available?",
    "How can I apply as an exchange student?"
]

# Hybrid queries (both courses and general info)
hybrid_examples = [
    "I'm interested in data science courses and want to know about housing and campus life",
    "What are the best machine learning courses available? Also tell me about student research opportunities",
    "Tell me about advanced programming courses and application deadlines for exchange students"
]

course_db = CourseDatabase()
print("Course DB path:", course_db.db_path)
print("Course DB stats (if available):", course_db.get_database_stats())

rag = RAGPipeline(vector_db=None, course_db=course_db)

def test_queries(query_list, category):
    print(f"\n{'='*70}")
    print(f"{category} QUERIES")
    print(f"{'='*70}")
    for q in query_list:
        qtype, meta = rag.classify_query(q)
        print(f"\nQuery: {q}")
        print(f"Classified as: {qtype.value.upper()}, meta: {meta}")
        sql_res = rag.query_sql(q, meta, k=10)
        print(f"SQL results: {len(sql_res)}")
        for i, r in enumerate(sql_res[:5], 1):
            title = r.get('course_title') or r.get('course_code')
            ects = r.get('ects')
            code = r.get('course_code')
            print(f"  {i}. {title} ({code}) - {ects} ECTS")

# Test each category
test_queries(sql_examples, "📚 SQL-ONLY")
test_queries(vector_examples, "🌐 VECTOR-ONLY")
test_queries(hybrid_examples, "🔀 HYBRID")

print('\nTest complete')
