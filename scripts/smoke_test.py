"""Simple smoke test for RAG pipeline and course DB.

Run from repo root: python3 scripts/smoke_test.py
"""
import sys
from pathlib import Path
# Ensure repo root is on sys.path so imports work when running this script from scripts/
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from course_db import CourseDatabase
from rag_pipeline import RAGPipeline

def run():
    db = CourseDatabase()
    rag = RAGPipeline(course_db=db, vector_db=None)

    queries = [
        "tell me about advanced machine learning course",
        "15 ects",
        "autumn semester",
        "find me courses that start in autumn semester",
        "Which courses are available for exchange students in Autumn 2026?"
    ]

    for q in queries:
        qtype, meta = rag.classify_query(q)
        merged = rag.retrieve(q, sql_k=5, vector_k=0)
        sql_count = len(merged.get('sql_results', []))
        vector_count = len(merged.get('vector_results', []))
        print(f"Query: {q!r}\n  -> Type: {qtype.value}, Meta: {meta}\n  -> SQL results: {sql_count}, Vector results: {vector_count}\n")

if __name__ == '__main__':
    run()
