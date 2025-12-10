#!/usr/bin/env python3
"""
Complete ITU Website Scraper and Vector Database Builder

This script will:
1. Scrape pages from the base site (excluding blocked areas)
2. Persist pages and chunks into SQLite (itu_content.db)
3. Create embeddings for chunks and build a FAISS vector index

Usage:
    python run_scraper.py
"""

import os
# Load environment variables from .env if present
try:
  from dotenv import load_dotenv
  load_dotenv()
except Exception:
  pass
from utils.scraper import ITUWebScraper
from database.vector_db import ITUVectorDatabase, chunk_text
from database.sql_store import SQLStore


def main():
  print("ITU Website Scraper and Vector Database Builder")
  print("=" * 60)

  # Step 1: Scrape the website
  print("\nStep 1: Scraping site...")
  scraper = ITUWebScraper()
  scraped_data = scraper.scrape_all_pages(max_pages=None)  # No limit - scrape all pages
  print(scraped_data)
  


  
  if not scraped_data:
    print("❌ No data scraped. Exiting.")
    return

  # Save JSON snapshot for reference in data/vectors
  json_path = os.path.join('data', 'vectors', 'itu_scraped_data.json')
  scraper.save_to_json(json_path)
  
  # Save URLs list for easy reference
  scraper.save_urls_list(os.path.join('data', 'itu_scraped_urls.txt'))



  import json

 # Step 2.5: Load courses from JSON and add to scraped_data
  print("\nStep 2.5: Loading courses from JSON file...")
  courses_json_path = "data/courses/courses.json"
  if os.path.exists(courses_json_path):
    try:
      with open(courses_json_path, 'r', encoding='utf-8') as f:
        courses_data = json.load(f)

      print(f"  Loaded {len(courses_data)} courses from JSON")

      # Convert courses to scraped_data format
      for course in courses_data:
        # Create a text representation optimized for semantic search
        course_text_parts = []

        # Title and code - put at the beginning for better searchability
        title = course.get('course_title', '')
        code = course.get('course_code', '')

        # Start with structured course information for easy querying
        if title:
          course_text_parts.append(f"Course Title: {title}")
          course_text_parts.append(f"ITU offers the course: {title}")
        if code:
          course_text_parts.append(f"Course Code: {code}")

        # Semester information - make it very searchable
        semester = course.get('semester', '')
        if semester:
          course_text_parts.append(f"Semester: {semester}")
          course_text_parts.append(f"This course is offered in {semester}")
          # Add variations for better search
          if 'Spring' in semester:
            course_text_parts.append(f"Spring semester course")
            course_text_parts.append(f"Available in Spring")
          if 'Autumn' in semester:
            course_text_parts.append(f"Autumn semester course")
            course_text_parts.append(f"Available in Autumn")
          if 'Summer' in semester:
            course_text_parts.append(f"Summer semester course")
            course_text_parts.append(f"Available in Summer")

        # Key details
        if course.get('ects'):
          course_text_parts.append(f"ECTS: {course.get('ects')} ECTS credits")
        if course.get('language'):
          course_text_parts.append(f"Language: {course.get('language')}")
        if course.get('level'):
          level = course.get('level', '')
          course_text_parts.append(f"Level: {level}")
          # Add level variations for search
          if 'MSc' in level or 'Master' in level:
            course_text_parts.append("Master's level course")
          if 'BSc' in level or 'Bachelor' in level:
            course_text_parts.append("Bachelor's level course")

        # Exchange availability - make it clear
        if course.get('offered_exchange') == 'yes':
          course_text_parts.append("Available for exchange students: Yes")
          course_text_parts.append("This course is available for exchange students")

        # Abstract/Description
        abstract = course.get('abstract', '')
        description = course.get('description', '')
        if abstract:
          course_text_parts.append(f"Course Abstract: {abstract}")
        if description:
          course_text_parts.append(f"Course Description: {description}")

        # Learning outcomes
        learning_outcomes = course.get('learning_outcomes', [])
        if learning_outcomes:
          course_text_parts.append("Learning Outcomes:")
          for outcome in learning_outcomes:
            course_text_parts.append(f"  - {outcome}")

        # Prerequisites
        if course.get('formal_prerequisites'):
          course_text_parts.append(f"Prerequisites: {course.get('formal_prerequisites')}")

        # Teachers
        teachers = course.get('teachers', [])
        if teachers:
          course_text_parts.append(f"Instructors: {', '.join(teachers)}")

        # Programme information
        programme = course.get('programme_name', '')
        if programme:
          course_text_parts.append(f"Programme: {programme}")

        # Create course document in scraped_data format
        # Use title as the main identifier
        course_doc = {
          'url': f"https://learnit.itu.dk/course/{code}" if code else "https://learnit.itu.dk/",
          'title': title or 'Course',
          'full_text': '\n'.join(course_text_parts),
          'headings': [title, f"Semester: {semester}"] if title and semester else ([title] if title else []),
          'paragraphs': [abstract, description] if abstract or description else []
        }



        scraped_data.append(course_doc)

      print(f"  Added {len(courses_data)} courses to scraped data")
      print(f"  Total documents for embedding: {len(scraped_data)}")
    except Exception as e:
      print(f"  ⚠️  Error loading courses from JSON: {e}")
      print(f"  Continuing with scraped data only...")
  else:
    print(f"  ⚠️  Courses JSON file not found at {courses_json_path}")
    print(f"  Continuing with scraped data only...")

  # print(scraped_data)
  


  # Step 3: Build vector index
  print("\nStep 3: Building FAISS vector database...")
  vdb = ITUVectorDatabase()
  embeddings, metadata = vdb.create_embeddings(scraped_data, max_tokens=500, overlap=100)
  vdb.metadata = metadata
  vdb.build_index(embeddings)
  vdb.save_database()

  stats = vdb.get_database_stats()
  print("\nVector DB Stats:")
  print(f"  Total vectors: {stats.get('total_vectors')}")
  print(f"  Dimension: {stats.get('dimension')}")
  print(f"  Model: {stats.get('model_name')}")

  print("\n✅ All steps completed successfully!")


if __name__ == "__main__":
  main()
