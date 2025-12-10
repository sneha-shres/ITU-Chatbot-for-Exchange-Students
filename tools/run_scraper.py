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
