#!/usr/bin/env python3
"""
Complete ITU Website Scraper and Vector Database Builder

"""

import os

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

  #Scrape the website
  scraper = ITUWebScraper()
  scraped_data = scraper.scrape_all_pages(max_pages=None)  # No limit - scrape all pages
  print(scraped_data)
  
  
  if not scraped_data:
    print(" No data scraped. Exiting.")
    return

  json_path = os.path.join('data', 'vectors', 'itu_scraped_data.json')
  scraper.save_to_json(json_path)
  
  scraper.save_urls_list(os.path.join('data', 'itu_scraped_urls.txt'))

  #build vector index
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

  print("\n All steps completed successfully!")


if __name__ == "__main__":
  main()
