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
from scraper import ITUWebScraper
from vector_db import ITUVectorDatabase, chunk_text
from sql_store import SQLStore


def main():
  print("ITU Website Scraper and Vector Database Builder")
  print("=" * 60)

  # Step 1: Scrape the website
  print("\nStep 1: Scraping site...")
  scraper = ITUWebScraper()
  scraped_data = scraper.scrape_all_pages(max_pages=None)  # No limit - scrape all pages

  if not scraped_data:
    print("❌ No data scraped. Exiting.")
    return

  # Save JSON snapshot for reference
  json_path = "itu_scraped_data.json"
  scraper.save_to_json(json_path)
  
  # Save URLs list for easy reference
  scraper.save_urls_list("itu_scraped_urls.txt")

  # Step 2: Store to SQLite
  print("\nStep 2: Storing pages and chunks to SQLite (itu_content.db)...")
  store = SQLStore("itu_content.db")
  total_chunks = 0
  for page in scraped_data:
    page_id = store.upsert_page(page)
    chunks = chunk_text(page.get('full_text', ''), max_tokens=500, overlap=100)
    indexed_chunks = [(i, ch) for i, ch in enumerate(chunks) if len(ch.split()) >= 20]
    if indexed_chunks:
      store.insert_chunks(page_id, indexed_chunks)
      total_chunks += len(indexed_chunks)
  print(f"Stored {len(scraped_data)} pages and {total_chunks} chunks")

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
