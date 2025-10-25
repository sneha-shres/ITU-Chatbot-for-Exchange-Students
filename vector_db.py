import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 100) -> List[str]:
  """
  Simple token-approximate chunker by words with overlap.
  max_tokens and overlap are in words (approximate tokens for English).
  """
  if not text:
    return []
  words = text.split()
  if len(words) <= max_tokens:
    return [' '.join(words)]

  chunks = []
  start = 0
  while start < len(words):
    end = min(start + max_tokens, len(words))
    chunk_words = words[start:end]
    if not chunk_words:
      break
    chunks.append(' '.join(chunk_words))
    if end == len(words):
      break
    start = max(0, end - overlap)
  return chunks


class ITUVectorDatabase:
  def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
    self.model_name = model_name
    self.model = SentenceTransformer(model_name)
    self.index = None
    self.documents = []
    self.metadata = []
    # all-MiniLM-L6-v2 has 384-dim embeddings
    self.dimension = 384
  
  def load_scraped_data(self, json_file: str) -> List[Dict]:
    try:
      with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
      logger.info(f"Loaded {len(data)} documents from {json_file}")
      return data
    except FileNotFoundError:
      logger.error(f"File {json_file} not found")
      return []
    except json.JSONDecodeError as e:
      logger.error(f"Error parsing JSON: {e}")
      return []

  def create_embeddings(self, scraped_data: List[Dict], max_tokens: int = 500, overlap: int = 100) -> Tuple[np.ndarray, List[Dict]]:
    all_embeddings = []
    all_metadata = []

    logger.info("Creating embeddings for documents...")

    for doc_idx, doc in enumerate(scraped_data):
      chunks = chunk_text(doc.get('full_text', ''), max_tokens=max_tokens, overlap=overlap)
      for chunk_idx, chunk in enumerate(chunks):
        if len(chunk.strip().split()) < 20:  # Skip very short chunks
          continue
        embedding = self.model.encode(chunk)
        all_embeddings.append(embedding)
        metadata = {
          'doc_id': doc_idx,
          'chunk_id': chunk_idx,
          'url': doc.get('url'),
          'title': doc.get('title'),
          'text': chunk,
          'word_count': len(chunk.split()),
          'headings': doc.get('headings', []),
          'paragraphs': doc.get('paragraphs', [])
        }
        all_metadata.append(metadata)

    embeddings_array = np.array(all_embeddings).astype('float32')
    logger.info(f"Created {len(embeddings_array)} embeddings")

    return embeddings_array, all_metadata

  def build_index(self, embeddings: np.ndarray):
    logger.info("Building FAISS index...")
    self.index = faiss.IndexFlatIP(self.dimension)
    faiss.normalize_L2(embeddings)
    self.index.add(embeddings)
    logger.info(f"FAISS index built with {self.index.ntotal} vectors")

  def save_database(self, index_path: str = "itu_vector_index.faiss", metadata_path: str = "itu_metadata.pkl"):
    if self.index is None:
      logger.error("No index to save")
      return
    faiss.write_index(self.index, index_path)
    logger.info(f"Saved FAISS index to {index_path}")
    with open(metadata_path, 'wb') as f:
      pickle.dump(self.metadata, f)
    logger.info(f"Saved metadata to {metadata_path}")

  def load_database(self, index_path: str = "itu_vector_index.faiss", metadata_path: str = "itu_metadata.pkl"):
    try:
      self.index = faiss.read_index(index_path)
      logger.info(f"Loaded FAISS index from {index_path}")
      with open(metadata_path, 'rb') as f:
        self.metadata = pickle.load(f)
      logger.info(f"Loaded metadata from {metadata_path}")
    except Exception as e:
      logger.error(f"Error loading database: {e}")

  def search(self, query: str, k: int = 5) -> List[Dict]:
    if self.index is None:
      logger.error("No index loaded")
      return []
    query_embedding = self.model.encode(query)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_embedding)
    scores, indices = self.index.search(query_embedding, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
      if 0 <= idx < len(self.metadata):
        result = self.metadata[idx].copy()
        result['similarity_score'] = float(score)
        results.append(result)
    return results

  def get_database_stats(self) -> Dict:
    if self.index is None:
      return {}
    return {
      'total_vectors': self.index.ntotal,
      'dimension': self.dimension,
      'model_name': self.model_name,
      'total_documents': len(set(meta['doc_id'] for meta in self.metadata))
    }

def main():
    """Main function to build the vector database"""
    # Initialize vector database
    vector_db = ITUVectorDatabase()
    
    print("🚀 Building ITU Vector Database...")
    
    # Load scraped data
    scraped_data = vector_db.load_scraped_data("itu_scraped_data.json")
    
    if not scraped_data:
        print("❌ No scraped data found. Please run the scraper first.")
        return
    
    # Create embeddings
    embeddings, metadata = vector_db.create_embeddings(scraped_data)
    
    # Store metadata
    vector_db.metadata = metadata
    
    # Build FAISS index
    vector_db.build_index(embeddings)
    
    # Save database
    vector_db.save_database()
    
    # Print statistics
    stats = vector_db.get_database_stats()
    print("\n📊 Vector Database Statistics:")
    print(f"   Total vectors: {stats['total_vectors']:,}")
    print(f"   Dimension: {stats['dimension']}")
    print(f"   Model: {stats['model_name']}")
    print(f"   Total documents: {stats['total_documents']}")
    
    print("\n✅ Vector database built successfully!")
    
    # Test search
    print("\n🔍 Testing search functionality...")
    test_queries = [
        "computer science programs",
        "research opportunities",
        "admission requirements",
        "student life"
    ]
    
    for query in test_queries:
        results = vector_db.search(query, k=3)
        print(f"\nQuery: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title'][:60]}... (score: {result['similarity_score']:.3f})")

if __name__ == "__main__":
    main()
