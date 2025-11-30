import sqlite3
from typing import List, Dict, Optional, Tuple
import os
import json

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS pages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  url TEXT UNIQUE NOT NULL,
  title TEXT,
  metadata_json TEXT,
  full_text TEXT,
  word_count INTEGER,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  page_id INTEGER NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  word_count INTEGER,
  embedding BLOB, -- optional persisted embedding
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(page_id) REFERENCES pages(id)
);

CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url);
CREATE INDEX IF NOT EXISTS idx_chunks_page_id ON chunks(page_id);
"""

class SQLStore:
  def __init__(self, db_path: str = "itu_content.db") -> None:
    self.db_path = db_path
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    self._init_db()

  def _connect(self) -> sqlite3.Connection:
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row
    return conn

  def _init_db(self) -> None:
    with self._connect() as conn:
      conn.executescript(SCHEMA)

  def upsert_page(self, page: Dict) -> int:
    """Insert or replace a page. Returns page_id."""
    metadata_json = json.dumps(page.get('metadata', {}), ensure_ascii=False)
    with self._connect() as conn:
      cur = conn.cursor()
      cur.execute(
        """
        INSERT INTO pages(url, title, metadata_json, full_text, word_count)
        VALUES(?,?,?,?,?)
        ON CONFLICT(url) DO UPDATE SET
          title=excluded.title,
          metadata_json=excluded.metadata_json,
          full_text=excluded.full_text,
          word_count=excluded.word_count
        """,
        (
          page['url'],
          page.get('title'),
          metadata_json,
          page.get('full_text'),
          page.get('word_count', 0),
        ),
      )
      page_id = cur.lastrowid
      # If conflict (existing), fetch id
      if page_id == 0:
        cur.execute("SELECT id FROM pages WHERE url=?", (page['url'],))
        row = cur.fetchone()
        page_id = row['id'] if row else 0
      return page_id

  def insert_chunks(self, page_id: int, chunks: List[Tuple[int, str]]) -> None:
    """Insert chunks for a page. chunks = list of (chunk_index, text)."""
    with self._connect() as conn:
      conn.executemany(
        """
        INSERT INTO chunks(page_id, chunk_index, text, word_count)
        VALUES(?,?,?,?)
        """,
        [(page_id, idx, text, len(text.split())) for idx, text in chunks],
      )

  def fetch_pages(self, limit: int = 10) -> List[Dict]:
    with self._connect() as conn:
      cur = conn.execute(
        "SELECT id, url, title, metadata_json, word_count FROM pages ORDER BY id DESC LIMIT ?",
        (limit,),
      )
      rows = cur.fetchall()
      return [dict(row) for row in rows]

  def fetch_chunks_by_page(self, page_id: int) -> List[Dict]:
    with self._connect() as conn:
      cur = conn.execute(
        "SELECT id, chunk_index, text, word_count FROM chunks WHERE page_id=? ORDER BY chunk_index ASC",
        (page_id,),
      )
      rows = cur.fetchall()
      return [dict(row) for row in rows]
