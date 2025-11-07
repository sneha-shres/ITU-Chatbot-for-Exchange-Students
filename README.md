# ITU Chatbot with Vector Database

A modern chatbot interface for the IT University of Copenhagen (ITU) that uses web scraping and vector embeddings to provide intelligent responses about ITU programs, research, and services.

## Features

- 🕷️ **Web Scraping**: Automatically scrapes ITU website content (excluding news)
- 🧠 **Vector Database**: Uses FAISS for fast similarity search
- 🤖 **Smart Chatbot**: Provides contextual responses based on ITU knowledge
- 🎨 **Modern UI**: Beautiful, responsive chat interface
- 🔍 **Semantic Search**: Finds relevant information using embeddings

## Project Structure

```
Chatbot_ITU/
├── app.py                      # Flask backend server
├── scraper.py                  # Web scraper for ITU website
├── vector_db.py                # FAISS vector database implementation
├── sql_store.py                # SQL database storage utilities
├── course_db.py                # Course database interface
├── rag_pipeline.py             # RAG implementation
├── run_scraper.py              # Main script to run everything
├── requirements.txt            # Python dependencies
├── package.json                # Project configuration
├── README.md                   # This file
├── RAG_ARCHITECTURE.md         # RAG system documentation
│
├── Courses/                    # Course-related modules
│   ├── __pycache__/           # Python cache files
│   ├── course_scraper.py      # Course-specific scraper
│   ├── csv_to_sqlite.py       # CSV to SQLite converter
│   ├── course_pages/          # Scraped course HTML pages
│   │   └── [144 HTML files]   # Individual course page files
│   └── output/                # Course data outputs
│       ├── courses.csv        # Course data in CSV format
│       ├── courses.db         # Course data in SQLite database
│       ├── courses.json       # Course data in JSON format
│       └── read_csv.ipynb     # Jupyter notebook for data analysis
│
├── scripts/                    # Utility scripts
│   └── data_scraper.py        # Additional data scraping utilities
│
├── templates/                  # Flask HTML templates
│   └── index.html             # Main chat interface
│
├── static/                     # Static assets
│   ├── styles.css             # CSS styling
│   └── script.js              # Frontend JavaScript
│
├── itu_metadata.pkl           # Pickled metadata
├── itu_scraped_data.json      # Scraped ITU website data
├── itu_scraped_urls.txt       # List of scraped URLs
└── itu_vector_index.faiss     # FAISS vector index file
```

## Installation

1. **Clone or download the project**
2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Scrape ITU Website and Build Vector Database

```bash
python run_scraper.py
```

This will:
- Scrape ITU website (excluding news pages)
- Create embeddings using sentence transformers
- Build FAISS vector database
- Test the search functionality

### 2. Start the Chatbot

```bash
python app.py
```

Then open your browser and go to: `http://localhost:5000`

## API Endpoints

- `GET /` - Main chatbot interface
- `POST /api/chat` - Send messages to chatbot
- `GET /api/health` - Health check
- `GET /api/history` - Get conversation history
- `POST /api/search` - Search knowledge base
- `GET /api/database/stats` - Get database statistics

## Example Questions

Try asking the chatbot:

- "What computer science programs does ITU offer?"
- "How do I apply for admission?"
- "What research opportunities are available?"
- "Tell me about student life at ITU"
- "What are the admission requirements?"

## Configuration

### Scraper Settings

In `scraper.py`, you can modify:
- `max_pages`: Number of pages to scrape (default: 30)
- `base_url`: ITU website URL
- News filtering: Automatically skips news pages

### Vector Database Settings

In `vector_db.py`, you can modify:
- `model_name`: Sentence transformer model (default: "all-MiniLM-L6-v2")
- `max_length`: Text chunk size (default: 512)
- `k`: Number of search results (default: 5)

## Technologies Used

- **Backend**: Flask, Python
- **Web Scraping**: BeautifulSoup, Requests
- **Vector Database**: FAISS
- **Embeddings**: Sentence Transformers
- **Frontend**: HTML, CSS, JavaScript
- **Styling**: Modern CSS with gradients and animations

## Notes

- The scraper respects robots.txt and includes delays between requests
- News pages are automatically filtered out to focus on core content
- The vector database is built locally and can be reused
- All scraped data is saved as JSON for inspection

## Troubleshooting

1. **Import errors**: Make sure all dependencies are installed
2. **Scraping issues**: Check internet connection and ITU website availability
3. **Vector database errors**: Ensure sufficient disk space for embeddings
4. **Port conflicts**: Change port in `app.py` if 5000 is occupied

## License

MIT License - feel free to use and modify as needed.