# Policy Pulse - RAG Pipeline for Research-to-Policy Translation

**Policy Pulse** is an AI-powered system that transforms complex scientific research papers into actionable policy briefs using a comprehensive 5-step Retrieval-Augmented Generation (RAG) pipeline. The system combines multimodal document processing, semantic analysis, and intelligent content generation to bridge the gap between academic research and policy-making.

## Project Structure

The repository is organized as follows:

- **core_pipeline/**: Contains the 5-step RAG processing pipeline with modules for document processing, content analysis, vector embedding, query processing, and policy brief generation.
- **templates/**: Flask templates for the web interface.
- **app.py**: Main Flask application that integrates all pipeline components with real-time WebSocket updates.

## Pipeline Overview

```
Research Paper (PDF) 
    ↓
[1. Document Processing] → Extract text, tables, images, and metadata
    ↓
[2. Content Analysis] → Semantic chunking and structure understanding  
    ↓
[3. Vector Embedding] → Create searchable FAISS vector database
    ↓
[4. Query Processing] → Interactive Q&A with research documents
    ↓
[5. Policy Brief Generation] → Professional policy documents with visualizations
```

## Core Features

- **Multimodal Processing**: Extracts and processes text, tables, images, and document structure
- **Semantic Chunking**: Intelligent content segmentation preserving context and relationships
- **Vector Search**: FAISS-powered similarity search with hybrid keyword matching
- **Interactive Q&A**: Ask questions about research documents with confidence scoring
- **Professional Output**: Generate policy briefs in HTML, DOCX, and JSON formats
- **Real-time Updates**: WebSocket-powered status updates during processing
- **Session Management**: Document key system for resuming work across sessions

## Technologies Used

### Backend Pipeline
- **Python 3.8+** - Core processing language
- **Flask + SocketIO** - Web framework with real-time updates
- **OpenAI GPT-4** - Language model for content generation
- **FAISS** - Vector similarity search
- **sentence-transformers** - Text embeddings

### Document Processing
- **PyMuPDF (fitz)** - PDF parsing and extraction
- **Unstructured** - Document layout analysis
- **Pillow** - Image processing
- **python-docx** - Document generation

### Storage & Infrastructure
- **Firebase** - Document storage and realtime database
- **Google Cloud** - Vector database storage
- **matplotlib/seaborn** - Data visualization

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Firebase service account key
- OpenAI API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/carlos1021/REM2024Website.git
   cd REM2024Website
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file with:
   OPENAI_API_KEY=your_openai_api_key
   FIREBASE_SERVICE_KEY=your_base64_encoded_firebase_key
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the web interface:**
   Open your browser to `http://localhost:5000`

### Using the Pipeline

1. **Upload Document**: Upload a PDF research paper
2. **Content Analysis**: Analyze document structure and create semantic chunks  
3. **Vector Embedding**: Create searchable vector database
4. **Interactive Q&A**: Ask questions about the research
5. **Generate Brief**: Create professional policy brief with custom settings

## Core Pipeline Components

### 1. Document Processing (`document_processor.py`)
- PDF parsing with text, table, and image extraction
- Metadata extraction (title, author, language)
- Semantic chunking with context preservation
- Firebase storage integration

### 2. Content Analysis (`content_analyzer.py`) 
- Document structure analysis and section identification
- Enhanced semantic chunking with academic paper awareness
- Multi-modal relationship mapping between text and images
- Context-aware content enhancement

### 3. Vector Embedding (`vector_embedder.py`)
- FAISS vector database creation and management
- Hybrid search (semantic + keyword) capabilities
- Scalable embedding storage and retrieval
- Multi-document database support

### 4. Query Processing (`query_processor.py`)
- Intelligent query understanding and expansion
- Evidence retrieval with relevance scoring
- Interactive Q&A with confidence assessment
- Follow-up question generation

### 5. Policy Brief Generation (`policy_brief_generator.py`)
- Professional policy document structure
- Evidence-based content generation with citations
- Data visualization integration (charts, word clouds)
- Multiple export formats (HTML, DOCX, JSON)

## Document Key System

Each processed document receives a unique key (e.g., `doc_20250606_021624_zN9yWKgdIuiz9yGt_e7fcb303eb0c`) that allows users to:
- Resume work across different sessions
- Share document access securely
- Track processing history and generated briefs

## API Endpoints

- `POST /api/upload` - Upload and process PDF documents
- `POST /api/analyze/<document_key>` - Analyze document content
- `POST /api/embed/<document_key>` - Create vector embeddings
- `POST /api/ask` - Ask questions about documents
- `POST /api/generate-brief` - Generate policy briefs
- `POST /api/use-document-key` - Load existing documents

## Contributing

This project represents a comprehensive RAG pipeline for academic-to-policy translation. Contributions are welcome for:
- Additional document format support
- Enhanced visualization capabilities  
- Improved policy brief templates
- Performance optimizations

## License

This project is part of the REM2024 research initiative focused on bridging academic research and policy-making through AI-powered document analysis.
