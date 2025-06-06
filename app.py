#!/usr/bin/env python3
"""
RAG Pipeline Flask Application

This Flask app integrates all 5 steps of the RAG pipeline:
1. Document Processing (PDF upload & processing)
2. Content Analysis (Semantic chunking & structure)
3. Vector Embedding (FAISS + Firebase storage)
4. Query Processing (Interactive Q&A)
5. Policy Brief Generation (Professional documents)

Features:
- Real-time status updates via WebSocket
- Document key management for session persistence
- Interactive Q&A with research documents
- Customizable policy brief generation
- Multiple export formats
- RESTful API design
"""

import os
import sys
import json
import logging
import traceback
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import threading
import time

# Flask and extensions
from flask import Flask, request, jsonify, render_template, send_file, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename
import uuid

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add core_pipeline to path
sys.path.append(str(Path(__file__).parent / 'core_pipeline'))

# Import RAG pipeline components
from core_pipeline.document_processor import DocumentProcessor, ProcessingResult
from core_pipeline.content_analyzer import ContentAnalyzer, AnalysisResult
from core_pipeline.vector_embedder import VectorEmbedder, VectorDatabase
from core_pipeline.query_processor import QueryProcessor, ask_question, ResponseResult
from core_pipeline.policy_brief_generator import (
    PolicyBriefGenerator, 
    BriefConfig, 
    OutputFormat,
    generate_policy_brief
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__, static_folder='public', static_url_path='')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Enable CORS and WebSocket
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5000"])
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global storage for active sessions and processing status
active_sessions = {}
processing_status = {}

# Firebase database reference
database_ref = None

# File upload configuration
ALLOWED_EXTENSIONS = {'pdf'}


def init_firebase():
    """Initialize Firebase using environment variable"""
    global database_ref
    try:
        # Check if Firebase is already initialized
        firebase_admin.get_app()
        logger.info("Using existing Firebase app")
    except ValueError:
        # Firebase not initialized, initialize it
        try:
            firebase_key_base64 = os.getenv("FIREBASE_SERVICE_KEY")
            if not firebase_key_base64:
                logger.error("FIREBASE_SERVICE_KEY environment variable not found")
                return False
            
            # Decode the base64 service key
            firebase_key_json = base64.b64decode(firebase_key_base64).decode('utf-8')
            firebase_service_account = json.loads(firebase_key_json)
            
            # Initialize Firebase
            cred = credentials.Certificate(firebase_service_account)
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'rem2024-f429b.appspot.com',
                'databaseURL': 'https://rem2024-f429b-default-rtdb.firebaseio.com'
            })
            logger.info("Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            return False
    
    # Get database reference
    try:
        database_ref = db.reference()
        logger.info("Firebase database reference ready")
        return True
    except Exception as e:
        logger.error(f"Failed to get Firebase database reference: {e}")
        return False


class StatusManager:
    """Manages real-time status updates for the frontend"""
    
    def __init__(self, session_id: str, socketio_instance):
        self.session_id = session_id
        self.socketio = socketio_instance
        self.status_log = []
    
    def log(self, message: str, level: str = "info", data: Dict = None):
        """Log a status message and emit to frontend"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "data": data or {}
        }
        
        self.status_log.append(log_entry)
        
        # Emit to specific session room
        self.socketio.emit('status_update', log_entry, room=self.session_id)
        
        # Also log to server console
        if level == "error":
            logger.error(f"[{self.session_id}] {message}")
        elif level == "warning":
            logger.warning(f"[{self.session_id}] {message}")
        else:
            logger.info(f"[{self.session_id}] {message}")
    
    def progress(self, current: int, total: int, message: str = ""):
        """Update progress bar"""
        percentage = int((current / total) * 100) if total > 0 else 0
        self.log(f"Progress: {current}/{total} ({percentage}%) {message}", "progress", {
            "current": current,
            "total": total,
            "percentage": percentage
        })


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


# WebSocket events
@socketio.on('connect')
def handle_connect():
    session_id = get_session_id()
    join_room(session_id)
    emit('connected', {'session_id': session_id})
    logger.info(f"Client connected to session: {session_id}")


@socketio.on('disconnect')
def handle_disconnect():
    session_id = get_session_id()
    leave_room(session_id)
    logger.info(f"Client disconnected from session: {session_id}")


# API Routes

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    """Serve the upload page"""
    return render_template('upload.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "firebase_connected": database_ref is not None,
        "components": {
            "document_processor": True,
            "content_analyzer": True,
            "vector_embedder": True,
            "query_processor": True,
            "policy_brief_generator": True
        }
    })


@app.route('/api/session')
def get_session_info():
    """Get current session information"""
    session_id = get_session_id()
    return jsonify({
        "session_id": session_id,
        "active_documents": list(active_sessions.get(session_id, {}).keys()),
        "processing_status": processing_status.get(session_id, {})
    })


@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process a PDF document - Step 1"""
    session_id = get_session_id()
    status = StatusManager(session_id, socketio)
    
    try:
        status.log("📄 Starting document upload and processing...")
        
        # Check if file is present
        if 'file' not in request.files:
            status.log("No file provided in request", "error")
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            status.log("No file selected", "error")
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            status.log(f"Invalid file type: {file.filename}", "error")
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(temp_path)
        
        status.log(f"✅ File uploaded: {filename}")
        status.progress(1, 5, "Document uploaded")
        
        # Process document
        status.log("🔍 Processing PDF document...")
        processor = DocumentProcessor()
        result = processor.process_document(temp_path)
        
        if result.errors:
            status.log(f"⚠️  Processing completed with {len(result.errors)} warnings", "warning")
        else:
            status.log("✅ Document processing completed successfully")
        
        status.progress(2, 5, "Document processed")
        
        # Export to Firebase
        status.log("☁️  Uploading to Firebase...")
        firebase_paths = processor.export_results(result)
        
        status.log("✅ Document uploaded to Firebase")
        status.progress(3, 5, "Uploaded to Firebase")
        
        # Store session data
        if session_id not in active_sessions:
            active_sessions[session_id] = {}
        
        document_info = {
            "document_key": result.document_key,
            "filename": filename,
            "upload_time": datetime.now().isoformat(),
            "processing_result": {
                "pages": len(result.text_content),
                "images": len(result.image_urls),
                "tables": len(result.tables),
                "chunks": len(result.semantic_chunks),
                "processing_time": result.processing_stats.get('processing_time_seconds', 0)
            },
            "firebase_paths": firebase_paths,
            "temp_path": temp_path
        }
        
        active_sessions[session_id][result.document_key] = document_info
        
        status.log("🎉 Document processing pipeline completed!")
        status.progress(5, 5, "Complete")
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Could not remove temp file: {e}")
        
        return jsonify({
            "success": True,
            "document_key": result.document_key,
            "filename": filename,
            "processing_stats": result.processing_stats,
            "firebase_paths": firebase_paths,
            "summary": {
                "pages": len(result.text_content),
                "images": len(result.image_urls),
                "tables": len(result.tables),
                "semantic_chunks": len(result.semantic_chunks)
            }
        })
        
    except Exception as e:
        error_msg = f"Document processing failed: {str(e)}"
        status.log(error_msg, "error")
        logger.error(f"Upload error: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500


@app.route('/api/analyze/<document_key>', methods=['POST'])
def analyze_content(document_key):
    """Analyze document content and create enhanced chunks - Step 2"""
    session_id = get_session_id()
    status = StatusManager(session_id, socketio)
    
    try:
        status.log("🧠 Starting content analysis...")
        
        # Check if document exists in session
        if session_id not in active_sessions or document_key not in active_sessions[session_id]:
            status.log("Document not found in session", "error")
            return jsonify({"error": "Document not found"}), 404
        
        status.progress(1, 3, "Initializing analysis")
        
        # Analyze content
        analyzer = ContentAnalyzer()
        analysis_result = analyzer.analyze_document(document_key)
        
        status.log(f"✅ Analysis completed: {len(analysis_result.enhanced_chunks)} chunks created")
        status.progress(2, 3, "Analysis complete")
        
        # Export to Firebase
        firebase_path = analyzer.export_analysis(analysis_result)
        status.log("☁️  Analysis results uploaded to Firebase")
        
        # Update session data
        active_sessions[session_id][document_key]["analysis_result"] = {
            "enhanced_chunks": len(analysis_result.enhanced_chunks),
            "document_structure": len(analysis_result.document_structure),
            "relationships": len(analysis_result.chunk_relationships),
            "firebase_path": firebase_path,
            "analysis_time": analysis_result.analysis_stats.get('analysis_time_seconds', 0)
        }
        
        status.log("🎉 Content analysis completed!")
        status.progress(3, 3, "Complete")
        
        return jsonify({
            "success": True,
            "document_key": document_key,
            "analysis_stats": analysis_result.analysis_stats,
            "firebase_path": firebase_path,
            "summary": {
                "enhanced_chunks": len(analysis_result.enhanced_chunks),
                "document_structure": len(analysis_result.document_structure),
                "relationships": len(analysis_result.chunk_relationships)
            }
        })
        
    except Exception as e:
        error_msg = f"Content analysis failed: {str(e)}"
        status.log(error_msg, "error")
        logger.error(f"Analysis error: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500


@app.route('/api/embed/<document_key>', methods=['POST'])
def create_embeddings(document_key):
    """Create vector embeddings and store in FAISS database - Step 3"""
    session_id = get_session_id()
    status = StatusManager(session_id, socketio)
    
    try:
        status.log("🔮 Starting vector embedding creation...")
        
        # Check if document exists
        if session_id not in active_sessions or document_key not in active_sessions[session_id]:
            status.log("Document not found in session", "error")
            return jsonify({"error": "Document not found"}), 404
        
        status.progress(1, 4, "Initializing embedder")
        
        # Create embeddings
        embedder = VectorEmbedder()
        vector_db = embedder.create_vector_database([document_key])
        
        status.log(f"✅ Vector database created: {vector_db.database_id}")
        status.progress(2, 4, "Embeddings created")
        
        # Upload to Firebase
        status.log("☁️  Uploading vector database to Firebase...")
        embedder.save_to_firebase_storage(vector_db)
        
        status.log("✅ Vector database uploaded to Firebase")
        status.progress(3, 4, "Uploaded to Firebase")
        
        # Update session data
        active_sessions[session_id][document_key]["vector_database"] = {
            "database_id": vector_db.database_id,
            "total_chunks": vector_db.total_chunks,
            "embedding_dimension": vector_db.embedding_dimension,
            "model_name": vector_db.model_name,
            "hybrid_search": vector_db.tfidf_vectorizer is not None
        }
        
        status.log("🎉 Vector embedding completed!")
        status.progress(4, 4, "Complete")
        
        return jsonify({
            "success": True,
            "document_key": document_key,
            "vector_database_id": vector_db.database_id,
            "summary": {
                "total_chunks": vector_db.total_chunks,
                "embedding_dimension": vector_db.embedding_dimension,
                "model_name": vector_db.model_name,
                "hybrid_search_enabled": vector_db.tfidf_vectorizer is not None
            }
        })
        
    except Exception as e:
        error_msg = f"Vector embedding failed: {str(e)}"
        status.log(error_msg, "error")
        logger.error(f"Embedding error: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question_endpoint():
    """Ask a question about the document - Step 4"""
    session_id = get_session_id()
    status = StatusManager(session_id, socketio)
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        document_key = data.get('document_key')
        use_hybrid = data.get('use_hybrid', True)
        max_results = data.get('max_results', 5)
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        if not document_key:
            return jsonify({"error": "Document key is required"}), 400
        
        status.log(f"🤔 Processing question: {question[:50]}...")
        
        # Get vector database ID from session
        if (session_id not in active_sessions or 
            document_key not in active_sessions[session_id] or
            "vector_database" not in active_sessions[session_id][document_key]):
            return jsonify({"error": "Vector database not found. Please run embedding step first."}), 404
        
        vector_db_id = active_sessions[session_id][document_key]["vector_database"]["database_id"]
        
        # Process question
        result = ask_question(
            question=question,
            vector_db_id=vector_db_id,
            use_hybrid=use_hybrid,
            max_results=max_results
        )
        
        status.log(f"✅ Question answered with confidence: {result.confidence_score:.2f}")
        
        return jsonify({
            "success": True,
            "question": question,
            "answer": result.response,
            "confidence_score": result.confidence_score,
            "response_type": result.response_type,
            "quality_assessment": result.quality_assessment.value,
            "citations": result.citations,
            "suggested_followups": result.suggested_followups,
            "processing_stats": result.processing_stats
        })
        
    except Exception as e:
        error_msg = f"Question processing failed: {str(e)}"
        status.log(error_msg, "error")
        logger.error(f"Question error: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500


@app.route('/api/generate-brief', methods=['POST'])
def generate_policy_brief_endpoint():
    """Generate a policy brief from the document - Step 5"""
    session_id = get_session_id()
    status = StatusManager(session_id, socketio)
    
    try:
        data = request.get_json()
        document_key = data.get('document_key')
        config_data = data.get('config', {})
        
        if not document_key:
            return jsonify({"error": "Document key is required"}), 400
        
        status.log("📋 Starting policy brief generation...")
        
        # Check if vector database exists
        if (session_id not in active_sessions or 
            document_key not in active_sessions[session_id] or
            "vector_database" not in active_sessions[session_id][document_key]):
            return jsonify({"error": "Vector database not found. Please run embedding step first."}), 404
        
        vector_db_id = active_sessions[session_id][document_key]["vector_database"]["database_id"]
        
        status.progress(1, 5, "Initializing generator")
        
        # Create configuration
        config = BriefConfig(
            title=config_data.get('title', 'Research Policy Brief'),
            target_audience=config_data.get('target_audience', 'Policy Makers'),
            executive_length=config_data.get('executive_length', 'medium'),
            include_charts=config_data.get('include_charts', True),
            include_data_tables=config_data.get('include_data_tables', True),
            include_wordcloud=config_data.get('include_wordcloud', True),
            color_scheme=config_data.get('color_scheme', 'professional'),
            max_recommendations=config_data.get('max_recommendations', 5),
            citation_style=config_data.get('citation_style', 'policy')
        )
        
        status.progress(2, 5, "Configuration created")
        
        # Generate brief
        generator = PolicyBriefGenerator()
        brief = generator.generate_policy_brief(
            config=config,
            vector_db_id=vector_db_id,
            research_focus=config_data.get('research_focus')
        )
        
        status.log("✅ Policy brief generated successfully")
        status.progress(3, 5, "Brief generated")
        
        # Export in requested format
        output_format = config_data.get('output_format', 'html')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c for c in config.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:30]
        
        filename = f"policy_brief_{safe_title}_{timestamp}.{output_format}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        format_enum = OutputFormat(output_format.lower())
        generator.export_brief(brief, format_enum, output_path)
        
        status.log(f"📄 Brief exported as {output_format.upper()}")
        status.progress(4, 5, "Brief exported")
        
        # Store brief info in session
        brief_info = {
            "title": brief.title,
            "generation_time": brief.metadata['generation_time_seconds'],
            "evidence_sources": brief.metadata['evidence_sources'],
            "total_sections": brief.metadata['total_sections'],
            "filename": filename,
            "output_path": output_path,
            "output_format": output_format
        }
        
        if "policy_briefs" not in active_sessions[session_id][document_key]:
            active_sessions[session_id][document_key]["policy_briefs"] = []
        
        active_sessions[session_id][document_key]["policy_briefs"].append(brief_info)
        
        status.log("🎉 Policy brief generation completed!")
        status.progress(5, 5, "Complete")
        
        return jsonify({
            "success": True,
            "document_key": document_key,
            "brief_title": brief.title,
            "filename": filename,
            "download_url": f"/api/download/{filename}",
            "metadata": brief.metadata,
            "sections": {
                "executive_summary": {
                    "confidence": brief.executive_summary.confidence_score,
                    "length": len(brief.executive_summary.content)
                },
                "key_findings": {
                    "confidence": brief.key_findings.confidence_score,
                    "length": len(brief.key_findings.content)
                },
                "recommendations": {
                    "confidence": brief.policy_recommendations.confidence_score,
                    "length": len(brief.policy_recommendations.content)
                }
            }
        })
        
    except Exception as e:
        error_msg = f"Policy brief generation failed: {str(e)}"
        status.log(error_msg, "error")
        logger.error(f"Brief generation error: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download generated files"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/document/<document_key>')
def get_document_info(document_key):
    """Get information about a specific document"""
    session_id = get_session_id()
    
    if (session_id not in active_sessions or 
        document_key not in active_sessions[session_id]):
        return jsonify({"error": "Document not found"}), 404
    
    doc_info = active_sessions[session_id][document_key]
    
    return jsonify({
        "document_key": document_key,
        "filename": doc_info.get("filename"),
        "upload_time": doc_info.get("upload_time"),
        "processing_result": doc_info.get("processing_result"),
        "analysis_result": doc_info.get("analysis_result"),
        "vector_database": doc_info.get("vector_database"),
        "policy_briefs": doc_info.get("policy_briefs", [])
    })


@app.route('/api/documents')
def list_documents():
    """List all documents in the current session"""
    session_id = get_session_id()
    
    if session_id not in active_sessions:
        return jsonify({"documents": []})
    
    documents = []
    for doc_key, doc_info in active_sessions[session_id].items():
        documents.append({
            "document_key": doc_key,
            "filename": doc_info.get("filename"),
            "upload_time": doc_info.get("upload_time"),
            "has_analysis": "analysis_result" in doc_info,
            "has_embeddings": "vector_database" in doc_info,
            "policy_briefs_count": len(doc_info.get("policy_briefs", []))
        })
    
    return jsonify({"documents": documents})


@app.route('/api/use-document-key', methods=['POST'])
def use_document_key():
    """Load a document using its key (for returning users)"""
    session_id = get_session_id()
    status = StatusManager(session_id, socketio)
    
    try:
        data = request.get_json()
        document_key = data.get('document_key', '').strip()
        
        if not document_key:
            return jsonify({"error": "Document key is required"}), 400
        
        status.log(f"🔍 Looking up document: {document_key}")
        
        # Check if Firebase is available
        if not database_ref:
            return jsonify({"error": "Firebase not available"}), 500
        
        # Check if document exists in Firebase documents collection
        try:
            doc_ref = database_ref.child('documents').child(document_key)
            doc_data = doc_ref.get()
            
            if not doc_data:
                status.log(f"Document {document_key} not found in Firebase documents", "error")
                return jsonify({"error": "Document key not found in database"}), 404
            
            # Document exists! Load it into session
            if session_id not in active_sessions:
                active_sessions[session_id] = {}
            
            # Extract document info from Firebase data
            metadata = doc_data.get('metadata', {})
            content = doc_data.get('content', {})
            images = doc_data.get('images', {})
            
            document_info = {
                "document_key": document_key,
                "filename": metadata.get('title', 'Previously uploaded document'),
                "upload_time": metadata.get('created_at', ''),
                "loaded_from_key": True,
                "processing_result": {
                    "pages": metadata.get('page_count', 0),
                    "images": images.get('count', 0),
                    "tables": len(content.get('tables', [])),
                    "chunks": len(content.get('semantic_chunks', [])),
                    "processing_time": 0
                }
            }
            
            # Check if analysis exists
            analysis_ref = database_ref.child('analyzed_documents').child(document_key)
            analysis_data = analysis_ref.get()
            
            if analysis_data:
                analysis_metadata = analysis_data.get('analysis_metadata', {})
                document_info["analysis_result"] = {
                    "enhanced_chunks": analysis_metadata.get('chunk_count', 0),
                    "document_structure": len(analysis_data.get('document_structure', {})),
                    "relationships": len(analysis_data.get('multimodal_relationships', {})),
                    "analysis_time": 0
                }
                status.log("✅ Found existing analysis data")
            
            # Check if vector database exists for this document
            try:
                embedder = VectorEmbedder()
                
                # First, try to list all vector databases and find one with this document
                available_dbs = embedder.list_vector_databases()
                
                matching_db = None
                for db_info in available_dbs:
                    if document_key in db_info.get('documents', []):
                        matching_db = db_info
                        break
                
                # If not found by document list, try to find by database ID pattern
                if not matching_db:
                    # Vector database IDs often contain the document key or timestamp
                    for db_info in available_dbs:
                        db_id = db_info.get('database_id', '')
                        # Check if the document key is part of the database ID
                        if document_key[:20] in db_id or document_key[4:20] in db_id:
                            matching_db = db_info
                            status.log(f"✅ Found vector database by ID pattern: {db_id}")
                            break
                
                # If still not found, try to load the most recent database created around the document time
                if not matching_db and available_dbs:
                    # Sort by creation time and try the most recent one
                    sorted_dbs = sorted(available_dbs, key=lambda x: x.get('created_at', ''), reverse=True)
                    if sorted_dbs:
                        matching_db = sorted_dbs[0]
                        status.log(f"✅ Using most recent vector database: {matching_db['database_id']}")
                
                if matching_db:
                    document_info["vector_database"] = {
                        "database_id": matching_db['database_id'],
                        "total_chunks": matching_db.get('total_chunks', 0),
                        "embedding_dimension": matching_db.get('embedding_dimension', 384),
                        "model_name": matching_db.get('embedding_model', 'BAAI/bge-small-en-v1.5'),
                        "hybrid_search": True
                    }
                    status.log("✅ Found existing vector database")
                else:
                    # If document exists but no vector DB found, we should still proceed
                    # The user can create embeddings if needed
                    status.log("⚠️  No vector database found, but document exists. You can create embeddings in Step 3.")
                
            except Exception as e:
                logger.warning(f"Could not check vector databases: {e}")
                # Even if vector DB check fails, the document itself exists, so we can proceed
            
            active_sessions[session_id][document_key] = document_info
            
            status.log("✅ Document loaded successfully from key")
            
            return jsonify({
                "success": True,
                "document_key": document_key,
                "message": "Document loaded from key",
                "document_info": document_info
            })
            
        except Exception as e:
            logger.error(f"Error loading document by key: {e}")
            status.log(f"Failed to load document: {str(e)}", "error")
            return jsonify({"error": "Failed to load document from Firebase"}), 500
        
    except Exception as e:
        error_msg = f"Failed to load document: {str(e)}"
        status.log(error_msg, "error")
        return jsonify({"error": error_msg}), 500


# Cleanup function
def cleanup_old_files():
    """Clean up old temporary files"""
    try:
        temp_dir = app.config['UPLOAD_FOLDER']
        current_time = time.time()
        
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                # Remove files older than 24 hours
                if current_time - os.path.getctime(file_path) > 24 * 3600:
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {filename}")
                    except Exception as e:
                        logger.warning(f"Could not remove file {filename}: {e}")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 50MB."}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Initialize Firebase
    if not init_firebase():
        print("❌ Failed to initialize Firebase")
        print("💡 Please check your FIREBASE_SERVICE_KEY environment variable")
        sys.exit(1)
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=lambda: threading.Timer(3600, cleanup_old_files).start())
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    print("🚀 Policy Pulse - RAG Pipeline Application")
    print("=" * 50)
    print("📋 Available endpoints:")
    print("   • POST /api/upload - Upload and process PDF")
    print("   • POST /api/analyze/<document_key> - Analyze content")
    print("   • POST /api/embed/<document_key> - Create embeddings")
    print("   • POST /api/ask - Ask questions about document")
    print("   • POST /api/generate-brief - Generate policy brief")
    print("   • POST /api/use-document-key - Load existing document")
    print("   • GET /api/documents - List all documents")
    print("   • GET /api/document/<key> - Get document info")
    print("   • GET /api/download/<filename> - Download files")
    print("=" * 50)
    print("🌐 Starting server...")
    
    # Run the app
    socketio.run(
        app, 
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        allow_unsafe_werkzeug=True
    )


