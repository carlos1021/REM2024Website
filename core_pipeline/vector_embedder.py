"""
Vector Embedding & Storage Engine - Step 3 of RAG Pipeline

This module handles the creation of semantic embeddings and vector storage for enhanced content chunks.
It creates FAISS indexes locally and uploads them to Firebase Storage for persistent, scalable retrieval.

Key Features:
- Multilingual semantic embeddings using sentence-transformers
- Local FAISS vector database with fast similarity search
- Firebase Storage integration for persistent vector storage
- Hybrid search capabilities (semantic + keyword + metadata)
- Multi-modal embedding support (text, tables, figures)
- Incremental indexing and updates
"""

import os
import re
import json
import logging
import pickle
import tempfile
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import faiss
import firebase_admin
from firebase_admin import credentials, storage, db
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingMetadata:
    """Metadata for embedded content chunks"""
    chunk_id: str
    document_key: str
    chunk_type: str
    context_level: str
    section_title: str
    page_numbers: List[int]
    word_count: int
    position_in_document: float
    semantic_keywords: List[str]
    related_tables: List[str]
    related_figures: List[str]
    embedding_model: str
    embedded_at: str


@dataclass
class VectorSearchResult:
    """Search result with relevance scoring"""
    chunk_id: str
    content: str
    similarity_score: float
    metadata: EmbeddingMetadata
    context_info: Dict[str, Any]
    
    # Multi-modal results
    related_content: Dict[str, List[str]] = None


@dataclass
class VectorDatabase:
    """Container for complete vector database"""
    faiss_index: faiss.Index
    embeddings: np.ndarray
    metadata: List[EmbeddingMetadata]
    chunk_contents: List[str]
    
    # Index metadata
    database_id: str
    created_at: str
    embedding_model: str
    total_chunks: int
    documents_included: List[str]
    
    # Search capabilities
    tfidf_vectorizer: Optional[TfidfVectorizer] = None
    keyword_vectors: Optional[np.ndarray] = None


class VectorEmbedder:
    """
    Vector embedding and storage engine with Firebase integration.
    """
    
    def __init__(self,
                 firebase_bucket: str = "rem2024-f429b.appspot.com",
                 firebase_database_url: str = "https://rem2024-f429b-default-rtdb.firebaseio.com",
                 enable_firebase: bool = True,
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 dimension: int = 384,
                 enable_hybrid_search: bool = True):
        """
        Initialize the vector embedder.
        
        Args:
            firebase_bucket: Firebase storage bucket name
            firebase_database_url: Firebase Realtime Database URL
            enable_firebase: Whether to use Firebase for storage
            embedding_model: Sentence transformer model for embeddings
            dimension: Embedding dimension (should match model)
            enable_hybrid_search: Whether to enable keyword+semantic search
        """
        self.firebase_bucket = firebase_bucket
        self.firebase_database_url = firebase_database_url
        self.enable_firebase = enable_firebase
        self.embedding_model_name = embedding_model
        self.dimension = dimension
        self.enable_hybrid_search = enable_hybrid_search
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Verify embedding dimension
        test_embedding = self.embedding_model.encode(["test text"])
        actual_dimension = test_embedding.shape[1]
        if actual_dimension != dimension:
            logger.warning(f"Embedding dimension mismatch: expected {dimension}, got {actual_dimension}")
            self.dimension = actual_dimension
        
        # Initialize TF-IDF for hybrid search
        if self.enable_hybrid_search:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        # Initialize Firebase if enabled
        if self.enable_firebase:
            self._init_firebase()
        
        logger.info("Vector embedder initialized")
    
    def _init_firebase(self):
        """Initialize Firebase connection using existing app or create new one"""
        try:
            # Check if Firebase is already initialized
            firebase_admin.get_app()
            logger.info("Using existing Firebase app")
        except ValueError:
            # Initialize Firebase using environment variable
            try:
                firebase_key_base64 = os.getenv("FIREBASE_SERVICE_KEY")
                if not firebase_key_base64:
                    logger.warning("FIREBASE_SERVICE_KEY environment variable not found")
                    self.enable_firebase = False
                    return
                
                import base64
                firebase_key_json = base64.b64decode(firebase_key_base64).decode('utf-8')
                firebase_service_account = json.loads(firebase_key_json)
                
                cred = credentials.Certificate(firebase_service_account)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': self.firebase_bucket,
                    'databaseURL': self.firebase_database_url
                })
                logger.info("Firebase initialized successfully")
                
            except Exception as e:
                logger.warning(f"Failed to initialize Firebase: {e}")
                self.enable_firebase = False
                return
        
        # Get Firebase services
        try:
            self.bucket = storage.bucket()
            self.database = db.reference()
            logger.info("Firebase storage and database ready")
        except Exception as e:
            logger.warning(f"Failed to get Firebase services: {e}")
            self.enable_firebase = False
    
    def create_vector_database(self, document_keys: List[str] = None) -> VectorDatabase:
        """
        Create a vector database from enhanced content chunks.
        
        Args:
            document_keys: Specific documents to include (None = all available)
            
        Returns:
            VectorDatabase containing FAISS index and metadata
        """
        start_time = datetime.now()
        
        try:
            # Retrieve enhanced content chunks
            chunks_data = self._retrieve_enhanced_chunks(document_keys)
            
            if not chunks_data:
                raise ValueError("No enhanced chunks found for embedding")
            
            logger.info(f"Creating embeddings for {len(chunks_data)} chunks")
            
            # Extract content and metadata
            chunk_contents = []
            metadata_list = []
            
            for chunk_data in chunks_data:
                content = chunk_data['content']
                chunk_contents.append(content)
                
                # Create metadata object
                metadata = EmbeddingMetadata(
                    chunk_id=chunk_data['chunk_id'],
                    document_key=chunk_data.get('document_key', 'unknown'),
                    chunk_type=chunk_data['chunk_type'],
                    context_level=chunk_data['context_level'],
                    section_title=chunk_data.get('section_title', 'Unknown'),
                    page_numbers=chunk_data.get('page_numbers', []),
                    word_count=chunk_data.get('word_count', 0),
                    position_in_document=chunk_data.get('position_in_document', 0.0),
                    semantic_keywords=chunk_data.get('semantic_keywords', []),
                    related_tables=chunk_data.get('related_tables', []),
                    related_figures=chunk_data.get('related_figures', []),
                    embedding_model=self.embedding_model_name,
                    embedded_at=datetime.now().isoformat()
                )
                metadata_list.append(metadata)
            
            # Generate embeddings
            logger.info("Generating semantic embeddings...")
            embeddings = self.embedding_model.encode(
                chunk_contents,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Create FAISS index
            logger.info(f"Creating FAISS index with {self.dimension} dimensions...")
            faiss_index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            faiss_index.add(embeddings)
            
            # Create TF-IDF vectors for hybrid search
            keyword_vectors = None
            tfidf_vectorizer = None
            
            if self.enable_hybrid_search:
                logger.info("Creating TF-IDF vectors for hybrid search...")
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                keyword_vectors = tfidf_vectorizer.fit_transform(chunk_contents).toarray()
            
            # Generate database ID
            database_id = f"vectordb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get unique document keys
            unique_documents = list(set(metadata.document_key for metadata in metadata_list))
            
            # Create vector database object
            vector_db = VectorDatabase(
                faiss_index=faiss_index,
                embeddings=embeddings,
                metadata=metadata_list,
                chunk_contents=chunk_contents,
                database_id=database_id,
                created_at=datetime.now().isoformat(),
                embedding_model=self.embedding_model_name,
                total_chunks=len(chunk_contents),
                documents_included=unique_documents,
                tfidf_vectorizer=tfidf_vectorizer,
                keyword_vectors=keyword_vectors
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Vector database created successfully in {processing_time:.2f}s")
            logger.info(f"Database ID: {database_id}")
            logger.info(f"Total chunks: {len(chunk_contents)}")
            logger.info(f"Documents included: {len(unique_documents)}")
            
            return vector_db
            
        except Exception as e:
            logger.error(f"Failed to create vector database: {e}")
            raise
    
    def _retrieve_enhanced_chunks(self, document_keys: List[str] = None) -> List[Dict[str, Any]]:
        """Retrieve enhanced chunks from Firebase"""
        if not self.enable_firebase:
            raise ValueError("Firebase not enabled - cannot retrieve chunks")
        
        try:
            all_chunks = []
            
            if document_keys:
                # Retrieve specific documents
                for doc_key in document_keys:
                    analysis_ref = self.database.child('analyzed_documents').child(doc_key)
                    analysis_data = analysis_ref.get()
                    
                    if analysis_data and 'enhanced_chunks' in analysis_data:
                        chunks = analysis_data['enhanced_chunks']
                        # Add document key to each chunk
                        for chunk in chunks:
                            chunk['document_key'] = doc_key
                        all_chunks.extend(chunks)
                        logger.info(f"Retrieved {len(chunks)} chunks from {doc_key}")
            else:
                # Retrieve all analyzed documents
                analyzed_ref = self.database.child('analyzed_documents')
                all_analyses = analyzed_ref.get()
                
                if not all_analyses:
                    logger.warning("No analyzed documents found in Firebase")
                    return []
                
                for doc_key, analysis_data in all_analyses.items():
                    if 'enhanced_chunks' in analysis_data:
                        chunks = analysis_data['enhanced_chunks']
                        # Add document key to each chunk
                        for chunk in chunks:
                            chunk['document_key'] = doc_key
                        all_chunks.extend(chunks)
                
                logger.info(f"Retrieved chunks from {len(all_analyses)} analyzed documents")
            
            logger.info(f"Total chunks retrieved: {len(all_chunks)}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve enhanced chunks: {e}")
            raise 
    
    def semantic_search(self, vector_db: VectorDatabase, query: str, 
                       top_k: int = 10, include_metadata: bool = True) -> List[VectorSearchResult]:
        """
        Perform semantic similarity search on the vector database.
        
        Args:
            vector_db: VectorDatabase to search
            query: Search query text
            top_k: Number of top results to return
            include_metadata: Whether to include full metadata in results
            
        Returns:
            List of VectorSearchResult objects ranked by similarity
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            similarities, indices = vector_db.faiss_index.search(query_embedding, top_k)
            
            # Create search results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                metadata = vector_db.metadata[idx] if include_metadata else None
                
                result = VectorSearchResult(
                    chunk_id=vector_db.metadata[idx].chunk_id,
                    content=vector_db.chunk_contents[idx],
                    similarity_score=float(similarity),
                    metadata=metadata,
                    context_info={
                        'rank': i + 1,
                        'document_key': vector_db.metadata[idx].document_key,
                        'section': vector_db.metadata[idx].section_title,
                        'chunk_type': vector_db.metadata[idx].chunk_type
                    }
                )
                
                # Add related content if available
                if metadata:
                    result.related_content = {
                        'tables': metadata.related_tables,
                        'figures': metadata.related_figures,
                        'keywords': metadata.semantic_keywords
                    }
                
                results.append(result)
            
            logger.info(f"Semantic search returned {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def hybrid_search(self, vector_db: VectorDatabase, query: str, 
                     top_k: int = 10, semantic_weight: float = 0.7) -> List[VectorSearchResult]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            vector_db: VectorDatabase to search
            query: Search query text
            top_k: Number of top results to return
            semantic_weight: Weight for semantic similarity (0-1)
            
        Returns:
            List of VectorSearchResult objects ranked by combined score
        """
        if not self.enable_hybrid_search or vector_db.tfidf_vectorizer is None:
            logger.warning("Hybrid search not enabled, falling back to semantic search")
            return self.semantic_search(vector_db, query, top_k)
        
        try:
            # Semantic search
            semantic_results = self.semantic_search(vector_db, query, top_k * 2)  # Get more candidates
            
            # Keyword search using TF-IDF
            query_tfidf = vector_db.tfidf_vectorizer.transform([query]).toarray()
            keyword_similarities = np.dot(vector_db.keyword_vectors, query_tfidf.T).flatten()
            
            # Combine scores
            combined_results = []
            for result in semantic_results:
                # Find the index of this chunk
                chunk_idx = next(i for i, metadata in enumerate(vector_db.metadata) 
                               if metadata.chunk_id == result.chunk_id)
                
                # Combine semantic and keyword scores
                semantic_score = result.similarity_score
                keyword_score = keyword_similarities[chunk_idx]
                
                # Normalize and combine (simple weighted average)
                combined_score = (semantic_weight * semantic_score + 
                                (1 - semantic_weight) * keyword_score)
                
                result.similarity_score = combined_score
                result.context_info['semantic_score'] = semantic_score
                result.context_info['keyword_score'] = keyword_score
                
                combined_results.append(result)
            
            # Sort by combined score and take top_k
            combined_results.sort(key=lambda x: x.similarity_score, reverse=True)
            final_results = combined_results[:top_k]
            
            # Update ranks
            for i, result in enumerate(final_results):
                result.context_info['rank'] = i + 1
            
            logger.info(f"Hybrid search returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self.semantic_search(vector_db, query, top_k)
    
    def save_to_firebase_storage(self, vector_db: VectorDatabase) -> Dict[str, str]:
        """
        Save the vector database to Firebase Storage.
        
        Args:
            vector_db: VectorDatabase to save
            
        Returns:
            Dictionary with Firebase Storage paths
        """
        if not self.enable_firebase:
            raise ValueError("Firebase not enabled - cannot save to storage")
        
        try:
            logger.info(f"Uploading vector database {vector_db.database_id} to Firebase Storage...")
            
            # Create temporary directory for files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save FAISS index
                faiss_path = temp_path / "faiss_index.bin"
                faiss.write_index(vector_db.faiss_index, str(faiss_path))
                
                # Save embeddings
                embeddings_path = temp_path / "embeddings.npy"
                np.save(embeddings_path, vector_db.embeddings)
                
                # Save metadata and other data
                metadata_path = temp_path / "metadata.pkl"
                other_data = {
                    'metadata': [asdict(meta) for meta in vector_db.metadata],
                    'chunk_contents': vector_db.chunk_contents,
                    'database_id': vector_db.database_id,
                    'created_at': vector_db.created_at,
                    'embedding_model': vector_db.embedding_model,
                    'total_chunks': vector_db.total_chunks,
                    'documents_included': vector_db.documents_included,
                    'dimension': self.dimension,
                    'enable_hybrid_search': self.enable_hybrid_search
                }
                
                # Include TF-IDF data if available
                if vector_db.tfidf_vectorizer is not None:
                    tfidf_path = temp_path / "tfidf_vectorizer.pkl"
                    with open(tfidf_path, 'wb') as f:
                        pickle.dump(vector_db.tfidf_vectorizer, f)
                    
                    keyword_vectors_path = temp_path / "keyword_vectors.npy"
                    np.save(keyword_vectors_path, vector_db.keyword_vectors)
                    
                    other_data['has_tfidf'] = True
                else:
                    other_data['has_tfidf'] = False
                
                with open(metadata_path, 'wb') as f:
                    pickle.dump(other_data, f)
                
                # Upload files to Firebase Storage
                storage_paths = {}
                base_path = f"vector_databases/{vector_db.database_id}/"
                
                # Upload FAISS index
                faiss_blob = self.bucket.blob(base_path + "faiss_index.bin")
                faiss_blob.upload_from_filename(faiss_path)
                storage_paths['faiss_index'] = faiss_blob.name
                
                # Upload embeddings
                embeddings_blob = self.bucket.blob(base_path + "embeddings.npy")
                embeddings_blob.upload_from_filename(embeddings_path)
                storage_paths['embeddings'] = embeddings_blob.name
                
                # Upload metadata
                metadata_blob = self.bucket.blob(base_path + "metadata.pkl")
                metadata_blob.upload_from_filename(metadata_path)
                storage_paths['metadata'] = metadata_blob.name
                
                # Upload TF-IDF files if they exist
                if vector_db.tfidf_vectorizer is not None:
                    tfidf_blob = self.bucket.blob(base_path + "tfidf_vectorizer.pkl")
                    tfidf_blob.upload_from_filename(tfidf_path)
                    storage_paths['tfidf_vectorizer'] = tfidf_blob.name
                    
                    keyword_blob = self.bucket.blob(base_path + "keyword_vectors.npy")
                    keyword_blob.upload_from_filename(keyword_vectors_path)
                    storage_paths['keyword_vectors'] = keyword_blob.name
                
                # Create database index entry in Realtime Database
                db_index_entry = {
                    'database_id': vector_db.database_id,
                    'created_at': vector_db.created_at,
                    'embedding_model': vector_db.embedding_model,
                    'total_chunks': vector_db.total_chunks,
                    'documents_included': vector_db.documents_included,
                    'storage_paths': storage_paths,
                    'dimension': self.dimension,
                    'has_hybrid_search': self.enable_hybrid_search
                }
                
                db_ref = self.database.child('vector_databases').child(vector_db.database_id)
                db_ref.set(db_index_entry)
                
                # Also create a timestamp index
                timestamp_ref = self.database.child('vector_db_index').child(vector_db.created_at.replace(':', '_').replace('.', '_'))
                timestamp_ref.set({
                    'database_id': vector_db.database_id,
                    'total_chunks': vector_db.total_chunks,
                    'documents_count': len(vector_db.documents_included)
                })
                
                logger.info(f"Vector database uploaded successfully to Firebase Storage")
                return {
                    'database_id': vector_db.database_id,
                    'storage_base_path': base_path,
                    'firebase_console': f"https://console.firebase.google.com/project/rem2024-f429b/storage/rem2024-f429b.appspot.com/files/~2Fvector_databases~2F{vector_db.database_id}",
                    **storage_paths
                }
        
        except Exception as e:
            logger.error(f"Failed to save vector database to Firebase Storage: {e}")
            raise
    
    def load_from_firebase_storage(self, database_id: str) -> VectorDatabase:
        """
        Load a vector database from Firebase Storage.
        
        Args:
            database_id: Database identifier to load
            
        Returns:
            VectorDatabase object
        """
        if not self.enable_firebase:
            raise ValueError("Firebase not enabled - cannot load from storage")
        
        try:
            logger.info(f"Loading vector database {database_id} from Firebase Storage...")
            
            # Get database info from Realtime Database
            db_ref = self.database.child('vector_databases').child(database_id)
            db_info = db_ref.get()
            
            if not db_info:
                raise ValueError(f"Vector database {database_id} not found in Firebase")
            
            storage_paths = db_info['storage_paths']
            
            # Create temporary directory for downloaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download FAISS index
                faiss_path = temp_path / "faiss_index.bin"
                faiss_blob = self.bucket.blob(storage_paths['faiss_index'])
                faiss_blob.download_to_filename(faiss_path)
                faiss_index = faiss.read_index(str(faiss_path))
                
                # Download embeddings
                embeddings_path = temp_path / "embeddings.npy"
                embeddings_blob = self.bucket.blob(storage_paths['embeddings'])
                embeddings_blob.download_to_filename(embeddings_path)
                embeddings = np.load(embeddings_path)
                
                # Download metadata
                metadata_path = temp_path / "metadata.pkl"
                metadata_blob = self.bucket.blob(storage_paths['metadata'])
                metadata_blob.download_to_filename(metadata_path)
                
                with open(metadata_path, 'rb') as f:
                    other_data = pickle.load(f)
                
                # Reconstruct metadata objects
                metadata_list = []
                for meta_dict in other_data['metadata']:
                    metadata = EmbeddingMetadata(**meta_dict)
                    metadata_list.append(metadata)
                
                # Download TF-IDF data if available
                tfidf_vectorizer = None
                keyword_vectors = None
                
                if other_data.get('has_tfidf', False):
                    tfidf_path = temp_path / "tfidf_vectorizer.pkl"
                    tfidf_blob = self.bucket.blob(storage_paths['tfidf_vectorizer'])
                    tfidf_blob.download_to_filename(tfidf_path)
                    
                    with open(tfidf_path, 'rb') as f:
                        tfidf_vectorizer = pickle.load(f)
                    
                    keyword_vectors_path = temp_path / "keyword_vectors.npy"
                    keyword_blob = self.bucket.blob(storage_paths['keyword_vectors'])
                    keyword_blob.download_to_filename(keyword_vectors_path)
                    keyword_vectors = np.load(keyword_vectors_path)
                
                # Create VectorDatabase object
                vector_db = VectorDatabase(
                    faiss_index=faiss_index,
                    embeddings=embeddings,
                    metadata=metadata_list,
                    chunk_contents=other_data['chunk_contents'],
                    database_id=other_data['database_id'],
                    created_at=other_data['created_at'],
                    embedding_model=other_data['embedding_model'],
                    total_chunks=other_data['total_chunks'],
                    documents_included=other_data['documents_included'],
                    tfidf_vectorizer=tfidf_vectorizer,
                    keyword_vectors=keyword_vectors
                )
                
                logger.info(f"Vector database {database_id} loaded successfully")
                return vector_db
        
        except Exception as e:
            logger.error(f"Failed to load vector database {database_id}: {e}")
            raise
    
    def list_vector_databases(self) -> List[Dict[str, Any]]:
        """
        List all available vector databases in Firebase.
        
        Returns:
            List of database information dictionaries
        """
        if not self.enable_firebase:
            raise ValueError("Firebase not enabled - cannot list databases")
        
        try:
            db_ref = self.database.child('vector_databases')
            databases = db_ref.get()
            
            if not databases:
                logger.info("No vector databases found in Firebase")
                return []
            
            db_list = []
            for db_id, db_info in databases.items():
                db_list.append({
                    'database_id': db_id,
                    'created_at': db_info.get('created_at'),
                    'embedding_model': db_info.get('embedding_model'),
                    'total_chunks': db_info.get('total_chunks', 0),
                    'documents_count': len(db_info.get('documents_included', [])),
                    'dimension': db_info.get('dimension'),
                    'has_hybrid_search': db_info.get('has_hybrid_search', False)
                })
            
            # Sort by creation time (most recent first)
            db_list.sort(key=lambda x: x['created_at'], reverse=True)
            
            logger.info(f"Found {len(db_list)} vector databases")
            return db_list
            
        except Exception as e:
            logger.error(f"Failed to list vector databases: {e}")
            return []


# Utility functions for easy usage
def create_vector_database_from_firebase(document_keys: List[str] = None,
                                       embedding_model: str = "BAAI/bge-small-en-v1.5",
                                       enable_hybrid_search: bool = True) -> VectorDatabase:
    """
    Convenience function to create a vector database from Firebase data.
    
    Args:
        document_keys: Specific documents to include (None = all)
        embedding_model: Sentence transformer model name
        enable_hybrid_search: Whether to enable hybrid search
        
    Returns:
        VectorDatabase object
    """
    embedder = VectorEmbedder(
        embedding_model=embedding_model,
        enable_hybrid_search=enable_hybrid_search
    )
    
    return embedder.create_vector_database(document_keys)


def upload_vector_database(vector_db: VectorDatabase) -> Dict[str, str]:
    """
    Convenience function to upload a vector database to Firebase Storage.
    
    Args:
        vector_db: VectorDatabase to upload
        
    Returns:
        Dictionary with Firebase Storage paths
    """
    embedder = VectorEmbedder()
    return embedder.save_to_firebase_storage(vector_db)


def load_vector_database(database_id: str) -> VectorDatabase:
    """
    Convenience function to load a vector database from Firebase Storage.
    
    Args:
        database_id: Database identifier
        
    Returns:
        VectorDatabase object
    """
    embedder = VectorEmbedder()
    return embedder.load_from_firebase_storage(database_id)


def search_documents(database_id: str, query: str, top_k: int = 5, 
                    use_hybrid: bool = True) -> List[VectorSearchResult]:
    """
    Convenience function to search documents in a vector database.
    
    Args:
        database_id: Database identifier
        query: Search query
        top_k: Number of results to return
        use_hybrid: Whether to use hybrid search
        
    Returns:
        List of search results
    """
    embedder = VectorEmbedder()
    vector_db = embedder.load_from_firebase_storage(database_id)
    
    if use_hybrid and vector_db.tfidf_vectorizer is not None:
        return embedder.hybrid_search(vector_db, query, top_k)
    else:
        return embedder.semantic_search(vector_db, query, top_k) 