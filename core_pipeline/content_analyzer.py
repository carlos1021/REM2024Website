"""
Content Analysis & Chunking Engine - Step 2 of RAG Pipeline

This module performs advanced content analysis and semantic chunking on processed documents.
It retrieves data from Firebase, performs intelligent chunking with academic paper awareness,
and creates multi-modal content relationships.

Key Features:
- Academic paper structure recognition
- Multi-modal content integration (text + tables + figures)
- Context-preserving chunking strategies
- Hierarchical relationship mapping
- Firebase integration for data retrieval and storage
"""

import os
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import firebase_admin
from firebase_admin import credentials, storage, db
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
import openai

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of content chunks"""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCE = "reference"
    TABLE = "table"
    FIGURE = "figure"
    GENERAL_TEXT = "general_text"


class ContextLevel(Enum):
    """Levels of context preservation"""
    DOCUMENT = "document"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"


@dataclass
class ContentChunk:
    """Enhanced content chunk with academic structure awareness"""
    chunk_id: str
    content: str
    chunk_type: ChunkType
    context_level: ContextLevel
    
    # Academic structure
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    hierarchy_path: List[str] = None
    
    # Multi-modal relationships
    related_tables: List[str] = None
    related_figures: List[str] = None
    related_captions: List[str] = None
    
    # Context preservation
    preceding_context: Optional[str] = None
    following_context: Optional[str] = None
    semantic_keywords: List[str] = None
    
    # Metadata
    page_numbers: List[int] = None
    word_count: int = 0
    char_count: int = 0
    position_in_document: float = 0.0  # 0.0 to 1.0
    
    # Processing metadata
    created_at: str = None
    processing_method: str = None


@dataclass
class AnalysisResult:
    """Container for complete content analysis results"""
    document_key: str
    enhanced_chunks: List[ContentChunk]
    document_structure: Dict[str, Any]
    multi_modal_relationships: Dict[str, List[str]]
    processing_stats: Dict[str, Any]
    errors: List[str]


class ContentAnalyzer:
    """
    Advanced content analyzer for academic papers with Firebase integration.
    """
    
    def __init__(self,
                 firebase_bucket: str = "rem2024-f429b.appspot.com",
                 firebase_database_url: str = "https://rem2024-f429b-default-rtdb.firebaseio.com",
                 enable_firebase: bool = True,
                 chunk_size: int = 800,
                 chunk_overlap: int = 150,
                 enable_openai: bool = True):
        """
        Initialize the content analyzer.
        
        Args:
            firebase_bucket: Firebase storage bucket name
            firebase_database_url: Firebase Realtime Database URL
            enable_firebase: Whether to use Firebase for data storage
            chunk_size: Base chunk size for semantic chunking
            chunk_overlap: Overlap between chunks
            enable_openai: Whether to use OpenAI for enhanced analysis
        """
        self.firebase_bucket = firebase_bucket
        self.firebase_database_url = firebase_database_url
        self.enable_firebase = enable_firebase
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_openai = enable_openai
        
        # Academic paper section patterns
        self.section_patterns = {
            ChunkType.ABSTRACT: re.compile(r'\b(abstract|summary)\b', re.IGNORECASE),
            ChunkType.INTRODUCTION: re.compile(r'\b(introduction|background)\b', re.IGNORECASE),
            ChunkType.METHODOLOGY: re.compile(r'\b(method|methodology|approach|procedure|experiment)\b', re.IGNORECASE),
            ChunkType.RESULTS: re.compile(r'\b(results?|findings?|outcomes?)\b', re.IGNORECASE),
            ChunkType.DISCUSSION: re.compile(r'\b(discussion|analysis|interpretation)\b', re.IGNORECASE),
            ChunkType.CONCLUSION: re.compile(r'\b(conclusion|summary|implications?)\b', re.IGNORECASE),
            ChunkType.REFERENCE: re.compile(r'\b(references?|bibliography|citations?)\b', re.IGNORECASE),
        }
        
        # Initialize text splitters for different strategies
        self.semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.token_splitter = TokenTextSplitter(
            chunk_size=chunk_size // 4,  # Assuming ~4 chars per token
            chunk_overlap=chunk_overlap // 4
        )
        
        # Initialize OpenAI if enabled
        if self.enable_openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize Firebase if enabled
        if self.enable_firebase:
            self._init_firebase()
        
        logger.info("Content analyzer initialized")
    
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
        
        # Get database reference
        try:
            self.database = db.reference()
            logger.info("Firebase database ready")
        except Exception as e:
            logger.warning(f"Failed to get Firebase database: {e}")
            self.enable_firebase = False
    
    def analyze_document(self, document_key: str) -> AnalysisResult:
        """
        Perform comprehensive content analysis on a processed document.
        
        Args:
            document_key: Unique document identifier from step 1 processing
            
        Returns:
            AnalysisResult containing enhanced chunks and relationships
        """
        start_time = datetime.now()
        errors = []
        
        try:
            # Retrieve document from Firebase
            doc_data = self._retrieve_document(document_key)
            if not doc_data:
                raise ValueError(f"Document {document_key} not found in Firebase")
            
            logger.info(f"Analyzing document: {document_key}")
            
            # Extract content components
            text_content = doc_data.get('content', {}).get('text_content', [])
            texts = doc_data.get('content', {}).get('texts', [])
            tables = doc_data.get('content', {}).get('tables', [])
            semantic_chunks = doc_data.get('content', {}).get('semantic_chunks', [])
            captions = doc_data.get('content', {}).get('figure_table_captions', [])
            images = doc_data.get('images', {}).get('references', [])
            
            # Analyze document structure
            document_structure = self._analyze_document_structure(text_content, texts)
            
            # Create enhanced chunks with academic awareness
            enhanced_chunks = self._create_enhanced_chunks(
                texts, tables, captions, images, document_structure
            )
            
            # Build multi-modal relationships
            relationships = self._build_multimodal_relationships(
                enhanced_chunks, tables, images, captions
            )
            
            # Calculate processing statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            stats = {
                'processing_time_seconds': processing_time,
                'original_chunks': len(semantic_chunks),
                'enhanced_chunks': len(enhanced_chunks),
                'document_sections': len(document_structure.get('sections', [])),
                'multimodal_relationships': len(relationships),
                'total_word_count': sum(chunk.word_count for chunk in enhanced_chunks),
                'errors_count': len(errors)
            }
            
            result = AnalysisResult(
                document_key=document_key,
                enhanced_chunks=enhanced_chunks,
                document_structure=document_structure,
                multi_modal_relationships=relationships,
                processing_stats=stats,
                errors=errors
            )
            
            logger.info(f"Content analysis completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Failed to analyze document {document_key}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return AnalysisResult(
                document_key=document_key,
                enhanced_chunks=[],
                document_structure={},
                multi_modal_relationships={},
                processing_stats={'processing_time_seconds': 0, 'errors_count': len(errors)},
                errors=errors
            )
    
    def _retrieve_document(self, document_key: str) -> Dict[str, Any]:
        """Retrieve processed document data from Firebase"""
        if not self.enable_firebase:
            raise ValueError("Firebase not enabled - cannot retrieve document")
        
        try:
            doc_ref = self.database.child('documents').child(document_key)
            doc_data = doc_ref.get()
            
            if not doc_data:
                logger.warning(f"Document {document_key} not found in Firebase")
                return None
            
            logger.info(f"Retrieved document {document_key} from Firebase")
            return doc_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve document {document_key}: {e}")
            raise
    
    def _analyze_document_structure(self, text_content: List[str], texts: List[str]) -> Dict[str, Any]:
        """
        Analyze the academic structure of the document.
        
        Returns document hierarchy, sections, and structural metadata.
        """
        try:
            # Combine all text for structure analysis
            full_text = "\n\n".join(text_content)
            
            # Detect academic sections
            sections = []
            current_section = None
            
            for i, text_block in enumerate(texts):
                # Check for section headers
                section_type = self._identify_section_type(text_block)
                
                if section_type != ChunkType.GENERAL_TEXT:
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = {
                        'type': section_type.value,
                        'title': self._extract_section_title(text_block),
                        'start_index': i,
                        'content_blocks': [text_block]
                    }
                elif current_section:
                    current_section['content_blocks'].append(text_block)
                else:
                    # Create a general section for unclassified content
                    if not sections or sections[-1]['type'] != 'general_text':
                        current_section = {
                            'type': 'general_text',
                            'title': 'General Content',
                            'start_index': i,
                            'content_blocks': [text_block]
                        }
            
            # Add the last section
            if current_section:
                sections.append(current_section)
            
            # Calculate document statistics
            total_words = len(full_text.split())
            total_chars = len(full_text)
            
            structure = {
                'sections': sections,
                'total_sections': len(sections),
                'document_length': {
                    'characters': total_chars,
                    'words': total_words,
                    'text_blocks': len(texts)
                },
                'section_distribution': {
                    section['type']: len(section['content_blocks']) 
                    for section in sections
                }
            }
            
            logger.info(f"Identified {len(sections)} document sections")
            return structure
            
        except Exception as e:
            logger.warning(f"Structure analysis failed: {e}")
            return {'sections': [], 'total_sections': 0}
    
    def _identify_section_type(self, text_block: str) -> ChunkType:
        """Identify the academic section type of a text block"""
        text_lower = text_block.lower()
        
        # Check each section pattern
        for section_type, pattern in self.section_patterns.items():
            if pattern.search(text_lower):
                return section_type
        
        return ChunkType.GENERAL_TEXT
    
    def _extract_section_title(self, text_block: str) -> str:
        """Extract section title from text block"""
        lines = text_block.split('\n')
        
        # Look for title-like patterns in first few lines
        for line in lines[:3]:
            line = line.strip()
            if len(line) > 0 and len(line) < 100:  # Reasonable title length
                # Check if it looks like a title (short, potentially all caps, etc.)
                if (line.isupper() or 
                    any(pattern.search(line.lower()) for pattern in self.section_patterns.values())):
                    return line
        
        # Fallback: use first meaningful line
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Skip very short lines
                return line[:80] + "..." if len(line) > 80 else line
        
        return "Untitled Section"
    
    def _create_enhanced_chunks(self, texts: List[str], tables: List[str], 
                              captions: List[str], images: List[Dict],
                              document_structure: Dict[str, Any]) -> List[ContentChunk]:
        """
        Create enhanced content chunks with academic structure awareness and multi-modal integration.
        """
        enhanced_chunks = []
        chunk_counter = 0
        
        # Process text content by sections
        sections = document_structure.get('sections', [])
        
        for section_idx, section in enumerate(sections):
            section_type = ChunkType(section['type']) if section['type'] in [t.value for t in ChunkType] else ChunkType.GENERAL_TEXT
            section_title = section['title']
            content_blocks = section['content_blocks']
            
            # Combine section content
            section_content = "\n\n".join(content_blocks)
            
            # Apply different chunking strategies based on section type
            if section_type in [ChunkType.TABLE, ChunkType.FIGURE]:
                # Handle tables and figures as single chunks
                chunks = [section_content]
            elif section_type == ChunkType.REFERENCE:
                # Split references differently (by citation)
                chunks = self._split_references(section_content)
            else:
                # Use semantic chunking for regular text
                chunks = self.semantic_splitter.split_text(section_content)
            
            # Create enhanced chunks for this section
            for chunk_idx, chunk_content in enumerate(chunks):
                if len(chunk_content.strip()) < 50:  # Skip very short chunks
                    continue
                
                chunk_id = f"chunk_{chunk_counter:04d}_{section_type.value}_{chunk_idx}"
                
                # Calculate position in document
                position = (section_idx + (chunk_idx / len(chunks))) / len(sections)
                
                # Extract semantic keywords if OpenAI is available
                keywords = self._extract_keywords(chunk_content) if self.enable_openai else []
                
                # Determine context level
                context_level = self._determine_context_level(chunk_content, section_type)
                
                enhanced_chunk = ContentChunk(
                    chunk_id=chunk_id,
                    content=chunk_content,
                    chunk_type=section_type,
                    context_level=context_level,
                    section_title=section_title,
                    hierarchy_path=[section_title],
                    related_tables=[],
                    related_figures=[],
                    related_captions=[],
                    preceding_context=chunks[chunk_idx-1] if chunk_idx > 0 else None,
                    following_context=chunks[chunk_idx+1] if chunk_idx < len(chunks)-1 else None,
                    semantic_keywords=keywords,
                    word_count=len(chunk_content.split()),
                    char_count=len(chunk_content),
                    position_in_document=position,
                    created_at=datetime.now().isoformat(),
                    processing_method="academic_structure_aware"
                )
                
                enhanced_chunks.append(enhanced_chunk)
                chunk_counter += 1
        
        # Process tables as separate chunks
        for table_idx, table_content in enumerate(tables):
            chunk_id = f"chunk_{chunk_counter:04d}_table_{table_idx}"
            
            table_chunk = ContentChunk(
                chunk_id=chunk_id,
                content=table_content,
                chunk_type=ChunkType.TABLE,
                context_level=ContextLevel.DOCUMENT,
                section_title="Tables",
                hierarchy_path=["Tables"],
                word_count=len(table_content.split()),
                char_count=len(table_content),
                position_in_document=0.5,  # Tables can appear anywhere
                created_at=datetime.now().isoformat(),
                processing_method="table_extraction"
            )
            
            enhanced_chunks.append(table_chunk)
            chunk_counter += 1
        
        logger.info(f"Created {len(enhanced_chunks)} enhanced chunks")
        return enhanced_chunks
    
    def _split_references(self, reference_content: str) -> List[str]:
        """Split reference section by individual citations"""
        # Split by numbered references (1. 2. etc.) or line breaks
        reference_pattern = re.compile(r'\n\s*\d+\.\s+', re.MULTILINE)
        references = reference_pattern.split(reference_content)
        
        # Clean up and filter
        clean_refs = []
        for ref in references:
            ref = ref.strip()
            if len(ref) > 20:  # Minimum reference length
                clean_refs.append(ref)
        
        return clean_refs if clean_refs else [reference_content]
    
    def _determine_context_level(self, content: str, section_type: ChunkType) -> ContextLevel:
        """Determine the appropriate context level for a chunk"""
        if section_type in [ChunkType.ABSTRACT, ChunkType.CONCLUSION]:
            return ContextLevel.DOCUMENT
        elif section_type in [ChunkType.TABLE, ChunkType.FIGURE]:
            return ContextLevel.SECTION
        elif len(content.split()) > 200:
            return ContextLevel.SECTION
        else:
            return ContextLevel.PARAGRAPH
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract semantic keywords using OpenAI (if available)"""
        if not self.enable_openai or not openai.api_key:
            return []
        
        try:
            # Truncate content if too long
            if len(content) > 1000:
                content = content[:1000] + "..."
            
            response = openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "Extract 5-7 key semantic keywords from this academic text. Return only a comma-separated list."},
                    {"role": "user", "content": content}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(',')]
            
            return keywords[:7]  # Limit to 7 keywords
            
        except Exception as e:
            logger.debug(f"Keyword extraction failed: {e}")
            return []
    
    def _build_multimodal_relationships(self, chunks: List[ContentChunk], 
                                      tables: List[str], images: List[Dict],
                                      captions: List[str]) -> Dict[str, List[str]]:
        """
        Build relationships between text chunks, tables, figures, and captions.
        """
        relationships = {}
        
        # Create mappings for quick lookup
        table_chunks = [chunk for chunk in chunks if chunk.chunk_type == ChunkType.TABLE]
        text_chunks = [chunk for chunk in chunks if chunk.chunk_type != ChunkType.TABLE]
        
        # Find references to tables and figures in text
        for text_chunk in text_chunks:
            content_lower = text_chunk.content.lower()
            
            # Find table references (Table 1, Table 2, etc.)
            table_refs = re.findall(r'table\s+(\d+)', content_lower)
            figure_refs = re.findall(r'fig(?:ure)?\s*\.?\s*(\d+)', content_lower)
            
            # Link to related tables
            for table_ref in table_refs:
                related_tables = [t.chunk_id for t in table_chunks if table_ref in t.content.lower()]
                text_chunk.related_tables.extend(related_tables)
            
            # Link to related figures (image references)
            for figure_ref in figure_refs:
                related_figures = []
                for img in images:
                    if figure_ref in img.get('original_filename', ''):
                        related_figures.append(img.get('secret_key', ''))
                text_chunk.related_figures.extend(related_figures)
            
            # Link to related captions
            related_captions = []
            for caption in captions:
                caption_lower = caption.lower()
                # Check if caption mentions similar concepts
                if any(keyword.lower() in caption_lower for keyword in text_chunk.semantic_keywords):
                    related_captions.append(caption[:50] + "...")
            text_chunk.related_captions.extend(related_captions)
            
            # Add to relationships dictionary
            if text_chunk.related_tables or text_chunk.related_figures or text_chunk.related_captions:
                relationships[text_chunk.chunk_id] = {
                    'tables': text_chunk.related_tables,
                    'figures': text_chunk.related_figures,
                    'captions': text_chunk.related_captions
                }
        
        logger.info(f"Built {len(relationships)} multi-modal relationships")
        return relationships
    
    def export_analysis(self, result: AnalysisResult) -> Dict[str, str]:
        """
        Export enhanced analysis results to Firebase.
        
        Args:
            result: AnalysisResult containing enhanced chunks and analysis
            
        Returns:
            Dictionary with Firebase paths for the exported data
        """
        if not self.enable_firebase:
            raise ValueError("Firebase not enabled - cannot export analysis")
        
        try:
            doc_key = result.document_key
            timestamp = datetime.now().isoformat().replace(':', '_').replace('.', '_')
            
            # Prepare enhanced chunks for Firebase storage
            chunks_data = []
            for chunk in result.enhanced_chunks:
                chunk_dict = asdict(chunk)
                # Convert enum values to strings
                chunk_dict['chunk_type'] = chunk.chunk_type.value
                chunk_dict['context_level'] = chunk.context_level.value
                chunks_data.append(chunk_dict)
            
            # Prepare complete analysis data
            analysis_data = {
                'document_key': doc_key,
                'analysis_metadata': {
                    'analyzed_at': timestamp,
                    'analyzer_version': '1.0',
                    'chunk_count': len(result.enhanced_chunks),
                    'processing_stats': result.processing_stats
                },
                'enhanced_chunks': chunks_data,
                'document_structure': result.document_structure,
                'multimodal_relationships': result.multi_modal_relationships,
                'errors': result.errors
            }
            
            # Upload to Firebase under analyzed_documents/{document_key}
            analysis_ref = self.database.child('analyzed_documents').child(doc_key)
            analysis_ref.set(analysis_data)
            
            # Create an index entry for easier discovery
            index_entry = {
                'document_key': doc_key,
                'analyzed_at': timestamp,
                'chunk_count': len(result.enhanced_chunks),
                'section_count': result.document_structure.get('total_sections', 0),
                'word_count': result.processing_stats.get('total_word_count', 0),
                'has_multimodal': len(result.multi_modal_relationships) > 0
            }
            
            index_ref = self.database.child('analysis_index').child(timestamp)
            index_ref.set(index_entry)
            
            firebase_paths = {
                'analysis_document': f"analyzed_documents/{doc_key}",
                'analysis_index': f"analysis_index/{timestamp}",
                'firebase_console': f"https://console.firebase.google.com/project/rem2024-f429b/database/rem2024-f429b-default-rtdb/data/analyzed_documents/{doc_key}"
            }
            
            logger.info(f"Enhanced analysis for {doc_key} exported to Firebase successfully")
            return firebase_paths
            
        except Exception as e:
            logger.error(f"Failed to export analysis for {doc_key}: {e}")
            raise
    
    def list_available_documents(self) -> List[Dict[str, Any]]:
        """
        List all processed documents available for analysis.
        
        Returns:
            List of document summaries from Firebase
        """
        if not self.enable_firebase:
            raise ValueError("Firebase not enabled - cannot list documents")
        
        try:
            # Get document index
            index_ref = self.database.child('document_index')
            documents = index_ref.get()
            
            if not documents:
                logger.info("No processed documents found in Firebase")
                return []
            
            # Convert to list format
            doc_list = []
            for timestamp, doc_info in documents.items():
                doc_list.append({
                    'document_key': doc_info.get('document_key'),
                    'title': doc_info.get('title', 'Untitled'),
                    'processed_at': timestamp,
                    'page_count': doc_info.get('page_count', 0),
                    'image_count': doc_info.get('image_count', 0),
                    'processing_time': doc_info.get('processing_time', 0)
                })
            
            # Sort by processing time (most recent first)
            doc_list.sort(key=lambda x: x['processed_at'], reverse=True)
            
            logger.info(f"Found {len(doc_list)} processed documents")
            return doc_list
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def get_analysis_summary(self, document_key: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis summary for a specific document.
        
        Args:
            document_key: Document identifier
            
        Returns:
            Analysis summary or None if not found
        """
        if not self.enable_firebase:
            raise ValueError("Firebase not enabled - cannot retrieve analysis")
        
        try:
            analysis_ref = self.database.child('analyzed_documents').child(document_key)
            analysis_data = analysis_ref.get()
            
            if not analysis_data:
                logger.info(f"No analysis found for document {document_key}")
                return None
            
            # Return summary information
            summary = {
                'document_key': document_key,
                'analyzed_at': analysis_data.get('analysis_metadata', {}).get('analyzed_at'),
                'chunk_count': analysis_data.get('analysis_metadata', {}).get('chunk_count', 0),
                'section_count': analysis_data.get('document_structure', {}).get('total_sections', 0),
                'multimodal_relationships': len(analysis_data.get('multimodal_relationships', {})),
                'processing_stats': analysis_data.get('analysis_metadata', {}).get('processing_stats', {}),
                'errors': analysis_data.get('errors', [])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get analysis summary for {document_key}: {e}")
            return None


# Utility functions for easy usage
def analyze_document_from_firebase(document_key: str, 
                                 enable_openai: bool = True,
                                 chunk_size: int = 800) -> AnalysisResult:
    """
    Convenience function to analyze a document from Firebase.
    
    Args:
        document_key: Document identifier from step 1 processing
        enable_openai: Whether to use OpenAI for enhanced analysis
        chunk_size: Base chunk size for semantic chunking
        
    Returns:
        AnalysisResult containing enhanced chunks and analysis
    """
    analyzer = ContentAnalyzer(
        enable_openai=enable_openai,
        chunk_size=chunk_size
    )
    
    return analyzer.analyze_document(document_key)


def list_processed_documents() -> List[Dict[str, Any]]:
    """
    Convenience function to list all processed documents.
    
    Returns:
        List of document summaries
    """
    analyzer = ContentAnalyzer()
    return analyzer.list_available_documents()


def export_enhanced_analysis(result: AnalysisResult) -> Dict[str, str]:
    """
    Convenience function to export analysis results to Firebase.
    
    Args:
        result: AnalysisResult to export
        
    Returns:
        Dictionary with Firebase paths
    """
    analyzer = ContentAnalyzer()
    return analyzer.export_analysis(result) 