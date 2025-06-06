"""
Document Processing Engine

This module handles the extraction of text, tables, images, and metadata from research papers.
Designed to be modular, testable, and enterprise-ready.
"""

import os
import re
import io
import json
import logging
import secrets
import hashlib
import base64
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF
from PIL import Image
import firebase_admin
from firebase_admin import credentials, storage, db
from langdetect import detect, LangDetectException
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

import nltk
nltk.download('averaged_perceptron_tagger_eng')


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """Container for image information with secret key"""
    public_url: str
    secret_key: str
    original_filename: str
    page_number: int
    image_index: int


@dataclass
class DocumentMetadata:
    """Container for document metadata"""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    page_count: int = 0
    file_size: int = 0
    language: Optional[str] = None


@dataclass
class ProcessingResult:
    """Container for complete document processing results"""
    text_content: List[str]  # page-by-page
    image_urls: List[str]    # Firebase URLs for backward compatibility
    image_info: List[ImageInfo]  # Enhanced image info with secret keys
    figure_table_captions: List[str]  # Multi-line captions as existing
    tables: List[str]        # Separated table content
    texts: List[str]         # Non-table text content
    
    # Enhanced structured data
    semantic_chunks: List[str]       # Semantically meaningful chunks
    document_key: str               # Unique document identifier for Firebase
    
    metadata: DocumentMetadata
    processing_stats: Dict[str, Any]
    errors: List[str]


class DocumentProcessor:
    """
    Document processing engine with enhanced table detection and semantic chunking.
    """
    
    def __init__(self, 
                 firebase_bucket: str = "rem2024-f429b.appspot.com",
                 firebase_database_url: str = "https://rem2024-f429b-default-rtdb.firebaseio.com",
                 enable_firebase: bool = True,
                 image_format: str = "PNG",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            firebase_bucket: Firebase storage bucket name
            firebase_database_url: Firebase Realtime Database URL
            enable_firebase: Whether to upload to Firebase
            image_format: Format for image conversion (PNG/JPEG)
            chunk_size: Size for semantic chunking
            chunk_overlap: Overlap between chunks
        """
        self.firebase_bucket = firebase_bucket
        self.firebase_database_url = firebase_database_url
        self.enable_firebase = enable_firebase
        self.image_format = image_format
        
        self.title_pattern = re.compile(r'^(Fig(?:ure)?\.?|Table)\s*\d+(\.|:)?', re.IGNORECASE)
        self.ip_address_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        
        # Enhanced patterns for structure detection
        self.header_patterns = [
            re.compile(r'^[A-Z\s]{5,}$'),  # ALL CAPS headers
            re.compile(r'^\d+\.?\s+[A-Z][^.!?]*$'),  # Numbered sections
            re.compile(r'^[IVX]+\.?\s+[A-Z][^.!?]*$'),  # Roman numeral sections
        ]
        
        # Initialize semantic chunker
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize Firebase if enabled
        if self.enable_firebase:
            self._init_firebase()
        
        logger.info("Document processor initialized")
    
    def _init_firebase(self):
        """Initialize Firebase storage and database using environment variable like app.py"""
        try:
            # Check if Firebase is already initialized
            firebase_admin.get_app()
            logger.info("Using existing Firebase app")
        except ValueError:
            # Firebase not initialized, initialize it using environment variable
            try:
                # Get Firebase service key from environment (following app.py pattern)
                firebase_key_base64 = os.getenv("FIREBASE_SERVICE_KEY")
                if not firebase_key_base64:
                    logger.warning("FIREBASE_SERVICE_KEY environment variable not found")
                    self.enable_firebase = False
                    return
                
                # Decode the base64 service key
                firebase_key_json = base64.b64decode(firebase_key_base64).decode('utf-8')
                firebase_service_account = json.loads(firebase_key_json)
                
                # Initialize Firebase with decoded credentials
                cred = credentials.Certificate(firebase_service_account)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': self.firebase_bucket,
                    'databaseURL': self.firebase_database_url
                })
                logger.info("Firebase initialized successfully with environment credentials")
                
            except Exception as e:
                logger.warning(f"Failed to initialize Firebase: {e}")
                self.enable_firebase = False
                return
        
        # Get storage bucket and database reference
        try:
            self.bucket = storage.bucket()
            self.database = db.reference()
            logger.info("Firebase storage and database ready")
        except Exception as e:
            logger.warning(f"Failed to get Firebase services: {e}")
            self.enable_firebase = False
    
    def _generate_document_key(self, pdf_file_path: str) -> str:
        """Generate a unique secret key for the entire document"""
        # Use file properties and timestamp for uniqueness
        file_stats = os.stat(pdf_file_path)
        content_hash = hashlib.md5(f"{file_stats.st_size}_{file_stats.st_mtime}_{pdf_file_path}".encode()).hexdigest()[:12]
        random_part = secrets.token_urlsafe(12)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"doc_{timestamp}_{random_part}_{content_hash}"
    
    def _generate_secret_key(self, page_num: int, img_index: int, content_hash: str) -> str:
        """Generate a unique secret key for an image"""
        # Combine random string with content-based hash for uniqueness
        random_part = secrets.token_urlsafe(16)
        content_part = hashlib.md5(f"{page_num}_{img_index}_{content_hash}".encode()).hexdigest()[:8]
        return f"img_{random_part}_{content_part}"
    
    def process_document(self, pdf_file_path: str) -> ProcessingResult:
        """
        Process a PDF document using enhanced logic with semantic chunking.
        
        Args:
            pdf_file_path: Path to the PDF file
            
        Returns:
            ProcessingResult containing all extracted content
        """
        start_time = datetime.now()
        errors = []
        
        try:
            # Validate file
            if not self._validate_file(pdf_file_path):
                raise ValueError(f"Invalid file: {pdf_file_path}")
            
            # Extract content using existing proven logic (maintaining compatibility)
            text_content, image_urls, image_info, figure_table_captions = self._extract_images_and_text_from_pdf(pdf_file_path)
            
            # Separate tables and texts using existing logic
            texts, tables = self._process_text(text_content)
            
            # Enhanced: Improve table detection using unstructured (do this once)
            try:
                enhanced_tables = self._extract_tables_with_unstructured(pdf_file_path)
                if enhanced_tables:
                    # Merge with existing table detection
                    tables.extend(enhanced_tables)
                    # Remove duplicates while preserving order
                    tables = list(dict.fromkeys(tables))
            except Exception as e:
                logger.warning(f"Enhanced table extraction failed: {e}")
                errors.append(f"Enhanced table extraction error: {str(e)}")
            
            # Enhanced: Create semantic chunks
            semantic_chunks = []
            try:
                # Combine all text for chunking
                combined_text = "\n\n".join(texts)
                if combined_text.strip():
                    semantic_chunks = self.text_splitter.split_text(combined_text)
            except Exception as e:
                logger.warning(f"Semantic chunking failed: {e}")
                errors.append(f"Chunking error: {str(e)}")
            
            # Extract metadata
            metadata = self._extract_metadata(pdf_file_path)
            
            # Detect language from text content
            if text_content:
                metadata.language = self._detect_language(text_content)
            
            # Calculate processing stats
            processing_time = (datetime.now() - start_time).total_seconds()
            stats = {
                'processing_time_seconds': processing_time,
                'pages_processed': len(text_content),
                'text_chunks': len(texts),
                'tables_found': len(tables),
                'images_found': len(image_urls),
                'captions_found': len(figure_table_captions),
                'semantic_chunks': len(semantic_chunks),
                'errors_count': len(errors)
            }
            
            logger.info(f"Document processed successfully in {processing_time:.2f}s")
            
            return ProcessingResult(
                text_content=text_content,
                image_urls=image_urls,
                image_info=image_info,
                figure_table_captions=figure_table_captions,
                tables=tables,
                texts=texts,
                semantic_chunks=semantic_chunks,
                document_key=self._generate_document_key(pdf_file_path),
                metadata=metadata,
                processing_stats=stats,
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Failed to process document: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return ProcessingResult(
                text_content=[],
                image_urls=[],
                image_info=[],
                figure_table_captions=[],
                tables=[],
                texts=[],
                semantic_chunks=[],
                document_key="",
                metadata=DocumentMetadata(),
                processing_stats={'processing_time_seconds': 0, 'errors_count': len(errors)},
                errors=errors
            )
    
    def _extract_images_and_text_from_pdf(self, pdf_file_path: str) -> Tuple[List[str], List[str], List[ImageInfo], List[str]]:
        """
        Extracts text and images from a PDF file.
        """
        pdf_document = fitz.open(pdf_file_path)
        image_urls = []
        image_info = []
        text_content = []
        figure_table_captions = []
        
        # Get Firebase bucket if available
        bucket = None
        if self.enable_firebase:
            try:
                bucket = storage.bucket()
            except Exception as e:
                logger.warning(f"Firebase bucket not available: {e}")
                self.enable_firebase = False

        # Track uploaded images to prevent duplicates (works for both Firebase and local)
        uploaded_images = set()

        # Iterate through each page
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)

            # Extract text from this page
            page_text = page.get_text("text")

            # Remove any lines containing IP addresses (existing security logic)
            filtered_lines = []
            for line in page_text.split('\n'):
                if not self.ip_address_pattern.search(line):
                    filtered_lines.append(line)

            # Join filtered lines back into text
            filtered_text = '\n'.join(filtered_lines)
            text_content.append(filtered_text)

            # Extract and upload images with secret keys
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Generate content hash for secret key and duplicate detection
                    content_hash = hashlib.md5(image_bytes).hexdigest()[:16]
                    
                    # Skip if we've already processed this exact image (works for both modes)
                    if content_hash in uploaded_images:
                        logger.debug(f"Skipping duplicate image: page {page_num + 1}, index {img_index + 1}")
                        continue
                    
                    uploaded_images.add(content_hash)
                    secret_key = self._generate_secret_key(page_num + 1, img_index + 1, content_hash)

                    # Convert image to specified format
                    image_pil = Image.open(io.BytesIO(image_bytes))
                    image_pil = image_pil.convert("RGB")

                    if self.enable_firebase and bucket:
                        # Upload to Firebase with secret key as filename
                        image_buffer = io.BytesIO()
                        image_pil.save(image_buffer, format=self.image_format)
                        image_buffer.seek(0)

                        # Use secret key as the blob name instead of predictable filename
                        blob = bucket.blob(f"{secret_key}.{self.image_format.lower()}")
                        blob.upload_from_file(image_buffer, content_type=f"image/{self.image_format.lower()}")

                        blob.make_public()
                        img_url = blob.public_url
                        image_urls.append(img_url)
                        
                        # Store enhanced image info
                        original_filename = f"image_{page_num + 1}_{img_index + 1}.{self.image_format.lower()}"
                        img_info = ImageInfo(
                            public_url=img_url,
                            secret_key=secret_key,
                            original_filename=original_filename,
                            page_number=page_num + 1,
                            image_index=img_index + 1
                        )
                        image_info.append(img_info)
                        logger.debug(f"Uploaded image to Firebase: {secret_key}.{self.image_format.lower()}")
                    else:
                        # Save locally only when Firebase is disabled
                        output_dir = './extracted_images'
                        os.makedirs(output_dir, exist_ok=True)
                        local_filename = f"{secret_key}.{self.image_format.lower()}"
                        local_path = os.path.join(output_dir, local_filename)
                        image_pil.save(local_path, self.image_format)
                        
                        image_urls.append(local_path)
                        
                        # Store enhanced image info for local images too
                        original_filename = f"image_{page_num + 1}_{img_index + 1}.{self.image_format.lower()}"
                        img_info = ImageInfo(
                            public_url=local_path,
                            secret_key=secret_key,
                            original_filename=original_filename,
                            page_number=page_num + 1,
                            image_index=img_index + 1
                        )
                        image_info.append(img_info)
                        logger.debug(f"Saved image locally: {local_path}")
                        
                except Exception as e:
                    logger.warning(f"Error extracting image {img_index} from page {page_num + 1}: {e}")

            # Detect figure/table headings and collect multi-line captions (existing logic)
            lines = filtered_text.split('\n')
            current_caption = None

            for line in lines:
                line_strip = line.strip()

                # Check if this line starts a new figure/table caption
                if self.title_pattern.match(line_strip):
                    # If we were already building a caption, finalize and store it
                    if current_caption:
                        figure_table_captions.append(current_caption.strip())

                    # Start a new caption with the heading
                    current_caption = line_strip
                else:
                    # Continue building caption (existing logic)
                    if current_caption is not None:
                        if line_strip:
                            current_caption += " " + line_strip

            # End of the page: if there's a caption being built, store it
            if current_caption:
                figure_table_captions.append(current_caption.strip())
                current_caption = None

        pdf_document.close()
        return text_content, image_urls, image_info, figure_table_captions
    
    def _extract_tables_with_unstructured(self, pdf_file_path: str) -> List[str]:
        """
        Extract tables using unstructured library for better table detection.
        """
        enhanced_tables = []
        
        try:
            # Use unstructured to extract table elements
            elements = partition_pdf(
                filename=pdf_file_path,
                strategy="hi_res",
                infer_table_structure=True,
                chunking_strategy=None,  # We'll do our own chunking
                max_characters=4000
            )
            
            for element in elements:
                if "Table" in str(type(element)):
                    table_text = str(element)
                    if len(table_text.strip()) > 20:  # Skip very short tables
                        enhanced_tables.append(table_text)
            
            logger.info(f"Extracted {len(enhanced_tables)} tables using unstructured")
            
        except Exception as e:
            logger.warning(f"Enhanced table extraction failed: {e}")
        
        return enhanced_tables
    
    def _process_text(self, text_content: List[str]) -> Tuple[List[str], List[str]]:
        """
        Separates tables and normal text, with structure recognition.
        """
        tables, texts = [], []

        for text in text_content:
            if self._is_table_like(text):
                tables.append(text)
            else:
                # Additional processing for better text structure
                processed_text = self._enhance_text_structure(text)
                texts.append(processed_text)

        return texts, tables
    
    def _enhance_text_structure(self, text: str) -> str:
        """
        Enhance text structure by identifying headers and improving formatting.
        """
        lines = text.split('\n')
        enhanced_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                enhanced_lines.append('')
                continue
            
            # Check if line is a header using multiple patterns
            is_header = any(pattern.match(line) for pattern in self.header_patterns)
            
            if is_header:
                # Mark headers with special formatting for better chunking
                enhanced_lines.append(f"\n## {line}\n")
            else:
                enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def _is_table_like(self, text: str) -> bool:
        """
        Enhanced heuristic to determine if a block of text is table-like.
        """
        
        basic_table_indicators = "\t" in text or " | " in text or "-----" in text
        
        if basic_table_indicators:
            return True
        
        # Enhanced heuristics
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Check for numeric patterns that suggest tabular data
        numeric_lines = 0
        for line in lines:
            if re.search(r'\d+.*\d+.*\d+', line):  # Line with multiple numbers
                numeric_lines += 1
        
        # If more than 50% of lines have multiple numbers, likely a table
        if numeric_lines > len(lines) * 0.5:
            return True
        
        # Check for consistent spacing patterns
        consistent_spacing = True
        first_line_parts = len(lines[0].split())
        for line in lines[1:]:
            if abs(len(line.split()) - first_line_parts) > 2:  # Allow some variation
                consistent_spacing = False
                break
        
        if consistent_spacing and first_line_parts >= 3:
            return True
        
        return False
    
    def _validate_file(self, file_path: str) -> bool:
        """Validate the input file"""
        if not os.path.exists(file_path):
            return False
        if not file_path.lower().endswith('.pdf'):
            return False
        if os.path.getsize(file_path) == 0:
            return False
        return True
    
    def _extract_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract metadata from PDF document"""
        try:
            pdf_document = fitz.open(file_path)
            metadata_dict = pdf_document.metadata
            file_stats = os.stat(file_path)
            
            result = DocumentMetadata(
                title=metadata_dict.get('title'),
                author=metadata_dict.get('author'),
                subject=metadata_dict.get('subject'),
                creator=metadata_dict.get('creator'),
                page_count=len(pdf_document),
                file_size=file_stats.st_size
            )
            
            pdf_document.close()
            return result
            
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            return DocumentMetadata()
    
    def _detect_language(self, text_content: List[str]) -> Optional[str]:
        """Detect document language"""
        try:
            # Combine text from multiple pages for better detection
            combined_text = ' '.join(text_content[:3])  # Use first 3 pages
            if len(combined_text) > 100:
                return detect(combined_text)
        except (LangDetectException, Exception) as e:
            logger.warning(f"Language detection failed: {e}")
        
        return None
    
    def export_results(self, result: ProcessingResult, output_dir: str = None) -> Dict[str, str]:
        """
        Export processing results to Firebase Realtime Database.
        
        Args:
            result: ProcessingResult containing all extracted content
            output_dir: Only used when Firebase is disabled
        
        Returns:
            Dictionary mapping content type to Firebase paths or local file paths
        """
        if self.enable_firebase and hasattr(self, 'database'):
            return self._export_to_firebase(result)
        else:
            # Only export locally if Firebase is disabled
            return self._export_to_local(result, output_dir or './export_output')
    
    def _export_to_firebase(self, result: ProcessingResult) -> Dict[str, str]:
        """Upload all extracted content to Firebase Realtime Database"""
        try:
            doc_key = result.document_key
            timestamp = datetime.now().isoformat().replace(':', '_').replace('.', '_')  # Fix illegal characters
            
            # Prepare image references for database
            image_references = []
            for img_info in result.image_info:
                image_references.append({
                    'storage_filename': f"{img_info.secret_key}.{self.image_format.lower()}",
                    'secret_key': img_info.secret_key,
                    'original_filename': img_info.original_filename,
                    'page_number': img_info.page_number,
                    'image_index': img_info.image_index,
                    'public_url': img_info.public_url
                })
            
            # Prepare complete document data
            document_data = {
                'metadata': {
                    'document_key': doc_key,
                    'created_at': timestamp,
                    'title': result.metadata.title,
                    'author': result.metadata.author,
                    'subject': result.metadata.subject,
                    'creator': result.metadata.creator,
                    'page_count': result.metadata.page_count,
                    'file_size': result.metadata.file_size,
                    'language': result.metadata.language
                },
                'content': {
                    'text_content': result.text_content,
                    'texts': result.texts,
                    'tables': result.tables,
                    'semantic_chunks': result.semantic_chunks,
                    'figure_table_captions': result.figure_table_captions
                },
                'images': {
                    'count': len(image_references),
                    'references': image_references,
                    'urls': result.image_urls  # For backward compatibility
                },
                'processing_stats': {
                    **result.processing_stats,
                    'processed_at': timestamp,
                    'document_key': doc_key
                },
                'errors': result.errors
            }
            
            # Upload to Firebase under documents/{document_key}
            doc_ref = self.database.child('documents').child(doc_key)
            doc_ref.set(document_data)
            
            # Also create an index by timestamp for easier retrieval
            index_ref = self.database.child('document_index').child(timestamp)
            index_ref.set({
                'document_key': doc_key,
                'title': result.metadata.title or 'Untitled',
                'page_count': result.metadata.page_count,
                'image_count': len(image_references),
                'processing_time': result.processing_stats.get('processing_time_seconds', 0)
            })
            
            firebase_paths = {
                'main_document': f"documents/{doc_key}",
                'document_index': f"document_index/{timestamp}",
                'firebase_console': f"https://console.firebase.google.com/project/rem2024-f429b/database/rem2024-f429b-default-rtdb/data/documents/{doc_key}"
            }
            
            logger.info(f"Document {doc_key} uploaded to Firebase Realtime Database successfully")
            logger.info(f"View at: {firebase_paths['firebase_console']}")
            
            return firebase_paths
            
        except Exception as e:
            logger.error(f"Failed to upload to Firebase: {e}")
            raise Exception(f"Firebase upload failed: {e}")  # Make it clear that Firebase failed
    
    def _export_to_local(self, result: ProcessingResult, output_dir: str) -> Dict[str, str]:
        """Fallback method to export results locally"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        
        # Export complete document data as single JSON (matching Firebase structure)
        complete_data = {
            'document_key': result.document_key,
            'metadata': {
                'title': result.metadata.title,
                'author': result.metadata.author,
                'subject': result.metadata.subject,
                'creator': result.metadata.creator,
                'page_count': result.metadata.page_count,
                'file_size': result.metadata.file_size,
                'language': result.metadata.language
            },
            'content': {
                'text_content': result.text_content,
                'texts': result.texts,
                'tables': result.tables,
                'semantic_chunks': result.semantic_chunks,
                'figure_table_captions': result.figure_table_captions
            },
            'images': {
                'count': len(result.image_info),
                'references': [
                    {
                        'storage_filename': f"{img.secret_key}.{self.image_format.lower()}",
                        'secret_key': img.secret_key,
                        'original_filename': img.original_filename,
                        'page_number': img.page_number,
                        'image_index': img.image_index,
                        'public_url': img.public_url
                    } for img in result.image_info
                ],
                'urls': result.image_urls
            },
            'processing_stats': result.processing_stats,
            'errors': result.errors
        }
        
        complete_path = output_path / f"{result.document_key}_complete.json"
        with open(complete_path, 'w', encoding='utf-8') as f:
            json.dump(complete_data, f, indent=2, ensure_ascii=False)
        file_paths['complete_document'] = str(complete_path)
        
        logger.info(f"Results exported locally to {output_dir}")
        return file_paths


# Compatibility functions
def extract_images_and_text_from_pdf(pdf_file_path: str) -> Tuple[List[str], List[str], List[str]]:
    processor = DocumentProcessor()
    text_content, image_urls, image_info, figure_table_captions = processor._extract_images_and_text_from_pdf(pdf_file_path)
    return text_content, image_urls, figure_table_captions  # Maintain backward compatibility


def process_text(text_content: List[str]) -> Tuple[List[str], List[str]]:
    processor = DocumentProcessor()
    return processor._process_text(text_content)


def is_table_like(text: str) -> bool:
    processor = DocumentProcessor()
    return processor._is_table_like(text) 