"""
Test script for the enhanced document processor.

This script tests the document processor with paper.pdf and shows
the enhanced capabilities including semantic chunking and table detection.
"""

import os
import sys
import json
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables only")

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core_pipeline.document_processor import DocumentProcessor, extract_images_and_text_from_pdf


def main():
    """Test the document processor with paper.pdf"""
    
    print("🔬 Testing Enhanced Document Processor")
    print("=" * 50)
    
    # Check if paper.pdf exists
    pdf_path = "paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Error: {pdf_path} not found in current directory")
        print("📁 Current directory contents:")
        for item in os.listdir('.'):
            print(f"   - {item}")
        return
    
    print(f"📄 Processing: {pdf_path}")
    print()
    
    # Check if Firebase is available
    firebase_key = os.getenv("FIREBASE_SERVICE_KEY")
    use_firebase = firebase_key is not None
    
    if use_firebase:
        print("🔥 Firebase environment detected - running Firebase integration test")
        firebase_success = test_firebase_integration()
        
        print("\n" + "=" * 60)
        print(f"📈 Firebase Integration Test: {'✅ PASS' if firebase_success else '❌ FAIL'}")
        
        if firebase_success:
            print(f"\n🎉 Firebase integration is working! Your extracted content is stored in:")
            print(f"   - Images: Firebase Storage with secret key filenames") 
            print(f"   - Data: Firebase Realtime Database with organized structure")
            print(f"   - Console: https://console.firebase.google.com/project/rem2024-f429b")
    else:
        print("📁 Firebase not configured - running local mode tests")
        
        # Test 1: Basic compatibility (like existing app.py)
        print("🧪 Test 1: Basic Compatibility Mode")
        print("-" * 30)
        try:
            text_content, image_urls, captions = extract_images_and_text_from_pdf(pdf_path)
            print(f"✅ Extracted {len(text_content)} pages of text")
            print(f"✅ Found {len(image_urls)} images")
            print(f"✅ Detected {len(captions)} captions")
            
            if captions:
                print("📝 Sample captions:")
                for i, caption in enumerate(captions[:3]):  # Show first 3
                    print(f"   {i+1}. {caption[:100]}...")
            print()
        except Exception as e:
            print(f"❌ Basic test failed: {e}")
            return

        # Test 2: Enhanced processor with local storage
        print("🚀 Test 2: Enhanced Processing (Local Mode)")
        print("-" * 30)
        try:
            # Initialize with Firebase disabled
            processor = DocumentProcessor(
                enable_firebase=False,
                chunk_size=800,
                chunk_overlap=100
            )
            
            result = processor.process_document(pdf_path)
            
            # Display results
            print("📊 Processing Results:")
            print(f"   • Pages processed: {result.processing_stats.get('pages_processed', 0)}")
            print(f"   • Text chunks: {result.processing_stats.get('text_chunks', 0)}")
            print(f"   • Tables found: {result.processing_stats.get('tables_found', 0)}")
            print(f"   • Images found: {result.processing_stats.get('images_found', 0)}")
            print(f"   • Captions found: {result.processing_stats.get('captions_found', 0)}")
            print(f"   • Semantic chunks: {result.processing_stats.get('semantic_chunks', 0)}")
            print(f"   • Processing time: {result.processing_stats.get('processing_time_seconds', 0):.2f}s")
            print(f"   • Language detected: {result.metadata.language or 'Unknown'}")
            print()
            
            # Show errors if any
            if result.errors:
                print("⚠️  Processing Warnings/Errors:")
                for error in result.errors:
                    print(f"   - {error}")
                print()
            
            # Test 3: Export results locally
            print("💾 Test 3: Local Export Functionality")
            print("-" * 30)
            output_dir = "test_output"
            try:
                file_paths = processor.export_results(result, output_dir)
                print(f"✅ Results exported to: {output_dir}/")
                print("📁 Generated files:")
                for content_type, file_path in file_paths.items():
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    print(f"   • {content_type}: {os.path.basename(file_path)} ({file_size:,} bytes)")
                print()
            except Exception as e:
                print(f"❌ Export failed: {e}")
            
            # Show sample content
            if result.texts:
                print("📄 Sample text (first chunk):")
                sample_text = result.texts[0][:200] + "..." if len(result.texts[0]) > 200 else result.texts[0]
                print(f"   {sample_text}")
                print()
            
            if result.semantic_chunks:
                print("🧠 Semantic Chunks (first 3):")
                for i, chunk in enumerate(result.semantic_chunks[:3]):
                    chunk_preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
                    print(f"   Chunk {i+1}: {chunk_preview}")
                print()
                
            print("🎉 Local mode tests completed successfully!")
            
        except Exception as e:
            print(f"❌ Enhanced processing failed: {e}")
            import traceback
            traceback.print_exc()


def test_individual_components():
    """Test individual components separately for debugging"""
    
    print("\n🔧 Component-Level Testing")
    print("=" * 50)
    
    pdf_path = "paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ {pdf_path} not found")
        return
    
    # Test just the processor initialization
    print("🏗️  Testing processor initialization...")
    try:
        processor = DocumentProcessor(
            enable_firebase=False
        )
        print("✅ Basic processor initialized")
    except Exception as e:
        print(f"❌ Processor initialization failed: {e}")
        return
    
    # Test file validation
    print("📋 Testing file validation...")
    if processor._validate_file(pdf_path):
        print("✅ File validation passed")
    else:
        print("❌ File validation failed")
        return
    
    # Test metadata extraction
    print("📊 Testing metadata extraction...")
    try:
        metadata = processor._extract_metadata(pdf_path)
        print(f"✅ Metadata extracted: {metadata.page_count} pages")
    except Exception as e:
        print(f"❌ Metadata extraction failed: {e}")
    
    # Test basic text extraction
    print("📄 Testing basic text extraction...")
    try:
        text_content, image_urls, captions = processor._extract_images_and_text_from_pdf(pdf_path)
        print(f"✅ Text extraction: {len(text_content)} pages, {len(image_urls)} images")
    except Exception as e:
        print(f"❌ Text extraction failed: {e}")
    
    # Test semantic chunking
    print("🧠 Testing semantic chunking...")
    try:
        if text_content:
            chunks = processor.text_splitter.split_text(" ".join(text_content[:2]))  # Test with first 2 pages
            print(f"✅ Semantic chunking: {len(chunks)} chunks created")
        else:
            print("❌ No text available for chunking")
    except Exception as e:
        print(f"❌ Semantic chunking failed: {e}")
    
    # Test enhanced table detection
    print("📊 Testing enhanced table detection...")
    try:
        enhanced_tables = processor._extract_tables_with_unstructured(pdf_path)
        print(f"✅ Enhanced table detection: {len(enhanced_tables)} tables found")
    except Exception as e:
        print(f"❌ Enhanced table detection failed: {e}")


def test_firebase_integration():
    """Test Firebase integration with environment variable authentication"""
    
    print("\n🔥 Firebase Integration Test")
    print("=" * 50)
    
    # Check if FIREBASE_SERVICE_KEY environment variable is set
    firebase_key = os.getenv("FIREBASE_SERVICE_KEY")
    if not firebase_key:
        print("❌ FIREBASE_SERVICE_KEY environment variable not set")
        print("💡 To test Firebase integration:")
        print("   1. Set FIREBASE_SERVICE_KEY environment variable")
        print("   2. Run the test again")
        return False
    
    pdf_path = "paper.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ {pdf_path} not found")
        return False
    
    print("🚀 Testing Firebase integration with environment credentials...")
    
    try:
        # Initialize processor with Firebase enabled
        processor = DocumentProcessor(
            enable_firebase=True,
            chunk_size=800,
            chunk_overlap=100
        )
        
        print("✅ Firebase processor initialized")
        
        # Process the document
        print(f"📄 Processing document: {pdf_path}")
        result = processor.process_document(pdf_path)
        
        print("✅ Document processed successfully!")
        print(f"📊 Processing Results:")
        print(f"   • Document Key: {result.document_key}")
        print(f"   • Pages: {result.processing_stats.get('pages_processed', 0)}")
        print(f"   • Images: {result.processing_stats.get('images_found', 0)}")
        print(f"   • Tables: {result.processing_stats.get('tables_found', 0)}")
        print(f"   • Semantic Chunks: {result.processing_stats.get('semantic_chunks', 0)}")
        print(f"   • Processing Time: {result.processing_stats.get('processing_time_seconds', 0):.2f}s")
        
        # Test Firebase upload
        print("\n🔥 Uploading to Firebase...")
        firebase_paths = processor.export_results(result)
        
        print("✅ Firebase upload successful!")
        print("📍 Firebase Locations:")
        for key, path in firebase_paths.items():
            print(f"   • {key}: {path}")
        
        # Show image organization with secret keys
        if result.image_info:
            print(f"\n🖼️  Image Storage Organization:")
            print(f"   📊 Total Images: {len(result.image_info)}")
            print(f"   🔐 Secret Key Examples:")
            for i, img in enumerate(result.image_info[:3]):  # Show first 3
                print(f"     Image {i+1}:")
                print(f"       - Secret Key: {img.secret_key}")
                print(f"       - Storage File: {img.secret_key}.png")
                print(f"       - Original: {img.original_filename}")
                print(f"       - Page: {img.page_number}, Index: {img.image_index}")
            
            if len(result.image_info) > 3:
                print(f"     ... and {len(result.image_info) - 3} more images")
        
        print(f"\n🎯 Firebase Database Structure:")
        print(f"   documents/{result.document_key}/")
        print(f"   ├── metadata/ (title, author, language, etc.)")
        print(f"   ├── content/")
        print(f"   │   ├── text_content[] ({len(result.text_content)} pages)")
        print(f"   │   ├── texts[] ({len(result.texts)} chunks)")
        print(f"   │   ├── tables[] ({len(result.tables)} tables)")
        print(f"   │   ├── semantic_chunks[] ({len(result.semantic_chunks)} chunks)")
        print(f"   │   └── figure_table_captions[] ({len(result.figure_table_captions)} captions)")
        print(f"   ├── images/")
        print(f"   │   ├── count: {len(result.image_info)}")
        print(f"   │   └── references[] (with secret keys & storage filenames)")
        print(f"   ├── processing_stats/")
        print(f"   └── errors[] ({len(result.errors)} errors)")
        
        # Firebase console link
        console_url = firebase_paths.get('firebase_console', 'N/A')
        if console_url != 'N/A':
            print(f"\n🌐 View in Firebase Console:")
            print(f"   {console_url}")
        
        print(f"\n🎉 Firebase integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Firebase integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
    
    # Uncomment the line below for component-level debugging if main() fails
    # test_individual_components() 