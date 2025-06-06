#!/usr/bin/env python3
"""
Test Content Analysis & Chunking - Step 2 of RAG Pipeline

This script demonstrates the content analyzer capabilities:
- Retrieval of processed documents from Firebase
- Academic structure recognition and analysis
- Multi-modal content integration (text + tables + figures)
- Enhanced semantic chunking with context preservation
- Export of enhanced analysis back to Firebase
"""

import os
import sys
from pathlib import Path

# Add the core_pipeline directory to Python path
sys.path.append(str(Path(__file__).parent))
import core_pipeline.content_analyzer

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables only")

from core_pipeline.content_analyzer import (
    ContentAnalyzer, 
    analyze_document_from_firebase,
    list_processed_documents,
    export_enhanced_analysis
)


def test_content_analyzer():
    """Test the complete content analysis workflow"""
    
    print("🧠 Content Analysis & Chunking Test - Step 2")
    print("=" * 60)
    
    # Check Firebase environment
    firebase_key = os.getenv("FIREBASE_SERVICE_KEY")
    if not firebase_key:
        print("❌ FIREBASE_SERVICE_KEY environment variable not set")
        print("💡 This test requires Firebase integration")
        return False
    
    try:
        # Step 1: List available processed documents
        print("📋 Step 1: Listing processed documents from Firebase...")
        documents = list_processed_documents()
        
        if not documents:
            print("❌ No processed documents found in Firebase")
            print("💡 Please run document_processor.py first to process a document")
            return False
        
        print(f"✅ Found {len(documents)} processed documents:")
        for i, doc in enumerate(documents[:3]):  # Show first 3
            print(f"   {i+1}. {doc['title'][:60]}...")
            print(f"      Document Key: {doc['document_key']}")
            print(f"      Processed: {doc['processed_at']}")
            print(f"      Pages: {doc['page_count']}, Images: {doc['image_count']}")
        
        if len(documents) > 3:
            print(f"   ... and {len(documents) - 3} more documents")
        
        # Step 2: Analyze the most recent document
        print(f"\n🔬 Step 2: Analyzing document content...")
        latest_doc = documents[0]
        document_key = latest_doc['document_key']
        
        print(f"📄 Analyzing: {latest_doc['title'][:60]}...")
        print(f"🔑 Document Key: {document_key}")
        
        # Initialize content analyzer
        analyzer = ContentAnalyzer(
            enable_openai=True,  # Enable OpenAI for keyword extraction
            chunk_size=800,
            chunk_overlap=150
        )
        
        # Perform analysis
        analysis_result = analyzer.analyze_document(document_key)
        
        if analysis_result.errors:
            print("⚠️  Analysis completed with warnings:")
            for error in analysis_result.errors:
                print(f"   - {error}")
        else:
            print("✅ Analysis completed successfully!")
        
        # Step 3: Display analysis results
        print(f"\n📊 Step 3: Analysis Results Summary")
        print("-" * 40)
        stats = analysis_result.processing_stats
        print(f"   • Processing Time: {stats.get('processing_time_seconds', 0):.2f}s")
        print(f"   • Enhanced Chunks: {stats.get('enhanced_chunks', 0)}")
        print(f"   • Document Sections: {stats.get('document_sections', 0)}")
        print(f"   • Multi-modal Relationships: {stats.get('multimodal_relationships', 0)}")
        print(f"   • Total Word Count: {stats.get('total_word_count', 0):,}")
        
        # Show document structure
        print(f"\n🏗️  Document Structure:")
        structure = analysis_result.document_structure
        sections = structure.get('sections', [])
        for section in sections[:5]:  # Show first 5 sections
            print(f"   📑 {section['title'][:50]}")
            print(f"      Type: {section['type']}, Blocks: {len(section['content_blocks'])}")
        
        if len(sections) > 5:
            print(f"   ... and {len(sections) - 5} more sections")
        
        # Show sample enhanced chunks
        print(f"\n🧩 Sample Enhanced Chunks:")
        chunks = analysis_result.enhanced_chunks
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"   Chunk {i+1}: {chunk.chunk_id}")
            print(f"      Type: {chunk.chunk_type.value}, Context: {chunk.context_level.value}")
            print(f"      Section: {chunk.section_title}")
            print(f"      Words: {chunk.word_count}, Position: {chunk.position_in_document:.2f}")
            if chunk.semantic_keywords:
                print(f"      Keywords: {', '.join(chunk.semantic_keywords[:3])}...")
            print(f"      Content: {chunk.content[:100]}...")
            
            # Show relationships
            if chunk.related_tables or chunk.related_figures:
                relationships = []
                if chunk.related_tables:
                    relationships.append(f"{len(chunk.related_tables)} tables")
                if chunk.related_figures:
                    relationships.append(f"{len(chunk.related_figures)} figures")
                print(f"      Related: {', '.join(relationships)}")
            print()
        
        # Show multi-modal relationships
        if analysis_result.multi_modal_relationships:
            print(f"🔗 Multi-modal Relationships:")
            relationships = analysis_result.multi_modal_relationships
            for chunk_id, relations in list(relationships.items())[:3]:
                print(f"   {chunk_id}:")
                if relations.get('tables'):
                    print(f"      → Tables: {len(relations['tables'])}")
                if relations.get('figures'):
                    print(f"      → Figures: {len(relations['figures'])}")
                if relations.get('captions'):
                    print(f"      → Captions: {len(relations['captions'])}")
        
        # Step 4: Export enhanced analysis to Firebase
        print(f"\n💾 Step 4: Exporting enhanced analysis to Firebase...")
        firebase_paths = export_enhanced_analysis(analysis_result)
        
        print("✅ Enhanced analysis exported successfully!")
        print("📍 Firebase Locations:")
        for key, path in firebase_paths.items():
            print(f"   • {key}: {path}")
        
        # Step 5: Verify the export
        print(f"\n🔍 Step 5: Verifying export...")
        summary = analyzer.get_analysis_summary(document_key)
        
        if summary:
            print("✅ Enhanced analysis verified in Firebase!")
            print(f"   • Analyzed at: {summary['analyzed_at']}")
            print(f"   • Chunks: {summary['chunk_count']}")
            print(f"   • Sections: {summary['section_count']}")
            print(f"   • Relationships: {summary['multimodal_relationships']}")
        else:
            print("❌ Failed to verify export")
        
        print(f"\n🎉 Content Analysis Test Completed Successfully!")
        print(f"📊 Your document has been enhanced with:")
        print(f"   - Academic structure recognition")
        print(f"   - Context-preserving semantic chunks")
        print(f"   - Multi-modal content relationships")
        print(f"   - Hierarchical document organization")
        print(f"   - Semantic keyword extraction")
        
        return True
        
    except Exception as e:
        print(f"❌ Content analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analyzer_components():
    """Test individual analyzer components"""
    
    print("\n🔧 Component Testing")
    print("=" * 40)
    
    try:
        # Test analyzer initialization
        print("🏗️  Testing analyzer initialization...")
        analyzer = ContentAnalyzer(enable_openai=False)  # Disable OpenAI for speed
        print("✅ Analyzer initialized successfully")
        
        # Test document listing
        print("📋 Testing document listing...")
        docs = analyzer.list_available_documents()
        print(f"✅ Found {len(docs)} documents in Firebase")
        
        return True
        
    except Exception as e:
        print(f"❌ Component testing failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Content Analysis & Chunking Test Suite")
    print("=" * 60)
    
    # Run main test
    main_success = test_content_analyzer()
    
    # Run component tests if main test fails
    if not main_success:
        print("\n🔄 Running component tests for debugging...")
        component_success = test_analyzer_components()
    
    print("\n" + "=" * 60)
    if main_success:
        print("🎉 All tests passed! Content analysis pipeline is ready.")
    else:
        print("⚠️  Some tests failed. Check logs for details.")
        print("💡 Ensure Firebase is configured and documents are processed.") 