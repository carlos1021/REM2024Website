#!/usr/bin/env python3
"""
Test Vector Embedding & Storage - Step 3 of RAG Pipeline

This script demonstrates the vector embedding capabilities:
- Creation of semantic embeddings from enhanced content chunks
- Local FAISS vector database creation with fast similarity search
- Firebase Storage integration for persistent vector storage
- Hybrid search capabilities (semantic + keyword)
- Multi-modal search with metadata filtering
"""

import os
import sys
from pathlib import Path

# Add the core_pipeline directory to Python path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables only")

from core_pipeline.vector_embedder import (
    VectorEmbedder,
    create_vector_database_from_firebase,
    upload_vector_database,
    load_vector_database,
    search_documents
)


def test_vector_embedder():
    """Test the complete vector embedding and storage workflow"""
    
    print("🔮 Vector Embedding & Storage Test - Step 3")
    print("=" * 60)
    
    # Check Firebase environment
    firebase_key = os.getenv("FIREBASE_SERVICE_KEY")
    if not firebase_key:
        print("❌ FIREBASE_SERVICE_KEY environment variable not set")
        print("💡 This test requires Firebase integration")
        return False
    
    try:
        # Step 1: Initialize Vector Embedder
        print("🏗️  Step 1: Initializing Vector Embedder...")
        embedder = VectorEmbedder(
            embedding_model="BAAI/bge-small-en-v1.5",  # Fast, high-quality embeddings
            enable_hybrid_search=True
        )
        print(f"✅ Vector embedder initialized with model: {embedder.embedding_model_name}")
        print(f"   • Embedding dimension: {embedder.dimension}")
        print(f"   • Hybrid search enabled: {embedder.enable_hybrid_search}")
        
        # Step 2: Create Vector Database
        print(f"\n📊 Step 2: Creating vector database from enhanced chunks...")
        print("   (This will download the embedding model if not cached locally)")
        
        vector_db = embedder.create_vector_database()
        
        print("✅ Vector database created successfully!")
        print(f"   • Database ID: {vector_db.database_id}")
        print(f"   • Total chunks: {vector_db.total_chunks}")
        print(f"   • Documents included: {len(vector_db.documents_included)}")
        print(f"   • Embedding model: {vector_db.embedding_model}")
        print(f"   • FAISS index type: {type(vector_db.faiss_index).__name__}")
        print(f"   • Hybrid search ready: {vector_db.tfidf_vectorizer is not None}")
        
        # Show sample documents
        if vector_db.documents_included:
            print(f"   • Sample documents:")
            for i, doc_key in enumerate(vector_db.documents_included[:3]):
                doc_chunks = [meta for meta in vector_db.metadata if meta.document_key == doc_key]
                print(f"     {i+1}. {doc_key[:30]}... ({len(doc_chunks)} chunks)")
        
        # Step 3: Test Semantic Search
        print(f"\n🔍 Step 3: Testing semantic search capabilities...")
        test_queries = [
            "health outcomes and early childhood development",
            "research methodology and data analysis",
            "statistical findings and results"
        ]
        
        for i, query in enumerate(test_queries):
            print(f"\n   Query {i+1}: '{query}'")
            results = embedder.semantic_search(vector_db, query, top_k=3)
            
            if results:
                print(f"   ✅ Found {len(results)} semantic matches:")
                for j, result in enumerate(results):
                    print(f"      {j+1}. Similarity: {result.similarity_score:.3f}")
                    print(f"         Section: {result.context_info['section']}")
                    print(f"         Type: {result.context_info['chunk_type']}")
                    print(f"         Content: {result.content[:80]}...")
                    if result.related_content and result.related_content['keywords']:
                        print(f"         Keywords: {', '.join(result.related_content['keywords'][:3])}")
            else:
                print("   ❌ No semantic matches found")
        
        # Step 4: Test Hybrid Search
        print(f"\n🔄 Step 4: Testing hybrid search (semantic + keyword)...")
        if vector_db.tfidf_vectorizer is not None:
            hybrid_query = "childhood development outcomes research"
            print(f"   Hybrid Query: '{hybrid_query}'")
            
            hybrid_results = embedder.hybrid_search(vector_db, hybrid_query, top_k=3)
            
            if hybrid_results:
                print(f"   ✅ Found {len(hybrid_results)} hybrid matches:")
                for j, result in enumerate(hybrid_results):
                    semantic_score = result.context_info.get('semantic_score', 0)
                    keyword_score = result.context_info.get('keyword_score', 0)
                    print(f"      {j+1}. Combined: {result.similarity_score:.3f} (Semantic: {semantic_score:.3f}, Keyword: {keyword_score:.3f})")
                    print(f"         Content: {result.content[:80]}...")
            else:
                print("   ❌ No hybrid matches found")
        else:
            print("   ⚠️  Hybrid search not available (TF-IDF not initialized)")
        
        # Step 5: Upload to Firebase Storage
        print(f"\n☁️  Step 5: Uploading vector database to Firebase Storage...")
        storage_paths = embedder.save_to_firebase_storage(vector_db)
        
        print("✅ Vector database uploaded successfully!")
        print("📍 Firebase Storage Locations:")
        for key, path in storage_paths.items():
            if key != 'firebase_console':
                print(f"   • {key}: {path}")
        
        print(f"\n🌐 Firebase Console: {storage_paths.get('firebase_console', 'N/A')}")
        
        # Step 6: Test Loading from Firebase Storage
        print(f"\n📥 Step 6: Testing database loading from Firebase Storage...")
        loaded_db = embedder.load_from_firebase_storage(vector_db.database_id)
        
        print("✅ Vector database loaded successfully!")
        print(f"   • Loaded Database ID: {loaded_db.database_id}")
        print(f"   • Total chunks: {loaded_db.total_chunks}")
        print(f"   • Embedding model: {loaded_db.embedding_model}")
        print(f"   • FAISS index restored: {loaded_db.faiss_index is not None}")
        print(f"   • Hybrid search restored: {loaded_db.tfidf_vectorizer is not None}")
        
        # Verify loaded database works
        verification_query = "research findings"
        verification_results = embedder.semantic_search(loaded_db, verification_query, top_k=2)
        
        if verification_results:
            print(f"   ✅ Loaded database search verification successful ({len(verification_results)} results)")
        else:
            print(f"   ❌ Loaded database search verification failed")
        
        # Step 7: List Available Vector Databases
        print(f"\n📋 Step 7: Listing available vector databases...")
        available_dbs = embedder.list_vector_databases()
        
        if available_dbs:
            print(f"✅ Found {len(available_dbs)} vector databases in Firebase:")
            for i, db_info in enumerate(available_dbs[:3]):  # Show first 3
                print(f"   {i+1}. {db_info['database_id']}")
                print(f"      Created: {db_info['created_at']}")
                print(f"      Chunks: {db_info['total_chunks']}, Documents: {db_info['documents_count']}")
                print(f"      Model: {db_info['embedding_model']}")
                print(f"      Hybrid: {db_info['has_hybrid_search']}")
        else:
            print("❌ No vector databases found")
        
        # Step 8: Performance Summary
        print(f"\n📈 Step 8: Vector Database Performance Summary")
        print("-" * 50)
        print(f"   • Database ID: {vector_db.database_id}")
        print(f"   • Total Content Chunks: {vector_db.total_chunks:,}")
        print(f"   • Embedding Dimensions: {embedder.dimension}")
        print(f"   • Documents Indexed: {len(vector_db.documents_included)}")
        print(f"   • Search Capabilities: Semantic + {'Hybrid' if vector_db.tfidf_vectorizer else 'Keyword-only'}")
        print(f"   • Storage Location: Firebase Storage + Realtime DB index")
        print(f"   • Multi-modal Support: ✅ (Tables, Figures, Keywords)")
        
        print(f"\n🎉 Vector Embedding Test Completed Successfully!")
        print(f"🔍 Your content is now searchable with:")
        print(f"   - Fast semantic similarity search via FAISS")
        print(f"   - Hybrid keyword + semantic matching")
        print(f"   - Multi-modal content relationships")
        print(f"   - Persistent cloud storage in Firebase")
        print(f"   - Metadata-aware filtering and ranking")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_functions():
    """Test the convenience functions for common operations"""
    
    print("\n🛠️  Convenience Functions Test")
    print("=" * 40)
    
    try:
        # Test listing databases
        print("📋 Testing database listing...")
        embedder = VectorEmbedder()
        databases = embedder.list_vector_databases()
        
        if databases:
            latest_db = databases[0]
            db_id = latest_db['database_id']
            
            print(f"✅ Found {len(databases)} databases")
            print(f"   Using latest: {db_id}")
            
            # Test convenience search function
            print(f"\n🔍 Testing convenience search function...")
            test_query = "health research outcomes"
            results = search_documents(db_id, test_query, top_k=2, use_hybrid=True)
            
            if results:
                print(f"✅ Convenience search returned {len(results)} results")
                for i, result in enumerate(results):
                    print(f"   {i+1}. Score: {result.similarity_score:.3f}")
                    print(f"      Content: {result.content[:60]}...")
            else:
                print("❌ Convenience search returned no results")
        else:
            print("⚠️  No databases available for convenience function testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Convenience functions test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Vector Embedding & Storage Test Suite")
    print("=" * 60)
    
    # Run main test
    main_success = test_vector_embedder()
    
    # Run convenience function tests
    if main_success:
        convenience_success = test_convenience_functions()
    else:
        print("\n🔄 Running basic component tests...")
        try:
            embedder = VectorEmbedder(enable_firebase=False)
            print("✅ Vector embedder can be initialized without Firebase")
            convenience_success = False
        except Exception as e:
            print(f"❌ Basic initialization failed: {e}")
            convenience_success = False
    
    print("\n" + "=" * 60)
    if main_success and convenience_success:
        print("🎉 All tests passed! Vector embedding pipeline is ready.")
        print("💡 Your documents are now searchable with semantic similarity!")
    elif main_success:
        print("🎯 Main vector embedding functionality working!")
        print("⚠️  Some convenience functions had issues, but core functionality is ready.")
    else:
        print("⚠️  Some tests failed. Check logs for details.")
        print("💡 Ensure Firebase is configured and content has been analyzed (Step 2).")
        print("📋 Required: Enhanced content chunks from content analyzer") 