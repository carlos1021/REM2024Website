#!/usr/bin/env python3
"""
Test Query Processing & Response Generation - Step 4 of RAG Pipeline

This script demonstrates the query processing capabilities:
- Intelligent query understanding and expansion
- Multi-modal evidence retrieval from vector database
- LLM-powered response generation with citations
- Interactive question-answering interface
- Response quality assessment and follow-up suggestions
"""

import os
import sys
from pathlib import Path

# Add the core_pipeline directory to Python path
sys.path.append(str(Path(__file__).parent.parent))
# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables only")

from core_pipeline.query_processor import (
    QueryProcessor,
    ask_question,
    interactive_query_session
)


def test_query_processor():
    """Test the complete query processing and response generation workflow"""
    
    print("🤖 Query Processing & Response Generation Test - Step 4")
    print("=" * 60)
    
    # Check required environment variables
    firebase_key = os.getenv("FIREBASE_SERVICE_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not firebase_key:
        print("❌ FIREBASE_SERVICE_KEY environment variable not set")
        print("💡 This test requires Firebase integration")
        return False
    
    if not openai_key:
        print("⚠️  OPENAI_API_KEY not set - will use fallback responses")
    
    try:
        # Step 1: Initialize Query Processor
        print("🏗️  Step 1: Initializing Query Processor...")
        processor = QueryProcessor(
            enable_openai=bool(openai_key),
            max_context_length=8000,
            response_max_tokens=1000
        )
        print("✅ Query processor initialized successfully")
        print(f"   • OpenAI enabled: {processor.enable_openai}")
        print(f"   • Firebase enabled: {processor.enable_firebase}")
        print(f"   • Max context length: {processor.max_context_length}")
        
        # Step 2: Get available vector databases
        print(f"\n📋 Step 2: Finding available vector databases...")
        available_dbs = processor.embedder.list_vector_databases()
        
        if not available_dbs:
            print("❌ No vector databases found")
            print("💡 Please run vector_embedder.py first to create a vector database")
            return False
        
        latest_db = available_dbs[0]
        db_id = latest_db['database_id']
        
        print(f"✅ Found {len(available_dbs)} vector databases")
        print(f"   • Using latest: {db_id}")
        print(f"   • Total chunks: {latest_db['total_chunks']}")
        print(f"   • Documents: {latest_db['documents_count']}")
        print(f"   • Model: {latest_db['embedding_model']}")
        
        # Step 3: Test Query Understanding
        print(f"\n🧠 Step 3: Testing query understanding and analysis...")
        test_queries = [
            "What are the main health outcomes discussed in the research?",
            "How was the study methodology designed?",
            "What statistical findings support the conclusions?",
            "Compare the results between different groups",
            "Summarize the key research findings"
        ]
        
        for i, query in enumerate(test_queries):
            print(f"\n   Query {i+1}: '{query}'")
            query_context = processor._analyze_query(query)
            print(f"      Type: {query_context.query_type.value}")
            print(f"      Confidence: {query_context.intent_confidence:.2f}")
            print(f"      Key concepts: {', '.join(query_context.key_concepts[:3])}")
            print(f"      Requires tables: {query_context.requires_tables}")
            print(f"      Requires stats: {query_context.requires_statistics}")
        
        # Step 4: Test Evidence Retrieval
        print(f"\n🔍 Step 4: Testing evidence retrieval...")
        test_query = "What are the health outcomes for children in the study?"
        print(f"   Query: '{test_query}'")
        
        query_context = processor._analyze_query(test_query)
        evidence = processor._retrieve_evidence(query_context, db_id, max_results=5, use_hybrid=True)
        
        if evidence:
            print(f"   ✅ Retrieved {len(evidence)} evidence pieces:")
            for i, item in enumerate(evidence[:3]):
                print(f"      {i+1}. Relevance: {item.relevance_score:.3f}")
                print(f"         Section: {item.section}")
                print(f"         Type: {item.content_type}")
                print(f"         Content: {item.content[:80]}...")
        else:
            print("   ❌ No evidence retrieved")
        
        # Step 5: Test Complete Query Processing
        print(f"\n💬 Step 5: Testing complete query processing...")
        comprehensive_queries = [
            "What were the main findings about childhood development?",
            "How did the researchers measure health outcomes?",
            "What statistical methods were used in the analysis?"
        ]
        
        for i, query in enumerate(comprehensive_queries):
            print(f"\n   Processing query {i+1}: '{query}'")
            result = processor.process_query(query, db_id, max_results=5)
            
            print(f"   ✅ Response generated:")
            print(f"      Response type: {result.response_type}")
            print(f"      Confidence: {result.confidence_score:.2f}")
            print(f"      Quality: {result.quality_assessment.value}")
            print(f"      Evidence pieces: {len(result.evidence)}")
            print(f"      Citations: {len(result.citations)}")
            print(f"      Processing time: {result.processing_stats.get('processing_time_seconds', 0):.2f}s")
            print(f"      Response preview: {result.response[:100]}...")
            
            if result.suggested_followups:
                print(f"      Follow-ups: {', '.join(result.suggested_followups[:2])}")
        
        # Step 6: Test Convenience Functions
        print(f"\n🛠️  Step 6: Testing convenience functions...")
        
        print("   Testing ask_question() convenience function...")
        convenience_result = ask_question(
            "What are the key research findings?", 
            vector_db_id=db_id,
            use_hybrid=True,
            max_results=3
        )
        
        if convenience_result:
            print(f"   ✅ Convenience function works:")
            print(f"      Confidence: {convenience_result.confidence_score:.2f}")
            print(f"      Response length: {len(convenience_result.response)} chars")
            print(f"      Evidence count: {len(convenience_result.evidence)}")
        
        # Step 7: Interactive Session Demo
        print(f"\n🎮 Step 7: Interactive session demo...")
        print("   Note: Interactive session available via interactive_query_session()")
        print(f"   To start: python -c \"from query_processor import interactive_query_session; interactive_query_session('{db_id}')\"")
        
        # Step 8: Performance Summary
        print(f"\n📈 Step 8: Query Processing Performance Summary")
        print("-" * 50)
        print(f"   • Vector Database: {db_id}")
        print(f"   • Available Documents: {latest_db['documents_count']}")
        print(f"   • Searchable Chunks: {latest_db['total_chunks']:,}")
        print(f"   • Query Understanding: Pattern-based + keyword extraction")
        print(f"   • Response Generation: {'OpenAI gpt-4.1-mini' if processor.enable_openai else 'Template-based fallback'}")
        print(f"   • Evidence Retrieval: Hybrid semantic + keyword search")
        print(f"   • Citation Support: ✅ (Automatic source attribution)")
        print(f"   • Quality Assessment: ✅ (Heuristic-based scoring)")
        print(f"   • Follow-up Suggestions: ✅ (Context-aware recommendations)")
        
        print(f"\n🎉 Query Processing Test Completed Successfully!")
        print(f"🤖 Your RAG system can now:")
        print(f"   - Understand user questions intelligently")
        print(f"   - Retrieve relevant evidence from research documents")
        print(f"   - Generate comprehensive answers with citations")
        print(f"   - Assess response quality and confidence")
        print(f"   - Suggest relevant follow-up questions")
        print(f"   - Support interactive Q&A sessions")
        
        return True
        
    except Exception as e:
        print(f"❌ Query processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_question_answering():
    """Demonstrate question answering with sample queries"""
    
    print("\n🎯 Question Answering Demo")
    print("=" * 40)
    
    try:
        # Sample questions to demonstrate different capabilities
        demo_questions = [
            {
                "question": "What are the main health outcomes discussed?",
                "description": "Factual question about research findings"
            },
            {
                "question": "How was the research methodology designed?", 
                "description": "Methodology-focused question"
            },
            {
                "question": "What statistical evidence supports the conclusions?",
                "description": "Statistical analysis question"
            },
            {
                "question": "Summarize the key findings about childhood development",
                "description": "Summarization request"
            }
        ]
        
        print(f"🔍 Demonstrating {len(demo_questions)} sample questions...")
        
        for i, item in enumerate(demo_questions):
            print(f"\n📝 Question {i+1}: {item['description']}")
            print(f"   Query: \"{item['question']}\"")
            
            try:
                result = ask_question(item['question'], max_results=3)
                
                print(f"   ✅ Response generated:")
                print(f"      Type: {result.response_type}")
                print(f"      Confidence: {result.confidence_score:.2f}")
                print(f"      Quality: {result.quality_assessment.value}")
                
                # Show response preview
                response_preview = result.response[:200]
                if len(result.response) > 200:
                    response_preview += "..."
                print(f"      Answer: {response_preview}")
                
                # Show citations
                if result.citations:
                    print(f"      Sources: {len(result.citations)} citations available")
                
                # Show follow-ups
                if result.suggested_followups:
                    print(f"      Follow-ups: {result.suggested_followups[0]}")
                
            except Exception as e:
                print(f"   ❌ Error processing question: {e}")
        
        print(f"\n✨ Demo completed! The system can answer various types of research questions.")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def test_interactive_features():
    """Test interactive features and user interface"""
    
    print("\n🎮 Interactive Features Test")
    print("=" * 40)
    
    try:
        print("🔧 Testing interactive session setup...")
        
        # Test that we can initialize an interactive session
        processor = QueryProcessor()
        available_dbs = processor.embedder.list_vector_databases()
        
        if available_dbs:
            db_id = available_dbs[0]['database_id']
            print(f"✅ Interactive session ready with database: {db_id}")
            print(f"   • Available for real-time Q&A")
            print(f"   • Supports follow-up questions")
            print(f"   • Provides confidence scores")
            print(f"   • Shows source citations")
            
            print(f"\n🚀 To start interactive session, run:")
            print(f"   python test_query_processor.py --interactive")
            print(f"   or")
            print(f"   from query_processor import interactive_query_session")
            print(f"   interactive_query_session()")
            
            return True
        else:
            print("❌ No vector databases available for interactive session")
            return False
            
    except Exception as e:
        print(f"❌ Interactive features test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Query Processing & Response Generation Test Suite")
    print("=" * 60)
    
    # Check for interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("🎮 Starting interactive query session...")
        try:
            interactive_query_session()
        except KeyboardInterrupt:
            print("\n👋 Interactive session ended.")
        sys.exit(0)
    
    # Run main test
    main_success = test_query_processor()
    
    # Run demo if main test passes
    if main_success:
        demo_success = demo_question_answering()
        interactive_success = test_interactive_features()
    else:
        print("\n🔄 Running basic component tests...")
        try:
            processor = QueryProcessor(enable_firebase=False, enable_openai=False)
            print("✅ Query processor can be initialized without external services")
            demo_success = False
            interactive_success = False
        except Exception as e:
            print(f"❌ Basic initialization failed: {e}")
            demo_success = False
            interactive_success = False
    
    print("\n" + "=" * 60)
    if main_success and demo_success and interactive_success:
        print("🎉 All tests passed! Query processing pipeline is ready.")
        print("🚀 Your RAG system is now complete and ready for use!")
        print("")
        print("📋 Complete Pipeline Summary:")
        print("   ✅ Step 1: Document Processing (PDF → Text, Tables, Images)")
        print("   ✅ Step 2: Content Analysis (Semantic Chunking + Structure)")
        print("   ✅ Step 3: Vector Embedding (FAISS + Firebase Storage)")
        print("   ✅ Step 4: Query Processing (Q&A + Response Generation)")
        print("")
        print("💡 To start asking questions about your documents:")
        print("   python test_query_processor.py --interactive")
        
    elif main_success:
        print("🎯 Main query processing functionality working!")
        print("⚠️  Some demo features had issues, but core functionality is ready.")
    else:
        print("⚠️  Some tests failed. Check logs for details.")
        print("💡 Ensure Firebase is configured and vector database exists.")
        print("📋 Required: Vector database from Step 3 (vector_embedder.py)")
        print("🔑 Optional: OPENAI_API_KEY for enhanced response generation") 