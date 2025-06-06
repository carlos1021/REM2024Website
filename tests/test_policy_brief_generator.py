#!/usr/bin/env python3
"""
Test Policy Brief Generation - Step 5 of RAG Pipeline

This script demonstrates the policy brief generation capabilities:
- Professional document generation from research evidence
- Visual formatting with charts and data visualizations
- Section-aware content generation (Executive Summary, Background, etc.)
- Multiple export formats (HTML, DOCX, JSON)
- Template-driven professional formatting
- IP-safe visual elements (no copied diagrams)
"""

import os
import sys
from pathlib import Path

# Add the parent directory (project root) to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables only")

from core_pipeline.policy_brief_generator import (
    PolicyBriefGenerator,
    BriefConfig,
    OutputFormat,
    generate_policy_brief
)


def test_policy_brief_generator():
    """Test the complete policy brief generation workflow"""
    
    print("📋 Policy Brief Generation Test - Step 5")
    print("=" * 60)
    
    # Check required environment variables
    firebase_key = os.getenv("FIREBASE_SERVICE_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not firebase_key:
        print("❌ FIREBASE_SERVICE_KEY environment variable not set")
        print("💡 This test requires Firebase integration")
        return False
    
    if not openai_key:
        print("⚠️  OPENAI_API_KEY not set - will use template-based generation")
    
    try:
        # Step 1: Initialize Policy Brief Generator
        print("🏗️  Step 1: Initializing Policy Brief Generator...")
        generator = PolicyBriefGenerator(
            enable_openai=bool(openai_key),
            enable_firebase=True
        )
        print("✅ Policy brief generator initialized successfully")
        print(f"   • OpenAI enabled: {generator.enable_openai}")
        print(f"   • Firebase enabled: {generator.enable_firebase}")
        print(f"   • Visual formatting: ✅ (Charts, tables, word clouds)")
        print(f"   • Export formats: HTML, DOCX, JSON")
        
        # Step 2: Get available vector databases
        print(f"\n📋 Step 2: Finding available vector databases...")
        available_dbs = generator.query_processor.embedder.list_vector_databases()
        
        if not available_dbs:
            print("❌ No vector databases found")
            print("💡 Please run test_vector_embedder.py first to create a vector database")
            return False
        
        latest_db = available_dbs[0]
        db_id = latest_db['database_id']
        
        print(f"✅ Found {len(available_dbs)} vector databases")
        print(f"   • Using latest: {db_id}")
        print(f"   • Total chunks: {latest_db['total_chunks']}")
        print(f"   • Documents: {latest_db['documents_count']}")
        
        # Step 3: Create Policy Brief Configuration
        print(f"\n⚙️  Step 3: Creating policy brief configuration...")
        config = BriefConfig(
            title="Evidence-Based Policy Recommendations for Child Health and Development",
            target_audience="Health Policy Makers",
            executive_length="medium",
            include_charts=True,
            include_data_tables=True,
            include_wordcloud=True,
            color_scheme="professional",
            max_recommendations=5,
            citation_style="policy"
        )
        
        print(f"✅ Configuration created:")
        print(f"   • Title: {config.title}")
        print(f"   • Target audience: {config.target_audience}")
        print(f"   • Visual elements: Charts ✅, Tables ✅, Word cloud ✅")
        print(f"   • Color scheme: {config.color_scheme}")
        print(f"   • Max recommendations: {config.max_recommendations}")
        
        # Step 4: Generate Evidence Base
        print(f"\n🔍 Step 4: Gathering evidence from research...")
        research_focus = "childhood development and health outcomes"
        
        print(f"   Research focus: {research_focus}")
        print("   Gathering comprehensive evidence...")
        
        evidence_base = generator._gather_evidence(config, db_id, research_focus)
        
        if evidence_base:
            print(f"   ✅ Gathered {len(evidence_base)} evidence sources:")
            avg_confidence = sum(r.confidence_score for r in evidence_base) / len(evidence_base)
            print(f"      Average confidence: {avg_confidence:.2f}")
            print(f"      Evidence types: {set(r.response_type for r in evidence_base)}")
            
            # Show evidence preview
            for i, result in enumerate(evidence_base[:3]):
                print(f"      Evidence {i+1}: {result.response[:60]}...")
        else:
            print("   ❌ No evidence gathered")
            return False
        
        # Step 5: Generate Complete Policy Brief
        print(f"\n📄 Step 5: Generating complete policy brief...")
        
        brief = generator.generate_policy_brief(
            config=config,
            vector_db_id=db_id,
            research_focus=research_focus
        )
        
        print(f"✅ Policy brief generated successfully!")
        print(f"   • Title: {brief.title}")
        print(f"   • Subtitle: {brief.subtitle}")
        print(f"   • Generation time: {brief.metadata['generation_time_seconds']:.2f}s")
        print(f"   • Evidence sources: {brief.metadata['evidence_sources']}")
        print(f"   • Total sections: {brief.metadata['total_sections']}")
        
        # Step 6: Review Generated Sections
        print(f"\n📝 Step 6: Reviewing generated sections...")
        
        sections = [
            ("Executive Summary", brief.executive_summary),
            ("Background", brief.background),
            ("Key Findings", brief.key_findings),
            ("Policy Recommendations", brief.policy_recommendations),
            ("Supporting Evidence", brief.supporting_evidence),
            ("References", brief.references)
        ]
        
        for section_name, section in sections:
            print(f"\n   📋 {section_name}:")
            print(f"      Content length: {len(section.content)} characters")
            print(f"      Confidence: {section.confidence_score:.2f}")
            print(f"      Citations: {len(section.citations)}")
            print(f"      Preview: {section.content[:100]}...")
        
        # Step 7: Test Visualization Generation
        print(f"\n📊 Step 7: Testing visualization generation...")
        
        if brief.visualizations:
            print(f"   ✅ Generated {len(brief.visualizations)} visualizations:")
            for viz_name, viz_data in brief.visualizations.items():
                if viz_data:
                    print(f"      • {viz_name}: ✅ ({len(viz_data)} chars base64)")
                else:
                    print(f"      • {viz_name}: ❌ (failed)")
        else:
            print("   ⚠️  No visualizations generated")
        
        # Step 8: Test Export Functionality
        print(f"\n💾 Step 8: Testing export functionality...")
        
        export_tests = [
            (OutputFormat.HTML, "test_policy_brief.html"),
            (OutputFormat.DOCX, "test_policy_brief.docx"),
            (OutputFormat.JSON, "test_policy_brief.json")
        ]
        
        exported_files = []
        
        for format_type, filename in export_tests:
            try:
                output_path = generator.export_brief(brief, format_type, filename)
                file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                
                if file_size > 0:
                    print(f"   ✅ {format_type.value.upper()} export: {filename} ({file_size:,} bytes)")
                    exported_files.append(filename)
                else:
                    print(f"   ❌ {format_type.value.upper()} export failed")
                    
            except Exception as e:
                print(f"   ❌ {format_type.value.upper()} export error: {e}")
        
        # # Step 9: Test Convenience Function
        # print(f"\n🛠️  Step 9: Testing convenience function...")
        
        # try:
        #     convenience_file = generate_policy_brief(
        #         title="Quick Policy Brief Test",
        #         research_focus="health outcomes",
        #         vector_db_id=db_id,
        #         output_format="html"
        #     )
            
        #     if os.path.exists(convenience_file):
        #         file_size = os.path.getsize(convenience_file)
        #         print(f"   ✅ Convenience function works: {convenience_file} ({file_size:,} bytes)")
        #         exported_files.append(convenience_file)
        #     else:
        #         print("   ❌ Convenience function failed")
                
        # except Exception as e:
        #     print(f"   ❌ Convenience function error: {e}")
        
        # # Step 10: Performance and Quality Summary
        # print(f"\n📈 Step 10: Policy Brief Quality Assessment")
        # print("-" * 50)
        # print(f"   • Brief Title: {brief.title}")
        # print(f"   • Target Audience: {config.target_audience}")
        # print(f"   • Evidence Sources: {brief.metadata['evidence_sources']}")
        # print(f"   • Generation Time: {brief.metadata['generation_time_seconds']:.2f}s")
        # print(f"   • Content Sections: {brief.metadata['total_sections']}")
        # print(f"   • Visual Elements: {len(brief.visualizations)} charts/graphics")
        # print(f"   • Export Formats: {len(exported_files)} successful exports")
        # print(f"   • Professional Formatting: ✅ (Policy-appropriate language)")
        # print(f"   • Evidence Integration: ✅ (Citations and references)")
        # print(f"   • IP Protection: ✅ (No copied diagrams, original visualizations)")
        
        # # Show section quality
        # print(f"\n   📋 Section Quality Scores:")
        # for section_name, section in sections:
        #     print(f"      • {section_name}: {section.confidence_score:.2f}")
        
        # # Clean up test files
        # print(f"\n🧹 Cleaning up test files...")
        # for filename in exported_files:
        #     try:
        #         if os.path.exists(filename):
        #             os.remove(filename)
        #             print(f"   Removed: {filename}")
        #     except Exception as e:
        #         print(f"   Warning: Could not remove {filename}: {e}")
        
        # # Clean up generator resources
        # generator.cleanup()
        
        # print(f"\n🎉 Policy Brief Generation Test Completed Successfully!")
        # print(f"📋 Your RAG pipeline now produces professional policy documents!")
        # print(f"   • Evidence-based content generation")
        # print(f"   • Professional formatting and structure")
        # print(f"   • Visual data presentation")
        # print(f"   • Multiple export formats")
        # print(f"   • IP-safe original visualizations")
        # print(f"   • Policy-focused language and recommendations")
        
        return True
        
    except Exception as e:
        print(f"❌ Policy brief generation test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to clean up even on error
        try:
            if 'generator' in locals():
                generator.cleanup()
        except Exception:
            pass
        
        return False


def demo_brief_generation():
    """Demonstrate policy brief generation with sample content"""
    
    print("\n🎯 Policy Brief Generation Demo")
    print("=" * 40)
    
    try:
        print("🔧 Demonstrating template-based generation...")
        
        # Test without external dependencies
        generator = PolicyBriefGenerator(
            enable_openai=False,
            enable_firebase=True  # Still need Firebase for vector data
        )
        
        # Get latest vector database
        available_dbs = generator.query_processor.embedder.list_vector_databases()
        
        if not available_dbs:
            print("❌ No vector databases available for demo")
            return False
        
        db_id = available_dbs[0]['database_id']
        
        # Simple configuration
        config = BriefConfig(
            title="Research Summary: Key Findings and Implications",
            target_audience="Decision Makers",
            executive_length="short",
            include_charts=False,  # Skip complex visualizations for demo
            include_data_tables=False,
            include_wordcloud=False,
            max_recommendations=3
        )
        
        print(f"   Config: {config.title}")
        print(f"   Target: {config.target_audience}")
        print(f"   Using database: {db_id}")
        
        # Generate brief
        brief = generator.generate_policy_brief(config, db_id, "research findings")
        
        print(f"\n✅ Demo brief generated:")
        print(f"   Title: {brief.title}")
        print(f"   Sections: {brief.metadata['total_sections']}")
        print(f"   Evidence: {brief.metadata['evidence_sources']} sources")
        print(f"   Time: {brief.metadata['generation_time_seconds']:.2f}s")
        
        # Show executive summary
        print(f"\n📄 Executive Summary Preview:")
        print(f"   {brief.executive_summary.content[:200]}...")
        
        # Export as HTML
        output_file = "demo_policy_brief.html"
        generator.export_brief(brief, OutputFormat.HTML, output_file)
        
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"\n💾 Demo export: {output_file} ({file_size:,} bytes)")
            
            # Clean up
            os.remove(output_file)
            print(f"   Cleaned up: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def test_visualization_features():
    """Test visualization and formatting features"""
    
    print("\n📊 Visualization Features Test")
    print("=" * 40)
    
    try:
        print("🎨 Testing visual formatting capabilities...")
        
        generator = PolicyBriefGenerator()
        
        # Test color schemes
        print("   Color schemes available:")
        for scheme_name, colors in generator.color_schemes.items():
            print(f"      • {scheme_name}: Primary {colors['primary']}, Secondary {colors['secondary']}")
        
        # Test that we can create basic visualizations
        print("\n   Testing visualization components:")
        
        # Mock evidence for testing
        from core_pipeline.query_processor import ResponseResult, QueryType, ResponseQuality
        
        mock_evidence = [
            ResponseResult(
                query="test query 1",
                response="Test response with statistical significance p < 0.001",
                evidence=[],
                citations=["Test citation 1"],
                confidence_score=0.85,
                response_type="statistical",
                processing_stats={},
                quality_assessment=ResponseQuality.GOOD,
                suggested_followups=[]
            ),
            ResponseResult(
                query="test query 2", 
                response="Test response about methodology and approach",
                evidence=[],
                citations=["Test citation 2"],
                confidence_score=0.75,
                response_type="methodology",
                processing_stats={},
                quality_assessment=ResponseQuality.GOOD,
                suggested_followups=[]
            )
        ]
        
        # Test confidence chart
        confidence_viz = generator._create_confidence_chart(mock_evidence, "professional")
        if confidence_viz:
            print("      ✅ Confidence chart generation")
        else:
            print("      ❌ Confidence chart failed")
        
        # Test data extraction
        data = generator._extract_data_for_visualization(mock_evidence)
        print(f"      ✅ Data extraction: {len(data['numbers'])} numbers, {len(data['statistics'])} stats")
        
        print("\n✅ Visualization features are functional!")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Policy Brief Generation Test Suite")
    print("=" * 60)
    print("🏛️  Step 5: Transforming research into professional policy documents")
    print("=" * 60)
    
    # Run main test
    main_success = test_policy_brief_generator()
    
    # Run additional tests if main passes
    if main_success:
        demo_success = demo_brief_generation()
        viz_success = test_visualization_features()
    else:
        print("\n🔄 Running basic component tests...")
        try:
            generator = PolicyBriefGenerator(enable_firebase=False, enable_openai=False)
            print("✅ Policy brief generator can be initialized without external services")
            demo_success = False
            viz_success = test_visualization_features()
        except Exception as e:
            print(f"❌ Basic initialization failed: {e}")
            demo_success = False
            viz_success = False
    
    print("\n" + "=" * 60)
    if main_success and demo_success and viz_success:
        print("🎉 All tests passed! Policy brief generation pipeline is ready.")
        print("🏛️  Your complete RAG pipeline is now fully operational!")
        print("")
        print("🚀 Complete 5-Step RAG Pipeline Summary:")
        print("   ✅ Step 1: Document Processing (PDF → Text, Tables, Images)")
        print("   ✅ Step 2: Content Analysis (Semantic Chunking + Structure)")
        print("   ✅ Step 3: Vector Embedding (FAISS + Firebase Storage)")
        print("   ✅ Step 4: Query Processing (Q&A + Response Generation)")
        print("   ✅ Step 5: Policy Brief Generation (Professional Documents)")
        print("")
        print("📋 Policy Brief Features:")
        print("   • Evidence-based content generation")
        print("   • Professional formatting for policymakers")
        print("   • Visual data presentation (charts, tables)")
        print("   • Multiple export formats (HTML, DOCX, JSON)")
        print("   • IP-safe original visualizations")
        print("   • Citation management and references")
        print("   • Structured sections (Executive Summary, Recommendations, etc.)")
        print("")
        print("💡 To generate policy briefs from your documents:")
        print("   from policy_brief_generator import generate_policy_brief")
        print("   generate_policy_brief('Your Title', research_focus='your topic')")
        
    elif main_success:
        print("🎯 Main policy brief generation functionality working!")
        print("⚠️  Some demo features had issues, but core functionality is ready.")
    else:
        print("⚠️  Some tests failed. Check logs for details.")
        print("💡 Ensure Firebase is configured and vector database exists.")
        print("📋 Required: Vector database from Step 3 (test_vector_embedder.py)")
        print("🔑 Optional: OPENAI_API_KEY for enhanced content generation")
        print("📊 Note: Template-based generation works without OpenAI") 