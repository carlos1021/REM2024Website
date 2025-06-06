"""
Query Processing & Response Generation Engine - Step 4 of RAG Pipeline

This module handles intelligent query processing, retrieval, and response generation using
the vector database created in Step 3. It provides sophisticated query understanding,
multi-modal retrieval, and LLM-powered response synthesis.

Key Features:
- Intelligent query understanding and expansion
- Hybrid search with semantic and keyword matching
- Multi-modal content retrieval (text, tables, figures)
- Context assembly with relevance ranking
- LLM-powered response generation with citations
- Response quality validation and improvement
"""

import os
import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import openai
import firebase_admin
from firebase_admin import credentials, storage, db

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import from our pipeline components
from .vector_embedder import VectorEmbedder, VectorDatabase, VectorSearchResult, load_vector_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of user queries"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    SUMMARIZATION = "summarization"
    METHODOLOGY = "methodology"
    STATISTICAL = "statistical"
    GENERAL = "general"


class ResponseQuality(Enum):
    """Response quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"


@dataclass
class QueryContext:
    """Container for processed query information"""
    original_query: str
    expanded_query: str
    query_type: QueryType
    key_concepts: List[str]
    intent_confidence: float
    requires_tables: bool
    requires_figures: bool
    requires_statistics: bool


@dataclass
class RetrievedEvidence:
    """Container for retrieved evidence with metadata"""
    content: str
    source_info: Dict[str, str]
    relevance_score: float
    content_type: str  # text, table, figure_caption
    section: str
    keywords: List[str]
    related_content: Dict[str, List[str]]
    citation_info: Dict[str, str]


@dataclass
class ResponseResult:
    """Container for complete response with metadata"""
    query: str
    response: str
    evidence: List[RetrievedEvidence]
    citations: List[str]
    confidence_score: float
    response_type: str
    processing_stats: Dict[str, Any]
    quality_assessment: ResponseQuality
    suggested_followups: List[str]


class QueryProcessor:
    """
    Advanced query processing and response generation engine.
    """
    
    def __init__(self,
                 firebase_bucket: str = "rem2024-f429b.appspot.com",
                 firebase_database_url: str = "https://rem2024-f429b-default-rtdb.firebaseio.com",
                 enable_firebase: bool = True,
                 enable_openai: bool = True,
                 max_context_length: int = 8000,
                 response_max_tokens: int = 1000):
        """
        Initialize the query processor.
        
        Args:
            firebase_bucket: Firebase storage bucket name
            firebase_database_url: Firebase Realtime Database URL
            enable_firebase: Whether to use Firebase for data access
            enable_openai: Whether to use OpenAI for response generation
            max_context_length: Maximum context length for LLM
            response_max_tokens: Maximum tokens for response generation
        """
        self.firebase_bucket = firebase_bucket
        self.firebase_database_url = firebase_database_url
        self.enable_firebase = enable_firebase
        self.enable_openai = enable_openai
        self.max_context_length = max_context_length
        self.response_max_tokens = response_max_tokens
        
        # Initialize OpenAI
        if self.enable_openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                logger.warning("OPENAI_API_KEY not found - response generation will be limited")
                self.enable_openai = False
        
        # Initialize Firebase
        if self.enable_firebase:
            self._init_firebase()
        
        # Initialize vector embedder for retrieval
        self.embedder = VectorEmbedder()
        
        # Query understanding patterns
        self.query_patterns = {
            QueryType.FACTUAL: [
                r'\b(what|who|when|where|which|how many)\b',
                r'\bis\b.*\?',
                r'\bdoes\b.*\?'
            ],
            QueryType.ANALYTICAL: [
                r'\b(why|how|explain|analyze|evaluate)\b',
                r'\b(relationship|correlation|impact|effect)\b',
                r'\b(cause|reason|factor)\b'
            ],
            QueryType.COMPARATIVE: [
                r'\b(compare|contrast|difference|versus|vs)\b',
                r'\b(better|worse|more|less)\b.*\bthan\b',
                r'\b(similar|different)\b'
            ],
            QueryType.SUMMARIZATION: [
                r'\b(summarize|overview|summary|brief)\b',
                r'\b(main|key|primary)\b.*\b(points|findings|results)\b'
            ],
            QueryType.METHODOLOGY: [
                r'\b(method|methodology|approach|procedure)\b',
                r'\b(how.*conducted|how.*measured|how.*analyzed)\b',
                r'\b(study design|data collection|analysis)\b'
            ],
            QueryType.STATISTICAL: [
                r'\b(statistic|percentage|correlation|significance)\b',
                r'\b(p.?value|confidence|sample size)\b',
                r'\b(mean|median|standard deviation)\b'
            ]
        }
        
        logger.info("Query processor initialized")
    
    def _init_firebase(self):
        """Initialize Firebase connection"""
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
            logger.info("Firebase services ready")
        except Exception as e:
            logger.warning(f"Failed to get Firebase services: {e}")
            self.enable_firebase = False
    
    def process_query(self, query: str, vector_db_id: str, 
                     max_results: int = 10, use_hybrid: bool = True) -> ResponseResult:
        """
        Process a user query and generate a comprehensive response.
        
        Args:
            query: User's natural language question
            vector_db_id: Vector database identifier to search
            max_results: Maximum number of results to retrieve
            use_hybrid: Whether to use hybrid search
            
        Returns:
            ResponseResult containing answer, evidence, and metadata
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing query: '{query[:100]}...'")
            
            # Step 1: Analyze and understand the query
            query_context = self._analyze_query(query)
            
            # Step 2: Retrieve relevant content from vector database
            evidence = self._retrieve_evidence(query_context, vector_db_id, max_results, use_hybrid)
            
            # Step 3: Generate response using LLM
            response_text = self._generate_response(query_context, evidence)
            
            # Step 4: Extract citations and assess quality
            citations = self._extract_citations(evidence)
            quality = self._assess_response_quality(response_text, evidence)
            
            # Step 5: Generate follow-up suggestions
            followups = self._generate_followups(query_context, evidence)
            
            # Calculate processing statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            stats = {
                'processing_time_seconds': processing_time,
                'evidence_retrieved': len(evidence),
                'query_type': query_context.query_type.value,
                'intent_confidence': query_context.intent_confidence,
                'response_tokens': len(response_text.split()),
                'citations_found': len(citations)
            }
            
            result = ResponseResult(
                query=query,
                response=response_text,
                evidence=evidence,
                citations=citations,
                confidence_score=self._calculate_confidence(query_context, evidence, quality),
                response_type=query_context.query_type.value,
                processing_stats=stats,
                quality_assessment=quality,
                suggested_followups=followups
            )
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Failed to process query: {str(e)}"
            logger.error(error_msg)
            
            # Return error response
            return ResponseResult(
                query=query,
                response=f"I apologize, but I encountered an error processing your question: {error_msg}",
                evidence=[],
                citations=[],
                confidence_score=0.0,
                response_type="error",
                processing_stats={'error': error_msg},
                quality_assessment=ResponseQuality.POOR,
                suggested_followups=[]
            )
    
    def _analyze_query(self, query: str) -> QueryContext:
        """Analyze query to understand intent and requirements"""
        try:
            # Determine query type
            query_type = QueryType.GENERAL
            max_confidence = 0.0
            
            for qtype, patterns in self.query_patterns.items():
                confidence = 0.0
                for pattern in patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        confidence += 1.0
                
                # Normalize by number of patterns
                confidence = confidence / len(patterns)
                
                if confidence > max_confidence:
                    max_confidence = confidence
                    query_type = qtype
            
            # Extract key concepts (simple keyword extraction)
            key_concepts = self._extract_key_concepts(query)
            
            # Determine content type requirements
            requires_tables = bool(re.search(r'\b(table|data|statistic|number|result)\b', query, re.IGNORECASE))
            requires_figures = bool(re.search(r'\b(figure|chart|graph|image|visual)\b', query, re.IGNORECASE))
            requires_statistics = bool(re.search(r'\b(percentage|correlation|p.?value|significant)\b', query, re.IGNORECASE))
            
            # Expand query with synonyms and related terms
            expanded_query = self._expand_query(query, key_concepts)
            
            return QueryContext(
                original_query=query,
                expanded_query=expanded_query,
                query_type=query_type,
                key_concepts=key_concepts,
                intent_confidence=max_confidence,
                requires_tables=requires_tables,
                requires_figures=requires_figures,
                requires_statistics=requires_statistics
            )
            
        except Exception as e:
            logger.warning(f"Query analysis failed: {e}")
            return QueryContext(
                original_query=query,
                expanded_query=query,
                query_type=QueryType.GENERAL,
                key_concepts=[],
                intent_confidence=0.5,
                requires_tables=False,
                requires_figures=False,
                requires_statistics=False
            )
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query"""
        # Simple keyword extraction (could be enhanced with NLP)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                     'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'was', 'were', 'do', 'does'}
        
        # Remove punctuation and split
        clean_query = re.sub(r'[^\w\s]', ' ', query.lower())
        words = clean_query.split()
        
        # Filter stop words and short words
        concepts = [word for word in words if word not in stop_words and len(word) > 2]
        
        return concepts[:10]  # Limit to top 10 concepts
    
    def _expand_query(self, query: str, key_concepts: List[str]) -> str:
        """Expand query with synonyms and related terms"""
        # Simple query expansion (could be enhanced with word embeddings)
        expansions = {
            'health': ['wellbeing', 'medical', 'clinical'],
            'children': ['kids', 'youth', 'adolescents', 'pediatric'],
            'study': ['research', 'investigation', 'analysis'],
            'results': ['findings', 'outcomes', 'conclusions'],
            'effect': ['impact', 'influence', 'consequence'],
            'method': ['approach', 'technique', 'methodology']
        }
        
        expanded_terms = []
        for concept in key_concepts:
            if concept in expansions:
                expanded_terms.extend(expansions[concept])
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms[:5])}"  # Add up to 5 expansion terms
        
        return query
    
    def _retrieve_evidence(self, query_context: QueryContext, vector_db_id: str, 
                          max_results: int, use_hybrid: bool) -> List[RetrievedEvidence]:
        """Retrieve relevant evidence from vector database"""
        try:
            # Load vector database
            vector_db = load_vector_database(vector_db_id)
            
            # Perform search
            if use_hybrid and vector_db.tfidf_vectorizer is not None:
                search_results = self.embedder.hybrid_search(
                    vector_db, query_context.expanded_query, top_k=max_results
                )
            else:
                search_results = self.embedder.semantic_search(
                    vector_db, query_context.expanded_query, top_k=max_results
                )
            
            # Convert search results to evidence objects
            evidence = []
            for result in search_results:
                evidence_obj = RetrievedEvidence(
                    content=result.content,
                    source_info={
                        'document_key': result.context_info.get('document_key', ''),
                        'chunk_id': result.chunk_id,
                        'section': result.context_info.get('section', ''),
                        'chunk_type': result.context_info.get('chunk_type', '')
                    },
                    relevance_score=result.similarity_score,
                    content_type=result.context_info.get('chunk_type', 'text'),
                    section=result.context_info.get('section', 'Unknown'),
                    keywords=result.related_content.get('keywords', []) if result.related_content else [],
                    related_content=result.related_content or {},
                    citation_info=self._create_citation_info(result)
                )
                evidence.append(evidence_obj)
            
            # Filter evidence based on query requirements
            filtered_evidence = self._filter_evidence(evidence, query_context)
            
            logger.info(f"Retrieved {len(filtered_evidence)} relevant evidence pieces")
            return filtered_evidence
            
        except Exception as e:
            logger.error(f"Evidence retrieval failed: {e}")
            return []
    
    def _filter_evidence(self, evidence: List[RetrievedEvidence], 
                        query_context: QueryContext) -> List[RetrievedEvidence]:
        """Filter and prioritize evidence based on query requirements"""
        filtered = []
        
        for item in evidence:
            # Priority boost for required content types
            priority_boost = 0.0
            
            if query_context.requires_tables and item.content_type == 'table':
                priority_boost += 0.2
            if query_context.requires_statistics and any(
                stat_word in item.content.lower() 
                for stat_word in ['percentage', 'correlation', 'significant', 'p <', 'p=']
            ):
                priority_boost += 0.15
            
            # Adjust relevance score
            item.relevance_score += priority_boost
            filtered.append(item)
        
        # Sort by relevance score
        filtered.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return filtered
    
    def _create_citation_info(self, result: VectorSearchResult) -> Dict[str, str]:
        """Create citation information from search result"""
        return {
            'source_id': result.context_info.get('document_key', '')[:12],
            'section': result.context_info.get('section', 'Unknown'),
            'chunk_type': result.context_info.get('chunk_type', 'text'),
            'relevance': f"{result.similarity_score:.3f}"
        }
    
    def _generate_response(self, query_context: QueryContext, 
                          evidence: List[RetrievedEvidence]) -> str:
        """Generate response using LLM with retrieved evidence"""
        if not self.enable_openai or not evidence:
            return self._generate_fallback_response(query_context, evidence)
        
        try:
            # Prepare context from evidence
            context_parts = []
            for i, item in enumerate(evidence[:5]):  # Use top 5 pieces of evidence
                context_parts.append(
                    f"Evidence {i+1} (from {item.section}):\n{item.content[:500]}..."
                )
            
            context = "\n\n".join(context_parts)
            
            # Limit context length
            if len(context) > self.max_context_length:
                context = context[:self.max_context_length] + "..."
            
            # Create response prompt based on query type
            system_prompt = self._get_system_prompt(query_context.query_type)
            
            user_prompt = f"""
Question: {query_context.original_query}

Context from research documents:
{context}

Please provide a comprehensive answer based on the evidence provided. Include specific details and cite relevant information. If the evidence doesn't fully answer the question, acknowledge the limitations.
"""
            
            # Generate response
            response = openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.response_max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return self._generate_fallback_response(query_context, evidence)
    
    def _get_system_prompt(self, query_type: QueryType) -> str:
        """Get system prompt based on query type"""
        prompts = {
            QueryType.FACTUAL: "You are a research assistant providing factual answers based on academic research. Be precise and cite specific evidence.",
            QueryType.ANALYTICAL: "You are a research analyst explaining complex relationships and causalities found in academic research. Provide detailed explanations.",
            QueryType.COMPARATIVE: "You are a research analyst comparing different findings, approaches, or results. Highlight similarities, differences, and relative strengths.",
            QueryType.SUMMARIZATION: "You are a research summarizer creating concise overviews of key findings and insights from academic research.",
            QueryType.METHODOLOGY: "You are a research methodology expert explaining how studies were conducted, what methods were used, and their implications.",
            QueryType.STATISTICAL: "You are a statistical analyst interpreting quantitative findings, significance levels, and statistical relationships in research.",
            QueryType.GENERAL: "You are a helpful research assistant providing clear, accurate answers based on academic research evidence."
        }
        
        return prompts.get(query_type, prompts[QueryType.GENERAL])
    
    def _generate_fallback_response(self, query_context: QueryContext, 
                                   evidence: List[RetrievedEvidence]) -> str:
        """Generate fallback response when LLM is not available"""
        if not evidence:
            return "I couldn't find relevant information in the available documents to answer your question."
        
        # Simple template-based response
        response_parts = [
            f"Based on the available research documents, here's what I found regarding your question about {' '.join(query_context.key_concepts[:3])}:",
            ""
        ]
        
        for i, item in enumerate(evidence[:3]):
            response_parts.append(f"{i+1}. From {item.section}: {item.content[:200]}...")
        
        response_parts.append(
            "\nThis information is extracted from the research documents. For more detailed analysis, please refer to the original sources."
        )
        
        return "\n".join(response_parts)
    
    def _extract_citations(self, evidence: List[RetrievedEvidence]) -> List[str]:
        """Extract citation information from evidence"""
        citations = []
        
        for i, item in enumerate(evidence[:5]):
            citation = f"[{i+1}] {item.source_info['document_key'][:12]}... - {item.section} (Relevance: {item.relevance_score:.3f})"
            citations.append(citation)
        
        return citations
    
    def _assess_response_quality(self, response: str, evidence: List[RetrievedEvidence]) -> ResponseQuality:
        """Assess the quality of the generated response"""
        # Simple heuristic-based quality assessment
        score = 0
        
        # Length check
        if len(response) > 100:
            score += 1
        if len(response) > 300:
            score += 1
        
        # Evidence utilization
        if len(evidence) >= 3:
            score += 1
        if len(evidence) >= 5:
            score += 1
        
        # Content quality indicators
        quality_indicators = ['research', 'study', 'findings', 'evidence', 'analysis']
        for indicator in quality_indicators:
            if indicator in response.lower():
                score += 0.5
        
        # Map score to quality level
        if score >= 4:
            return ResponseQuality.EXCELLENT
        elif score >= 3:
            return ResponseQuality.GOOD
        elif score >= 2:
            return ResponseQuality.ADEQUATE
        else:
            return ResponseQuality.POOR
    
    def _calculate_confidence(self, query_context: QueryContext, 
                             evidence: List[RetrievedEvidence], 
                             quality: ResponseQuality) -> float:
        """Calculate overall confidence score for the response"""
        # Base confidence from query understanding
        confidence = query_context.intent_confidence * 0.3
        
        # Evidence quality contribution
        if evidence:
            avg_relevance = sum(item.relevance_score for item in evidence) / len(evidence)
            confidence += avg_relevance * 0.4
        
        # Response quality contribution
        quality_scores = {
            ResponseQuality.EXCELLENT: 1.0,
            ResponseQuality.GOOD: 0.8,
            ResponseQuality.ADEQUATE: 0.6,
            ResponseQuality.POOR: 0.3
        }
        confidence += quality_scores[quality] * 0.3
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _generate_followups(self, query_context: QueryContext, 
                           evidence: List[RetrievedEvidence]) -> List[str]:
        """Generate follow-up question suggestions"""
        followups = []
        
        # Generic follow-ups based on query type
        if query_context.query_type == QueryType.FACTUAL:
            followups.append("How was this information determined or measured?")
            followups.append("What are the implications of these findings?")
        elif query_context.query_type == QueryType.ANALYTICAL:
            followups.append("What other factors might influence these results?")
            followups.append("How do these findings compare to other studies?")
        elif query_context.query_type == QueryType.METHODOLOGY:
            followups.append("What are the limitations of this approach?")
            followups.append("How could this methodology be improved?")
        
        # Evidence-based follow-ups
        if evidence:
            unique_sections = set(item.section for item in evidence[:3])
            for section in unique_sections:
                if section != "Unknown":
                    followups.append(f"What else does the research say about {section.lower()}?")
        
        return followups[:3]  # Limit to 3 suggestions


# Utility functions for easy usage
def ask_question(question: str, vector_db_id: str = None, 
                use_hybrid: bool = True, max_results: int = 5) -> ResponseResult:
    """
    Convenience function to ask a question about the documents.
    
    Args:
        question: User's question
        vector_db_id: Vector database ID (uses latest if None)
        use_hybrid: Whether to use hybrid search
        max_results: Maximum results to retrieve
        
    Returns:
        ResponseResult with answer and evidence
    """
    processor = QueryProcessor()
    
    # Get latest vector database if none specified
    if vector_db_id is None:
        available_dbs = processor.embedder.list_vector_databases()
        if not available_dbs:
            raise ValueError("No vector databases found")
        vector_db_id = available_dbs[0]['database_id']
    
    return processor.process_query(question, vector_db_id, max_results, use_hybrid)


def interactive_query_session(vector_db_id: str = None):
    """
    Start an interactive query session.
    
    Args:
        vector_db_id: Vector database ID (uses latest if None)
    """
    processor = QueryProcessor()
    
    # Get latest vector database if none specified
    if vector_db_id is None:
        available_dbs = processor.embedder.list_vector_databases()
        if not available_dbs:
            print("No vector databases found. Please create one first.")
            return
        vector_db_id = available_dbs[0]['database_id']
        print(f"Using vector database: {vector_db_id}")
    
    print("Interactive Query Session Started!")
    print("Type 'quit' to exit, 'help' for commands\n")
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif question.lower() == 'help':
                print("Commands:")
                print("  - Ask any question about the research documents")
                print("  - Type 'quit' to exit")
                print("  - The system will provide answers with citations\n")
                continue
            elif not question:
                continue
            
            print("\nProcessing your question...")
            result = processor.process_query(question, vector_db_id)
            
            print(f"\n📝 Answer:")
            print(result.response)
            
            print(f"\n📊 Confidence: {result.confidence_score:.2f}")
            print(f"Quality: {result.quality_assessment.value}")
            
            if result.citations:
                print(f"\n📚 Sources:")
                for citation in result.citations:
                    print(f"  {citation}")
            
            if result.suggested_followups:
                print(f"\n💡 Follow-up suggestions:")
                for followup in result.suggested_followups:
                    print(f"  • {followup}")
            
            print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.") 