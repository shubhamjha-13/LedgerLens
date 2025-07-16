#!/usr/bin/env python3
"""
Complete Memory-Efficient FinancialLightRAG with Advanced Chunking
- Advanced semantic chunking with safety measures
- ChromaDB for vectors, SQLite for graph
- Sophisticated financial knowledge extraction
- Memory optimization and monitoring
"""

import asyncio
import json
import re
import os
import gc
import psutil
import sqlite3
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union, Literal, Generator
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import networkx as nx
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

# ChromaDB imports
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Memory Management Utilities
# =============================================================================

def get_memory_usage():
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / 1024 / 1024

def cleanup_memory():
    """Force garbage collection"""
    gc.collect()

class MemoryMonitor:
    """Advanced memory monitor with adaptive cleanup"""
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_mb = max_memory_mb
        self.start_memory = get_memory_usage()
        self.cleanup_count = 0
        
    def check_and_cleanup(self) -> Dict[str, float]:
        """Check memory and cleanup if needed"""
        current = get_memory_usage()
        
        if current > self.max_memory_mb * 0.8:  # Start cleanup at 80%
            logger.warning(f"Memory high: {current:.1f}MB, cleaning up...")
            cleanup_memory()
            self.cleanup_count += 1
            current = get_memory_usage()
            
            if current > self.max_memory_mb:
                logger.error(f"Memory still high after cleanup: {current:.1f}MB")
        
        return {
            "current_mb": current,
            "usage_percent": (current / self.max_memory_mb) * 100,
            "cleanup_count": self.cleanup_count
        }

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class FinancialEntity:
    """Financial entity with domain-specific attributes"""
    id: str
    name: str
    type: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    source_chunks: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        content = f"{self.name}_{self.type}_{self.description[:30]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

@dataclass
class FinancialRelationship:
    """Financial relationship between entities"""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    description: str
    keywords: str = ""
    weight: float = 1.0
    context: str = ""
    confidence: float = 1.0
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        content = f"{self.source_id}_{self.target_id}_{self.relation_type}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

@dataclass
class TextChunk:
    """Text chunk with comprehensive metadata"""
    id: str
    content: str
    doc_id: str
    chunk_index: int
    tokens: int
    entities: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    section_title: Optional[str] = None
    section_index: Optional[int] = None

@dataclass
class QueryParam:
    """Query parameters with advanced options"""
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "hybrid"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    top_k: int = 20
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000

# =============================================================================
# Advanced Chunking System
# =============================================================================

class AdvancedFinancialChunker:
    """Advanced chunker with financial document understanding"""
    
    def __init__(self, 
                 base_chunk_size: int = 1200,
                 overlap_ratio: float = 0.15,
                 max_chunks_per_doc: int = 1000,
                 min_chunk_size: int = 100):
        
        self.base_chunk_size = base_chunk_size
        self.overlap_ratio = overlap_ratio
        self.max_chunks_per_doc = max_chunks_per_doc
        self.min_chunk_size = min_chunk_size
        
        # Financial document structure patterns
        self.section_patterns = {
            'executive_summary': r'(?i)\b(executive\s+summary|management\s+summary|key\s+highlights)\b',
            'business_overview': r'(?i)\b(business\s+overview|company\s+overview|business\s+description)\b',
            'financial_performance': r'(?i)\b(financial\s+performance|financial\s+results|financial\s+highlights)\b',
            'revenue_analysis': r'(?i)\b(revenue|net\s+sales|total\s+revenue|sales\s+performance)\b',
            'profitability': r'(?i)\b(profit|earnings|income|ebitda|operating\s+income)\b',
            'balance_sheet': r'(?i)\b(balance\s+sheet|statement\s+of\s+financial\s+position|assets\s+and\s+liabilities)\b',
            'cash_flow': r'(?i)\b(cash\s+flow|statement\s+of\s+cash\s+flows|cash\s+position)\b',
            'risk_factors': r'(?i)\b(risk\s+factors|principal\s+risks|risk\s+management|risk\s+assessment)\b',
            'market_analysis': r'(?i)\b(market\s+analysis|industry\s+analysis|competitive\s+landscape)\b',
            'outlook': r'(?i)\b(outlook|guidance|forecast|future\s+prospects|forward\s+looking)\b',
            'segment_analysis': r'(?i)\b(segment|division|business\s+unit|product\s+line)\b'
        }
        
        # Financial importance weights for different sections
        self.section_importance = {
            'financial_performance': 1.0,
            'revenue_analysis': 0.9,
            'profitability': 0.9,
            'balance_sheet': 0.8,
            'cash_flow': 0.8,
            'risk_factors': 0.7,
            'executive_summary': 0.8,
            'business_overview': 0.6,
            'market_analysis': 0.6,
            'outlook': 0.7,
            'segment_analysis': 0.6
        }
    
    def create_chunks(self, text: str, doc_id: str) -> List[TextChunk]:
        """Create sophisticated financial document chunks"""
        
        if not text or len(text.strip()) < self.min_chunk_size:
            logger.warning(f"Text too short for chunking: {len(text)} chars")
            return []
        
        logger.info(f"Advanced chunking for {doc_id}: {len(text):,} characters")
        
        # Step 1: Detect document structure
        structure = self._analyze_document_structure(text)
        logger.info(f"Detected structure: {len(structure['sections'])} sections")
        
        # Step 2: Create adaptive chunks based on content type
        chunks = self._create_adaptive_chunks(text, doc_id, structure)
        
        # Step 3: Post-process and validate chunks
        validated_chunks = self._validate_and_enhance_chunks(chunks, doc_id)
        
        logger.info(f"Created {len(validated_chunks)} chunks for {doc_id}")
        return validated_chunks
    
    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure and identify financial sections"""
        
        sections = []
        section_scores = {}
        
        # Find all section markers
        all_matches = []
        for section_type, pattern in self.section_patterns.items():
            for match in re.finditer(pattern, text):
                all_matches.append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group().strip(),
                    'type': section_type,
                    'importance': self.section_importance.get(section_type, 0.5)
                })
        
        # Sort by position
        all_matches.sort(key=lambda x: x['start'])
        
        # Create sections with boundaries
        if all_matches:
            for i, match in enumerate(all_matches):
                section_start = match['start']
                
                # Find section end (next section or document end)
                if i + 1 < len(all_matches):
                    section_end = all_matches[i + 1]['start']
                else:
                    section_end = len(text)
                
                # Only include substantial sections
                section_length = section_end - section_start
                if section_length >= self.min_chunk_size:
                    sections.append({
                        'start': section_start,
                        'end': section_end,
                        'title': match['text'],
                        'type': match['type'],
                        'importance': match['importance'],
                        'length': section_length
                    })
                    section_scores[match['type']] = match['importance']
        
        # If no clear sections, create artificial ones based on content density
        if not sections:
            sections = self._create_artificial_sections(text)
        
        return {
            'sections': sections,
            'section_scores': section_scores,
            'total_length': len(text),
            'avg_section_length': sum(s['length'] for s in sections) / len(sections) if sections else 0
        }
    
    def _create_artificial_sections(self, text: str) -> List[Dict[str, Any]]:
        """Create artificial sections when no clear structure is found"""
        
        # Use paragraph breaks as section boundaries
        paragraphs = re.split(r'\n\s*\n', text)
        sections = []
        current_pos = 0
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) >= self.min_chunk_size:
                sections.append({
                    'start': current_pos,
                    'end': current_pos + len(paragraph),
                    'title': f"Section {i+1}",
                    'type': 'general',
                    'importance': 0.5,
                    'length': len(paragraph)
                })
            current_pos += len(paragraph) + 2  # Account for paragraph breaks
        
        return sections
    
    def _create_adaptive_chunks(self, text: str, doc_id: str, structure: Dict[str, Any]) -> List[TextChunk]:
        """Create chunks with adaptive sizing based on content importance"""
        
        chunks = []
        chunk_index = 0
        
        with tqdm(structure['sections'], desc="Creating adaptive chunks", leave=False) as section_pbar:
            
            for section in structure['sections']:
                section_text = text[section['start']:section['end']].strip()
                
                if len(section_text) < self.min_chunk_size:
                    section_pbar.update(1)
                    continue
                
                # Adaptive chunk size based on importance
                importance = section['importance']
                adaptive_chunk_size = int(self.base_chunk_size * (0.7 + 0.6 * importance))
                adaptive_overlap = int(adaptive_chunk_size * self.overlap_ratio)
                
                # Create chunks for this section
                section_chunks = self._split_section_smartly(
                    section_text, 
                    adaptive_chunk_size, 
                    adaptive_overlap,
                    section['type']
                )
                
                # Convert to TextChunk objects
                for chunk_text in section_chunks:
                    chunk = TextChunk(
                        id=f"{doc_id}_chunk_{chunk_index}",
                        content=chunk_text,
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        tokens=len(chunk_text.split()),
                        section_title=section['title'],
                        section_index=len(chunks),
                        metadata={
                            'section_type': section['type'],
                            'section_importance': section['importance'],
                            'adaptive_chunk_size': adaptive_chunk_size,
                            'section_start': section['start'],
                            'section_end': section['end'],
                            'created_at': datetime.now().isoformat(),
                            'chunk_chars': len(chunk_text)
                        }
                    )
                    
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Safety check
                    if len(chunks) >= self.max_chunks_per_doc:
                        logger.warning(f"Reached max chunks limit for {doc_id}")
                        section_pbar.close()
                        return chunks
                
                section_pbar.update(1)
                section_pbar.set_postfix({
                    "Chunks": len(chunks),
                    "Type": section['type'][:8],
                    "Importance": f"{importance:.2f}"
                })
        
        return chunks
    
    def _split_section_smartly(self, text: str, chunk_size: int, overlap: int, section_type: str) -> List[str]:
        """Smart section splitting with financial content awareness"""
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        iteration_count = 0
        max_iterations = len(text) // 50 + 100
        
        # Financial content markers for better splitting
        financial_markers = [
            r'\$[\d,.]+(?: million| billion| thousand)?',  # Dollar amounts
            r'\d+\.?\d*%',  # Percentages
            r'(?:Q[1-4]|quarter|fiscal year|FY)\s+\d{4}',  # Time periods
            r'(?:revenue|profit|loss|earnings|sales|income|ebitda)',  # Financial terms
        ]
        
        while start < len(text) and iteration_count < max_iterations:
            iteration_count += 1
            
            end = min(start + chunk_size, len(text))
            
            # Smart boundary detection based on content type
            if end < len(text):
                end = self._find_smart_boundary(text, start, end, section_type, financial_markers)
            
            # Extract and validate chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text and len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            elif chunk_text and len(chunks) == 0:
                # Keep short chunks if they're the only content
                chunks.append(chunk_text)
            
            # Calculate next start position
            new_start = end - overlap
            
            # Ensure progress to prevent infinite loops
            if new_start <= start:
                new_start = start + max(self.min_chunk_size, chunk_size // 4)
            
            start = new_start
            
            # Safety break
            if iteration_count >= max_iterations:
                logger.warning(f"Iteration limit reached for section type: {section_type}")
                break
        
        return chunks
    
    def _find_smart_boundary(self, text: str, start: int, end: int, section_type: str, financial_markers: List[str]) -> int:
        """Find intelligent chunk boundaries based on content type"""
        
        # Look for natural boundaries in a window around the target end
        search_window = min(200, (end - start) // 3)
        search_start = max(start, end - search_window)
        search_end = min(len(text), end + search_window // 2)
        search_text = text[search_start:search_end]
        
        # Priority 1: Financial statement boundaries (for financial sections)
        if section_type in ['financial_performance', 'revenue_analysis', 'profitability']:
            financial_boundaries = []
            for pattern in financial_markers:
                for match in re.finditer(pattern, search_text):
                    boundary_pos = search_start + match.end()
                    if boundary_pos > start + self.min_chunk_size:
                        financial_boundaries.append(boundary_pos)
            
            if financial_boundaries:
                # Use the boundary closest to target end
                closest_boundary = min(financial_boundaries, key=lambda x: abs(x - end))
                if abs(closest_boundary - end) < search_window:
                    return closest_boundary
        
        # Priority 2: Sentence boundaries
        sentence_pattern = r'[.!?]\s+'
        sentence_matches = []
        for match in re.finditer(sentence_pattern, search_text):
            boundary_pos = search_start + match.end()
            if boundary_pos > start + self.min_chunk_size:
                sentence_matches.append(boundary_pos)
        
        if sentence_matches:
            # Prefer boundaries closer to target end
            best_boundary = min(sentence_matches, key=lambda x: abs(x - end))
            return best_boundary
        
        # Priority 3: Paragraph boundaries
        paragraph_pattern = r'\n\s*\n'
        for match in re.finditer(paragraph_pattern, search_text):
            boundary_pos = search_start + match.start()
            if boundary_pos > start + self.min_chunk_size:
                return boundary_pos
        
        # Fallback: Use original end
        return end
    
    def _validate_and_enhance_chunks(self, chunks: List[TextChunk], doc_id: str) -> List[TextChunk]:
        """Validate and enhance chunks with additional metadata"""
        
        validated_chunks = []
        
        for chunk in chunks:
            # Skip empty or too-short chunks
            if not chunk.content or len(chunk.content.strip()) < self.min_chunk_size:
                continue
            
            # Enhance metadata with content analysis
            content_analysis = self._analyze_chunk_content(chunk.content)
            chunk.metadata.update(content_analysis)
            
            # Add financial relevance score
            chunk.metadata['financial_relevance'] = self._calculate_financial_relevance(chunk.content)
            
            validated_chunks.append(chunk)
        
        # Re-index chunks
        for i, chunk in enumerate(validated_chunks):
            chunk.chunk_index = i
            chunk.id = f"{doc_id}_chunk_{i}"
        
        return validated_chunks
    
    def _analyze_chunk_content(self, content: str) -> Dict[str, Any]:
        """Analyze chunk content for financial indicators"""
        
        analysis = {
            'has_numbers': bool(re.search(r'\d+', content)),
            'has_percentages': bool(re.search(r'\d+\.?\d*%', content)),
            'has_currency': bool(re.search(r'\$[\d,]+', content)),
            'has_dates': bool(re.search(r'\b\d{4}\b|\b(?:Q[1-4]|quarter)\b', content)),
            'sentence_count': len(re.findall(r'[.!?]+', content)),
            'word_count': len(content.split()),
            'avg_sentence_length': 0
        }
        
        if analysis['sentence_count'] > 0:
            analysis['avg_sentence_length'] = analysis['word_count'] / analysis['sentence_count']
        
        return analysis
    
    def _calculate_financial_relevance(self, content: str) -> float:
        """Calculate financial relevance score for content"""
        
        financial_terms = [
            'revenue', 'profit', 'loss', 'earnings', 'sales', 'income', 'ebitda',
            'margin', 'cash', 'debt', 'equity', 'assets', 'liabilities',
            'growth', 'decline', 'increase', 'decrease', 'performance',
            'quarter', 'annual', 'fiscal', 'financial', 'business'
        ]
        
        content_lower = content.lower()
        score = 0.0
        
        # Term frequency scoring
        for term in financial_terms:
            count = content_lower.count(term)
            score += count * 0.1
        
        # Financial indicators
        if re.search(r'\$[\d,]+', content):
            score += 0.3
        if re.search(r'\d+\.?\d*%', content):
            score += 0.2
        if re.search(r'\b(?:Q[1-4]|quarter|fiscal)\s+\d{4}', content):
            score += 0.2
        
        # Normalize to 0-1 range
        return min(1.0, score)

# =============================================================================
# ChromaDB Vector Storage (Enhanced)
# =============================================================================

class ChromaVectorStorage:
    """Enhanced ChromaDB vector storage with advanced features"""
    
    def __init__(self, working_dir: str, collection_name: str = "financial_documents"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True)
        
        # Initialize ChromaDB with persistent storage
        chroma_db_path = str(self.working_dir / "chroma_db")
        self.client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(allow_reset=True)
        )
        
        self.collection_name = collection_name
        
        # Use sentence transformer for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing ChromaDB collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Advanced financial document chunks"}
            )
            logger.info(f"Created new ChromaDB collection: {collection_name}")
    
    def add_chunks_batch(self, chunks: List[TextChunk], batch_size: int = 20) -> None:
        """Add chunks in optimized batches"""
        if not chunks:
            return
        
        total_chunks = len(chunks)
        logger.info(f"Adding {total_chunks} chunks to ChromaDB in batches of {batch_size}")
        
        with tqdm(range(0, total_chunks, batch_size), desc="Adding to ChromaDB", unit="batch") as pbar:
            for i in pbar:
                batch = chunks[i:i + batch_size]
                
                documents = []
                metadatas = []
                ids = []
                
                for chunk in batch:
                    documents.append(chunk.content)
                    ids.append(chunk.id)
                    
                    # Prepare metadata (flatten for ChromaDB)
                    metadata = {
                        "doc_id": chunk.doc_id,
                        "chunk_index": chunk.chunk_index,
                        "tokens": chunk.tokens,
                        "section_title": chunk.section_title or "Unknown",
                        "section_index": chunk.section_index or 0,
                        "entities_count": len(chunk.entities),
                        "relationships_count": len(chunk.relationships),
                        "created_at": datetime.now().isoformat(),
                        "financial_relevance": chunk.metadata.get('financial_relevance', 0.0),
                        "has_numbers": chunk.metadata.get('has_numbers', False),
                        "has_currency": chunk.metadata.get('has_currency', False),
                        "word_count": chunk.metadata.get('word_count', 0)
                    }
                    
                    # Add entities and relationships as JSON if present
                    if chunk.entities:
                        metadata["entities"] = json.dumps(chunk.entities)
                    if chunk.relationships:
                        metadata["relationships"] = json.dumps(chunk.relationships)
                    
                    metadatas.append(metadata)
                
                # Add batch to ChromaDB
                try:
                    self.collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                except Exception as e:
                    logger.error(f"Failed to add batch {i//batch_size + 1}: {e}")
                    continue
                
                pbar.set_postfix({
                    "Added": f"{min(i + batch_size, total_chunks)}/{total_chunks}",
                    "Memory": f"{get_memory_usage():.0f}MB"
                })
                
                # Memory cleanup every few batches
                if (i // batch_size) % 5 == 0:
                    cleanup_memory()
    
    def similarity_search(self, query: str, top_k: int = 20, filters: Dict[str, Any] = None) -> List[Tuple[str, float, str]]:
        """Enhanced similarity search with filtering"""
        try:
            # Build where clause
            where_clause = {}
            if filters:
                if 'doc_id' in filters:
                    where_clause['doc_id'] = filters['doc_id']
                if 'min_financial_relevance' in filters:
                    where_clause['financial_relevance'] = {"$gte": filters['min_financial_relevance']}
                if 'has_currency' in filters:
                    where_clause['has_currency'] = filters['has_currency']
            
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, 100),
                where=where_clause if where_clause else None,
                include=["documents", "distances", "metadatas"]
            )
            
            # Process results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                    chunk_id = results['ids'][0][i]
                    similarity = 1 - distance
                    search_results.append((chunk_id, similarity, doc))
            
            return search_results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[str]:
        """Get chunk content by ID"""
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["documents"]
            )
            
            if result['documents'] and result['documents'][0]:
                return result['documents'][0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None
    
    def get_chunks_by_entities(self, entity_ids: List[str]) -> List[Tuple[str, str]]:
        """Get chunks that contain specific entities"""
        try:
            # This is a simplified implementation
            # In a real scenario, you'd want to store entity mappings
            all_results = self.collection.get(
                include=["documents", "metadatas", "ids"]
            )
            
            matching_chunks = []
            if all_results['metadatas']:
                for i, metadata in enumerate(all_results['metadatas']):
                    entities_json = metadata.get('entities', '[]')
                    try:
                        chunk_entities = json.loads(entities_json)
                        if any(entity_id in chunk_entities for entity_id in entity_ids):
                            chunk_id = all_results['ids'][i]
                            content = all_results['documents'][i]
                            matching_chunks.append((chunk_id, content))
                    except:
                        continue
            
            return matching_chunks[:10]  # Limit results
            
        except Exception as e:
            logger.error(f"Failed to get chunks by entities: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced collection statistics"""
        try:
            count = self.collection.count()
            
            if count == 0:
                return {"total_chunks": 0, "collection_name": self.collection_name}
            
            # Sample analysis
            sample_size = min(100, count)
            sample_results = self.collection.get(
                limit=sample_size,
                include=["metadatas"]
            )
            
            # Analyze sample
            doc_ids = set()
            total_relevance = 0
            currency_chunks = 0
            
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    doc_ids.add(metadata.get('doc_id', 'unknown'))
                    total_relevance += metadata.get('financial_relevance', 0)
                    if metadata.get('has_currency', False):
                        currency_chunks += 1
            
            sample_count = len(sample_results['metadatas']) if sample_results['metadatas'] else 1
            
            return {
                "total_chunks": count,
                "unique_documents": len(doc_ids),
                "avg_financial_relevance": total_relevance / sample_count,
                "currency_chunks_percent": (currency_chunks / sample_count) * 100,
                "collection_name": self.collection_name,
                "sample_size": sample_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get ChromaDB stats: {e}")
            return {"total_chunks": 0, "error": str(e)}


# SQLite Knowledge Graph (Enhanced)

class SQLiteKnowledgeGraph:
    """Enhanced SQLite knowledge graph with better indexing"""
    
    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True)
        
        self.db_path = self.working_dir / "knowledge_graph.db"
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Better performance
        
        # In-memory NetworkX graph for fast traversal
        self.graph = None
        self.graph_dirty = True
        
        self._create_tables()
    
    def _create_tables(self):
        """Create optimized database tables"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                properties TEXT,
                source_chunks TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                description TEXT,
                keywords TEXT,
                weight REAL DEFAULT 1.0,
                context TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(source_id) REFERENCES entities(id),
                FOREIGN KEY(target_id) REFERENCES entities(id)
            )
        ''')
        
        # Create comprehensive indexes
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)',
            'CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)',
            'CREATE INDEX IF NOT EXISTS idx_entities_name_type ON entities(name, type)',
            'CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id)',
            'CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id)',
            'CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relation_type)',
            'CREATE INDEX IF NOT EXISTS idx_relationships_source_target ON relationships(source_id, target_id)'
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        self.conn.commit()
    
    def add_entity(self, entity: FinancialEntity) -> None:
        """Add or update entity with better duplicate handling"""
        cursor = self.conn.cursor()
        
        try:
            # Check for existing entity with better matching
            cursor.execute('''
                SELECT id FROM entities 
                WHERE name = ? AND type = ?
                ORDER BY confidence DESC
                LIMIT 1
            ''', (entity.name, entity.type))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing entity with higher confidence or more recent data
                cursor.execute('''
                    UPDATE entities SET 
                        description = CASE 
                            WHEN LENGTH(?) > LENGTH(description) THEN ?
                            ELSE description
                        END,
                        properties = ?, 
                        source_chunks = ?,
                        confidence = CASE
                            WHEN ? > confidence THEN ?
                            ELSE confidence
                        END
                    WHERE id = ?
                ''', (
                    entity.description, entity.description,
                    json.dumps(entity.properties),
                    json.dumps(entity.source_chunks),
                    entity.confidence, entity.confidence,
                    existing[0]
                ))
            else:
                # Insert new entity
                cursor.execute('''
                    INSERT INTO entities (id, name, type, description, properties, source_chunks, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entity.id,
                    entity.name,
                    entity.type,
                    entity.description,
                    json.dumps(entity.properties),
                    json.dumps(entity.source_chunks),
                    entity.confidence
                ))
            
            self.conn.commit()
            self.graph_dirty = True
            
        except Exception as e:
            logger.error(f"Failed to add entity {entity.name}: {e}")
            self.conn.rollback()
    
    def add_relationship(self, relationship: FinancialRelationship) -> None:
        """Add or update relationship with better handling"""
        cursor = self.conn.cursor()
        
        try:
            # Check for existing relationship
            cursor.execute('''
                SELECT id, confidence FROM relationships 
                WHERE source_id = ? AND target_id = ? AND relation_type = ?
                LIMIT 1
            ''', (relationship.source_id, relationship.target_id, relationship.relation_type))
            
            existing = cursor.fetchone()
            
            if existing and existing[1] < relationship.confidence:
                # Update with higher confidence relationship
                cursor.execute('''
                    UPDATE relationships SET 
                        description = ?, 
                        keywords = ?, 
                        weight = ?,
                        context = ?,
                        confidence = ?
                    WHERE id = ?
                ''', (
                    relationship.description,
                    relationship.keywords,
                    relationship.weight,
                    relationship.context,
                    relationship.confidence,
                    existing[0]
                ))
            elif not existing:
                # Insert new relationship
                cursor.execute('''
                    INSERT INTO relationships 
                    (id, source_id, target_id, relation_type, description, keywords, weight, context, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    relationship.id,
                    relationship.source_id,
                    relationship.target_id,
                    relationship.relation_type,
                    relationship.description,
                    relationship.keywords,
                    relationship.weight,
                    relationship.context,
                    relationship.confidence
                ))
            
            self.conn.commit()
            self.graph_dirty = True
            
        except Exception as e:
            logger.error(f"Failed to add relationship: {e}")
            self.conn.rollback()
    
    def _build_graph(self) -> None:
        """Build NetworkX graph from database with caching"""
        if not self.graph_dirty and self.graph is not None:
            return
        
        logger.debug("Rebuilding knowledge graph...")
        self.graph = nx.MultiDiGraph()
        cursor = self.conn.cursor()
        
        try:
            # Add nodes (entities)
            cursor.execute('SELECT id, name, type, description, confidence FROM entities')
            for entity_id, name, etype, description, confidence in cursor.fetchall():
                self.graph.add_node(
                    entity_id, 
                    name=name, 
                    type=etype, 
                    description=description,
                    confidence=confidence
                )
            
            # Add edges (relationships)
            cursor.execute('''
                SELECT id, source_id, target_id, relation_type, description, confidence, weight 
                FROM relationships
            ''')
            for rel_id, source_id, target_id, rel_type, description, confidence, weight in cursor.fetchall():
                if source_id in self.graph and target_id in self.graph:
                    self.graph.add_edge(
                        source_id, target_id, 
                        key=rel_id,
                        relation_type=rel_type, 
                        description=description,
                        confidence=confidence,
                        weight=weight
                    )
            
            self.graph_dirty = False
            logger.debug(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Failed to build graph: {e}")
            self.graph = nx.MultiDiGraph()  # Empty graph as fallback
    
    def get_local_subgraph(self, entity_names: List[str], depth: int = 2) -> Dict[str, Any]:
        """Get enhanced local subgraph with relevance scoring"""
        self._build_graph()
        
        if not entity_names:
            return {"entities": {}, "relationships": {}}
        
        # Find entity IDs by name with fuzzy matching
        entity_ids = set()
        cursor = self.conn.cursor()
        
        for name in entity_names:
            # Exact match first
            cursor.execute('SELECT id FROM entities WHERE name = ?', (name,))
            exact_matches = cursor.fetchall()
            entity_ids.update([r[0] for r in exact_matches])
            
            # Fuzzy match if no exact match
            if not exact_matches:
                cursor.execute('SELECT id FROM entities WHERE name LIKE ?', (f'%{name}%',))
                fuzzy_matches = cursor.fetchall()
                entity_ids.update([r[0] for r in fuzzy_matches])
        
        if not entity_ids:
            return {"entities": {}, "relationships": {}}
        
        # Get subgraph with relevance weighting
        local_entities = {}
        local_relationships = {}
        
        for entity_id in entity_ids:
            if entity_id in self.graph:
                # Add the entity itself
                node_data = self.graph.nodes[entity_id]
                local_entities[entity_id] = {
                    "id": entity_id,
                    "name": node_data.get('name', ''),
                    "type": node_data.get('type', ''),
                    "description": node_data.get('description', ''),
                    "confidence": node_data.get('confidence', 0.0),
                    "relevance": 1.0  # Source entities have max relevance
                }
                
                # Get neighbors within depth
                try:
                    neighbors = nx.single_source_shortest_path_length(
                        self.graph.to_undirected(), entity_id, cutoff=depth
                    )
                    
                    for neighbor_id, distance in neighbors.items():
                        if neighbor_id not in local_entities and neighbor_id in self.graph:
                            node_data = self.graph.nodes[neighbor_id]
                            # Relevance decreases with distance
                            relevance = 1.0 / (1.0 + distance * 0.5)
                            
                            local_entities[neighbor_id] = {
                                "id": neighbor_id,
                                "name": node_data.get('name', ''),
                                "type": node_data.get('type', ''),
                                "description": node_data.get('description', ''),
                                "confidence": node_data.get('confidence', 0.0),
                                "relevance": relevance
                            }
                    
                    # Get relationships between these entities
                    for source in local_entities:
                        for target in local_entities:
                            if self.graph.has_edge(source, target):
                                edges = self.graph[source][target]
                                for key, edge_data in edges.items():
                                    local_relationships[key] = {
                                        "id": key,
                                        "source_id": source,
                                        "target_id": target,
                                        "relation_type": edge_data.get('relation_type', ''),
                                        "description": edge_data.get('description', ''),
                                        "confidence": edge_data.get('confidence', 0.0),
                                        "weight": edge_data.get('weight', 1.0)
                                    }
                
                except Exception as e:
                    logger.warning(f"Error getting neighbors for {entity_id}: {e}")
        
        return {"entities": local_entities, "relationships": local_relationships}
    
    def get_global_subgraph(self, themes: List[str], max_entities: int = 20) -> Dict[str, Any]:
        """Get global subgraph based on themes"""
        cursor = self.conn.cursor()
        
        global_entities = {}
        global_relationships = {}
        
        try:
            # Search for entities related to themes
            for theme in themes:
                cursor.execute('''
                    SELECT id, name, type, description, confidence 
                    FROM entities 
                    WHERE description LIKE ? OR name LIKE ?
                    ORDER BY confidence DESC
                    LIMIT ?
                ''', (f'%{theme}%', f'%{theme}%', max_entities // len(themes) + 1))
                
                for entity_id, name, etype, description, confidence in cursor.fetchall():
                    global_entities[entity_id] = {
                        "id": entity_id,
                        "name": name,
                        "type": etype,
                        "description": description,
                        "confidence": confidence,
                        "relevance": 0.8  # Thematic entities have high relevance
                    }
            
            # Get relationships between these entities
            entity_ids = list(global_entities.keys())
            if entity_ids:
                placeholders = ','.join('?' * len(entity_ids))
                cursor.execute(f'''
                    SELECT id, source_id, target_id, relation_type, description, confidence, weight
                    FROM relationships
                    WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})
                ''', entity_ids + entity_ids)
                
                for rel_id, source_id, target_id, rel_type, description, confidence, weight in cursor.fetchall():
                    global_relationships[rel_id] = {
                        "id": rel_id,
                        "source_id": source_id,
                        "target_id": target_id,
                        "relation_type": rel_type,
                        "description": description,
                        "confidence": confidence,
                        "weight": weight
                    }
        
        except Exception as e:
            logger.error(f"Failed to get global subgraph: {e}")
        
        return {"entities": global_entities, "relationships": global_relationships}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive knowledge graph statistics"""
        cursor = self.conn.cursor()
        
        try:
            # Basic counts
            cursor.execute('SELECT COUNT(*) FROM entities')
            total_entities = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM relationships')
            total_relationships = cursor.fetchone()[0]
            
            # Entity types
            cursor.execute('SELECT type, COUNT(*) FROM entities GROUP BY type ORDER BY COUNT(*) DESC')
            entity_types = dict(cursor.fetchall())
            
            # Relationship types
            cursor.execute('SELECT relation_type, COUNT(*) FROM relationships GROUP BY relation_type ORDER BY COUNT(*) DESC')
            relation_types = dict(cursor.fetchall())
            
            # Confidence statistics
            cursor.execute('SELECT AVG(confidence), MIN(confidence), MAX(confidence) FROM entities')
            entity_confidence = cursor.fetchone()
            
            cursor.execute('SELECT AVG(confidence), MIN(confidence), MAX(confidence) FROM relationships')
            relationship_confidence = cursor.fetchone()
            
            return {
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "entity_types": entity_types,
                "relation_types": relation_types,
                "entity_confidence": {
                    "avg": entity_confidence[0] or 0,
                    "min": entity_confidence[1] or 0,
                    "max": entity_confidence[2] or 0
                },
                "relationship_confidence": {
                    "avg": relationship_confidence[0] or 0,
                    "min": relationship_confidence[1] or 0,
                    "max": relationship_confidence[2] or 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get knowledge graph stats: {e}")
            return {"total_entities": 0, "total_relationships": 0, "error": str(e)}

# Advanced Knowledge Extractor

class AdvancedKnowledgeExtractor:
    """Advanced knowledge extractor with financial domain expertise"""
    
    def __init__(self, llm_client: OpenAI, max_memory_mb: int = 2048):
        self.llm_client = llm_client
        self.memory_monitor = MemoryMonitor(max_memory_mb)
        
        # Enhanced financial entity types
        self.financial_entity_types = [
            "COMPANY", "SUBSIDIARY", "COMPETITOR", "SUPPLIER", "CUSTOMER",
            "FINANCIAL_METRIC", "KPI", "RATIO", "BENCHMARK",
            "PRODUCT", "SERVICE", "BUSINESS_UNIT", "SEGMENT",
            "PERSON", "EXECUTIVE", "ANALYST", "BOARD_MEMBER",
            "LOCATION", "MARKET", "REGION", "FACILITY", "HEADQUARTERS",
            "EVENT", "MERGER", "ACQUISITION", "IPO", "EARNINGS_RELEASE",
            "RISK_FACTOR", "MARKET_RISK", "OPERATIONAL_RISK", "CREDIT_RISK",
            "REGULATION", "LAW", "STANDARD", "COMPLIANCE_REQUIREMENT",
            "TIME_PERIOD", "QUARTER", "FISCAL_YEAR", "REPORTING_PERIOD"
        ]
        
        self.financial_relation_types = [
            "OWNS", "OWNS_STAKE_IN", "SUBSIDIARY_OF", "PARENT_OF",
            "REPORTS", "MEASURES", "TRACKS", "BENCHMARKS_AGAINST",
            "COMPETES_WITH", "COLLABORATES_WITH", "PARTNERS_WITH",
            "SUPPLIES_TO", "PURCHASES_FROM", "DISTRIBUTES_THROUGH",
            "AFFECTS", "IMPACTS", "INFLUENCES", "DRIVES",
            "LOCATED_IN", "OPERATES_IN", "HEADQUARTERED_IN",
            "EMPLOYED_BY", "MANAGES", "LEADS", "REPORTS_TO",
            "REGULATED_BY", "COMPLIES_WITH", "SUBJECT_TO",
            "CAUSES", "CORRELATES_WITH", "DEPENDS_ON", "RESULTS_FROM",
            "OCCURRED_IN", "SCHEDULED_FOR", "REPORTED_IN"
        ]
    
    async def extract_entities_and_relationships(self, text: str, chunk_id: str) -> Tuple[List[FinancialEntity], List[FinancialRelationship]]:
        """Extract entities and relationships with advanced techniques"""
        
        memory_info = self.memory_monitor.check_and_cleanup()
        
        # Extract entities with financial context
        entities = await self._extract_financial_entities(text, chunk_id)
        cleanup_memory()
        
        # Extract relationships with enhanced logic
        relationships = await self._extract_financial_relationships(text, entities, chunk_id)
        cleanup_memory()
        
        return entities, relationships
    
    async def _extract_financial_entities(self, text: str, chunk_id: str) -> List[FinancialEntity]:
        """Extract financial entities with domain expertise"""
        
        # Limit text size but preserve financial context
        text_for_extraction = self._prepare_text_for_extraction(text, 1500)
        
        extraction_prompt = f"""
You are a financial analyst expert. Extract financial entities from the text with high precision.

Financial Entity Types (choose the most specific):
ORGANIZATIONS:
- COMPANY: Public/private companies, corporations
- SUBSIDIARY: Company subsidiaries, divisions
- COMPETITOR: Direct competitors, market rivals
- SUPPLIER: Supply chain partners, vendors
- CUSTOMER: Key customers, client segments

FINANCIAL MEASURES:
- FINANCIAL_METRIC: Revenue, profit, EBITDA, cash flow
- KPI: Key performance indicators, business metrics
- RATIO: Financial ratios, margin percentages
- BENCHMARK: Industry benchmarks, comparisons

BUSINESS ELEMENTS:
- PRODUCT: Products, product lines, brands
- SERVICE: Service offerings, business services
- BUSINESS_UNIT: Business segments, divisions
- SEGMENT: Market segments, customer segments

PEOPLE:
- EXECUTIVE: CEOs, CFOs, senior management
- ANALYST: Financial analysts, researchers
- BOARD_MEMBER: Board of directors members

PLACES:
- MARKET: Geographic markets, target markets
- REGION: Geographic regions, territories
- FACILITY: Manufacturing facilities, offices
- HEADQUARTERS: Corporate headquarters, main offices

EVENTS & TIME:
- EVENT: Mergers, acquisitions, launches
- MERGER: Merger transactions, combinations
- ACQUISITION: Acquisition deals, purchases
- EARNINGS_RELEASE: Earnings announcements, reports
- TIME_PERIOD: Quarters, fiscal years, periods

RISKS & REGULATIONS:
- RISK_FACTOR: Business risks, market risks
- REGULATION: Laws, regulations, standards

Text to analyze:
{text_for_extraction}

Extract entities as JSON array. Focus on financially significant entities:
[
    {{
        "name": "precise entity name",
        "type": "MOST_SPECIFIC_TYPE",
        "description": "detailed description including financial significance and context"
    }}
]

Maximum 10 entities. Prioritize entities with clear financial relevance.
"""
        
        try:
            response = await self._llm_call(extraction_prompt, max_tokens=1200)
            entities_data = self._parse_json_response(response)
            
            entities = []
            for entity_data in entities_data[:10]:
                if self._validate_entity_data(entity_data):
                    # Enhanced entity creation with financial analysis
                    properties = self._analyze_entity_properties(entity_data, text_for_extraction)
                    
                    entity = FinancialEntity(
                        id="",
                        name=entity_data["name"].strip(),
                        type=entity_data["type"],
                        description=entity_data["description"],
                        properties=properties,
                        source_chunks=[chunk_id],
                        confidence=self._calculate_entity_confidence(entity_data, text_for_extraction)
                    )
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def _extract_financial_relationships(self, text: str, entities: List[FinancialEntity], chunk_id: str) -> List[FinancialRelationship]:
        """Extract financial relationships with advanced reasoning"""
        
        if len(entities) < 2:
            return []
        
        # Limit entities for processing efficiency
        entities = entities[:6]
        entity_names = [e.name for e in entities]
        entity_lookup = {e.name: e.id for e in entities}
        
        text_for_extraction = self._prepare_text_for_extraction(text, 1200)
        
        relationship_prompt = f"""
You are a financial analyst. Identify precise relationships between these entities found in the text.

Entities: {entity_names}

Financial Relationship Types (choose most specific):
OWNERSHIP & STRUCTURE:
- OWNS: Direct ownership relationships
- SUBSIDIARY_OF: Subsidiary relationships
- PARENT_OF: Parent company relationships

FINANCIAL REPORTING:
- REPORTS: Company reports financial metrics
- MEASURES: Tracking/measuring performance
- BENCHMARKS_AGAINST: Comparison relationships

BUSINESS RELATIONSHIPS:
- COMPETES_WITH: Direct competition
- PARTNERS_WITH: Business partnerships
- SUPPLIES_TO: Supply chain relationships
- PURCHASES_FROM: Procurement relationships

IMPACT & INFLUENCE:
- AFFECTS: Direct impact relationships
- DRIVES: Causal driving relationships
- DEPENDS_ON: Dependency relationships
- CORRELATES_WITH: Correlation relationships

ORGANIZATIONAL:
- EMPLOYED_BY: Employment relationships
- MANAGES: Management relationships
- LEADS: Leadership relationships

REGULATORY & LOCATION:
- REGULATED_BY: Regulatory oversight
- LOCATED_IN: Geographic relationships
- OPERATES_IN: Operational presence

TEMPORAL:
- OCCURRED_IN: Event timing
- REPORTED_IN: Reporting periods

Text context:
{text_for_extraction}

Identify relationships as JSON array:
[
    {{
        "source": "source entity name (exact match)",
        "target": "target entity name (exact match)",
        "relation": "SPECIFIC_RELATIONSHIP_TYPE",
        "description": "detailed description of the relationship with supporting evidence",
        "keywords": "key terms that support this relationship"
    }}
]

Only include relationships explicitly supported by the text. Maximum 8 relationships.
Focus on financially significant relationships.
"""
        
        try:
            response = await self._llm_call(relationship_prompt, max_tokens=1000)
            relationships_data = self._parse_json_response(response)
            
            relationships = []
            for rel_data in relationships_data[:8]:
                if self._validate_relationship_data(rel_data, entity_lookup):
                    # Enhanced relationship creation
                    confidence = self._calculate_relationship_confidence(rel_data, text_for_extraction)
                    weight = self._calculate_relationship_weight(rel_data)
                    
                    relationship = FinancialRelationship(
                        id="",
                        source_id=entity_lookup[rel_data["source"]],
                        target_id=entity_lookup[rel_data["target"]],
                        relation_type=rel_data["relation"],
                        description=rel_data["description"],
                        keywords=rel_data.get("keywords", ""),
                        weight=weight,
                        context=text_for_extraction[:400],
                        confidence=confidence
                    )
                    relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []
    
    def _prepare_text_for_extraction(self, text: str, max_length: int) -> str:
        """Prepare text for extraction by preserving financial context"""
        if len(text) <= max_length:
            return text
        
        # Try to preserve financial sentences
        sentences = re.split(r'[.!?]+', text)
        financial_sentences = []
        other_sentences = []
        
        financial_keywords = [
            'revenue', 'profit', 'loss', 'earnings', 'sales', 'income',
            'margin', 'growth', 'decline', 'performance', 'financial',
            'quarter', 'fiscal', 'annual', 'business', '$', '%'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in financial_keywords):
                financial_sentences.append(sentence)
            else:
                other_sentences.append(sentence)
        
        # Prioritize financial sentences
        result_text = ""
        for sentence in financial_sentences:
            if len(result_text) + len(sentence) <= max_length:
                result_text += sentence + ". "
        
        # Add other sentences if space allows
        for sentence in other_sentences:
            if len(result_text) + len(sentence) <= max_length:
                result_text += sentence + ". "
            else:
                break
        
        return result_text.strip()
    
    def _analyze_entity_properties(self, entity_data: Dict, text: str) -> Dict[str, Any]:
        """Analyze additional properties for financial entities"""
        properties = {}
        
        entity_name = entity_data["name"].lower()
        text_lower = text.lower()
        
        # Financial indicators
        properties["mentioned_with_numbers"] = bool(
            re.search(rf"{re.escape(entity_name)}.*?\d+", text_lower) or
            re.search(rf"\d+.*?{re.escape(entity_name)}", text_lower)
        )
        
        properties["mentioned_with_currency"] = bool(
            re.search(rf"{re.escape(entity_name)}.*?\$[\d,]+", text_lower) or
            re.search(rf"\$[\d,]+.*?{re.escape(entity_name)}", text_lower)
        )
        
        properties["mentioned_with_percentage"] = bool(
            re.search(rf"{re.escape(entity_name)}.*?\d+\.?\d*%", text_lower) or
            re.search(rf"\d+\.?\d*%.*?{re.escape(entity_name)}", text_lower)
        )
        
        # Context analysis
        properties["context_length"] = len([s for s in text.split('.') if entity_name in s.lower()])
        properties["financial_context_score"] = self._calculate_financial_context_score(entity_name, text)
        
        return properties
    
    def _calculate_financial_context_score(self, entity_name: str, text: str) -> float:
        """Calculate how financially relevant the entity context is"""
        financial_terms = [
            'revenue', 'profit', 'loss', 'earnings', 'sales', 'income', 'ebitda',
            'margin', 'growth', 'decline', 'performance', 'financial', 'business',
            'quarter', 'fiscal', 'annual', 'cash', 'debt', 'equity', 'assets'
        ]
        
        # Find sentences containing the entity
        sentences = [s for s in text.split('.') if entity_name.lower() in s.lower()]
        
        if not sentences:
            return 0.0
        
        total_score = 0.0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentence_score = sum(1 for term in financial_terms if term in sentence_lower)
            total_score += sentence_score
        
        # Normalize by number of sentences
        return min(1.0, total_score / len(sentences))
    
    def _calculate_entity_confidence(self, entity_data: Dict, text: str) -> float:
        """Calculate entity extraction confidence"""
        base_confidence = 0.7
        
        # Boost confidence for entities with clear financial context
        entity_name = entity_data["name"].lower()
        
        # Check for explicit financial mentions
        if re.search(rf"{re.escape(entity_name)}.*?(?:\$[\d,]+|\d+\.?\d*%)", text.lower()):
            base_confidence += 0.2
        
        # Check for entity type consistency
        entity_type = entity_data["type"]
        if entity_type in ["FINANCIAL_METRIC", "KPI", "RATIO"] and any(
            term in entity_name for term in ['revenue', 'profit', 'margin', 'ratio', 'earnings']
        ):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _calculate_relationship_confidence(self, rel_data: Dict, text: str) -> float:
        """Calculate relationship extraction confidence"""
        base_confidence = 0.6
        
        # Check for explicit relationship indicators
        source = rel_data["source"].lower()
        target = rel_data["target"].lower()
        relation_type = rel_data["relation"]
        
        # Look for supporting text patterns
        if relation_type == "OWNS" and ("owns" in text.lower() or "ownership" in text.lower()):
            base_confidence += 0.2
        elif relation_type == "REPORTS" and ("reported" in text.lower() or "reports" in text.lower()):
            base_confidence += 0.2
        elif relation_type == "COMPETES_WITH" and ("compete" in text.lower() or "competitor" in text.lower()):
            base_confidence += 0.2
        
        # Check for proximity of entities in text
        source_pos = text.lower().find(source)
        target_pos = text.lower().find(target)
        
        if source_pos >= 0 and target_pos >= 0:
            distance = abs(source_pos - target_pos)
            if distance < 100:  # Entities mentioned close together
                base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _calculate_relationship_weight(self, rel_data: Dict) -> float:
        """Calculate relationship importance weight"""
        relation_type = rel_data["relation"]
        
        # Weight based on financial significance
        high_importance = ["OWNS", "REPORTS", "AFFECTS", "DRIVES"]
        medium_importance = ["COMPETES_WITH", "PARTNERS_WITH", "SUPPLIES_TO"]
        
        if relation_type in high_importance:
            return 1.0
        elif relation_type in medium_importance:
            return 0.8
        else:
            return 0.6
    
    async def _llm_call(self, prompt: str, max_tokens: int = 800) -> str:
        """Enhanced LLM call with better error handling"""
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,  # Lower temperature for more consistent extraction
                max_tokens=max_tokens,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "[]"
    
    def _parse_json_response(self, response: str) -> List[Dict]:
        """Enhanced JSON parsing with better error handling"""
        try:
            # Clean up response
            response = response.strip()
            
            # Remove markdown formatting
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end > start:
                    response = response[start:end]
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                if end > start:
                    response = response[start:end]
            
            response = response.strip()
            
            # Try to parse JSON
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}. Response: {response[:200]}...")
            
            # Try to extract JSON array from response
            try:
                # Look for array pattern
                array_match = re.search(r'\[.*\]', response, re.DOTALL)
                if array_match:
                    return json.loads(array_match.group())
            except:
                pass
            
            return []
    
    def _validate_entity_data(self, entity_data: Dict) -> bool:
        """Enhanced entity validation"""
        if not isinstance(entity_data, dict):
            return False
        
        required_fields = ["name", "type", "description"]
        if not all(field in entity_data for field in required_fields):
            return False
        
        # Validate entity type
        if entity_data["type"] not in self.financial_entity_types:
            logger.warning(f"Unknown entity type: {entity_data['type']}")
            return False
        
        # Validate name length
        if len(entity_data["name"].strip()) < 2:
            return False
        
        return True
    
    def _validate_relationship_data(self, rel_data: Dict, entity_lookup: Dict) -> bool:
        """Enhanced relationship validation"""
        if not isinstance(rel_data, dict):
            return False
        
        required_fields = ["source", "target", "relation", "description"]
        if not all(field in rel_data for field in required_fields):
            return False
        
        # Validate entities exist
        if (rel_data["source"] not in entity_lookup or 
            rel_data["target"] not in entity_lookup):
            return False
        
        # Validate relationship type
        if rel_data["relation"] not in self.financial_relation_types:
            logger.warning(f"Unknown relationship type: {rel_data['relation']}")
            return False
        
        # Prevent self-relationships
        if rel_data["source"] == rel_data["target"]:
            return False
        
        return True

# Main Advanced System

class AdvancedMemoryEfficientFinancialRAG:
    """
    Advanced Memory-Efficient FinancialLightRAG
    - Advanced chunking with financial document understanding
    - Enhanced ChromaDB storage with filtering
    - Sophisticated knowledge extraction
    - Comprehensive memory management
    """
    
    def __init__(
        self,
        working_dir: str = "./advanced_financial_rag",
        max_memory_mb: int = 2048,
        chunk_size: int = 1200,
        chunk_overlap_ratio: float = 0.15,
        openai_api_key: Optional[str] = None
    ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True)
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor(max_memory_mb)
        
        # Initialize LLM client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm_client = OpenAI(api_key=api_key)
        
        # Initialize advanced components
        self.chunker = AdvancedFinancialChunker(
            base_chunk_size=chunk_size,
            overlap_ratio=chunk_overlap_ratio,
            max_chunks_per_doc=2000
        )
        
        self.knowledge_extractor = AdvancedKnowledgeExtractor(
            self.llm_client, max_memory_mb
        )
        
        self.knowledge_graph = SQLiteKnowledgeGraph(str(self.working_dir))
        self.vector_storage = ChromaVectorStorage(str(self.working_dir))
        
        # Document tracking
        self.processed_docs = {}
        self._save_config()
        
        logger.info("Advanced Memory-Efficient FinancialLightRAG initialized successfully")
    
    def _save_config(self):
        """Save system configuration"""
        config = {
            "version": "2.0-advanced",
            "chunk_size": self.chunker.base_chunk_size,
            "overlap_ratio": self.chunker.overlap_ratio,
            "max_memory_mb": self.memory_monitor.max_memory_mb,
            "created_at": datetime.now().isoformat()
        }
        
        config_path = self.working_dir / "system_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    async def ainsert(self, text: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Advanced document insertion with comprehensive processing
        """
        if doc_id is None:
            doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        
        start_time = time.time()
        initial_memory = get_memory_usage()
        
        logger.info(f"Processing document {doc_id} ({len(text):,} chars, Memory: {initial_memory:.1f}MB)")
        
        try:
            # Progress tracking
            with tqdm(total=5, desc=f"Processing {doc_id}", unit="step") as main_pbar:
                
                # Step 1: Advanced chunking
                main_pbar.set_description("Advanced chunking...")
                chunks = self.chunker.create_chunks(text, doc_id)
                main_pbar.update(1)
                
                if not chunks:
                    logger.warning(f"No chunks created for {doc_id}")
                    return {"status": "warning", "message": "No chunks created"}
                
                # Step 2: Knowledge extraction
                main_pbar.set_description("Extracting knowledge...")
                all_entities = []
                all_relationships = []
                
                # Process chunks in batches to manage memory
                batch_size = 5
                chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
                
                with tqdm(chunk_batches, desc="Processing chunk batches", leave=False) as batch_pbar:
                    for batch_idx, chunk_batch in enumerate(batch_pbar):
                        batch_entities = []
                        batch_relationships = []
                        
                        for chunk in chunk_batch:
                            entities, relationships = await self.knowledge_extractor.extract_entities_and_relationships(
                                chunk.content, chunk.id
                            )
                            
                            chunk.entities = [e.id for e in entities]
                            chunk.relationships = [r.id for r in relationships]
                            
                            batch_entities.extend(entities)
                            batch_relationships.extend(relationships)
                        
                        # Store knowledge for this batch
                        for entity in batch_entities:
                            self.knowledge_graph.add_entity(entity)
                        
                        for relationship in batch_relationships:
                            self.knowledge_graph.add_relationship(relationship)
                        
                        all_entities.extend(batch_entities)
                        all_relationships.extend(batch_relationships)
                        
                        # Memory management
                        cleanup_memory()
                        memory_info = self.memory_monitor.check_and_cleanup()
                        
                        batch_pbar.set_postfix({
                            "Entities": len(all_entities),
                            "Relations": len(all_relationships),
                            "Memory": f"{memory_info['current_mb']:.0f}MB"
                        })
                
                main_pbar.update(1)
                
                # Step 3: Store chunks in ChromaDB
                main_pbar.set_description("Storing in vector DB...")
                self.vector_storage.add_chunks_batch(chunks, batch_size=25)
                main_pbar.update(1)
                
                # Step 4: Final optimization
                main_pbar.set_description("Optimizing...")
                cleanup_memory()
                final_memory = get_memory_usage()
                main_pbar.update(1)
                
                # Step 5: Save metadata
                main_pbar.set_description("Saving metadata...")
                processing_time = time.time() - start_time
                
                doc_metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "chunks": len(chunks),
                    "entities": len(all_entities),
                    "relationships": len(all_relationships),
                    "processing_time_seconds": processing_time,
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "memory_delta_mb": final_memory - initial_memory,
                    "text_length": len(text),
                    "avg_chunk_size": sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0
                }
                
                self.processed_docs[doc_id] = doc_metadata
                main_pbar.update(1)
            
            logger.info(f"Successfully processed {doc_id}: {len(chunks)} chunks, "
                       f"{len(all_entities)} entities, {len(all_relationships)} relationships "
                       f"in {processing_time:.1f}s")
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "chunks_created": len(chunks),
                "entities_extracted": len(all_entities),
                "relationships_extracted": len(all_relationships),
                "processing_time": processing_time,
                "memory_usage_mb": final_memory,
                "avg_chunk_financial_relevance": sum(
                    c.metadata.get('financial_relevance', 0) for c in chunks
                ) / len(chunks) if chunks else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to process document {doc_id}: {e}")
            return {
                "status": "error",
                "doc_id": doc_id,
                "error": str(e),
                "memory_usage_mb": get_memory_usage()
            }
    
    def insert(self, text: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous insert wrapper"""
        return asyncio.run(self.ainsert(text, doc_id))
    
    async def aquery(self, query: str, param: QueryParam = QueryParam()) -> str:
        """
        Advanced query processing with enhanced retrieval
        """
        start_time = time.time()
        logger.info(f"Processing query: {query} [mode: {param.mode}] (Memory: {get_memory_usage():.1f}MB)")
        
        try:
            with tqdm(total=3, desc=f"Query [{param.mode}]", leave=False) as pbar:
                
                # Step 1: Query analysis and planning
                pbar.set_description("Analyzing query...")
                query_analysis = await self._analyze_query(query)
                pbar.update(1)
                
                # Step 2: Retrieve context based on query analysis and mode
                pbar.set_description("Retrieving context...")
                if param.mode == "naive":
                    context = await self._naive_retrieval(query, param)
                elif param.mode == "local":
                    context = await self._local_retrieval(query, param, query_analysis)
                elif param.mode == "global":
                    context = await self._global_retrieval(query, param, query_analysis)
                elif param.mode == "hybrid":
                    context = await self._hybrid_retrieval(query, param, query_analysis)
                elif param.mode == "mix":
                    context = await self._mix_retrieval(query, param, query_analysis)
                else:
                    context = await self._hybrid_retrieval(query, param, query_analysis)
                
                pbar.update(1)
                
                if param.only_need_context:
                    return context
                
                # Step 3: Generate enhanced response
                pbar.set_description("Generating response...")
                response = await self._generate_enhanced_response(query, context, param, query_analysis)
                pbar.update(1)
                
                processing_time = time.time() - start_time
                logger.info(f"Query processed in {processing_time:.2f}s")
                
                cleanup_memory()
                return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return f"I encountered an error processing your query: {str(e)}"
    
    def query(self, query: str, param: QueryParam = QueryParam()) -> str:
        """Synchronous query wrapper"""
        return asyncio.run(self.aquery(query, param))
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand intent and key concepts"""
        
        analysis_prompt = f"""
Analyze this financial query to understand the user's intent and key concepts:

Query: {query}

Provide analysis as JSON:
{{
    "intent": "information_seeking|comparison|analysis|prediction|explanation",
    "key_entities": ["entity1", "entity2"],
    "financial_concepts": ["concept1", "concept2"],
    "time_focus": "current|historical|future|general",
    "specificity": "specific|general",
    "requires_calculations": true/false,
    "requires_trends": true/false
}}

Focus on identifying:
1. What the user wants to know (intent)
2. Which companies/entities they're asking about
3. What financial concepts are involved
4. Whether they need current vs historical data
"""
        
        try:
            response = await self.knowledge_extractor._llm_call(analysis_prompt, max_tokens=400)
            analysis = self.knowledge_extractor._parse_json_response(response)
            
            if analysis and isinstance(analysis, list) and len(analysis) > 0:
                return analysis[0]
            elif isinstance(analysis, dict):
                return analysis
            else:
                # Fallback analysis
                return {
                    "intent": "information_seeking",
                    "key_entities": [],
                    "financial_concepts": [],
                    "time_focus": "general",
                    "specificity": "general",
                    "requires_calculations": False,
                    "requires_trends": False
                }
                
        except Exception as e:
            logger.warning(f"Query analysis failed: {e}")
            return {
                "intent": "information_seeking",
                "key_entities": [],
                "financial_concepts": [],
                "time_focus": "general",
                "specificity": "general",
                "requires_calculations": False,
                "requires_trends": False
            }
    
    async def _naive_retrieval(self, query: str, param: QueryParam) -> str:
        """Enhanced naive retrieval with financial filtering"""
        # Use financial relevance filtering
        filters = {"min_financial_relevance": 0.3}
        search_results = self.vector_storage.similarity_search(query, param.top_k, filters)
        
        if not search_results:
            # Fallback without filters
            search_results = self.vector_storage.similarity_search(query, param.top_k)
        
        context_parts = []
        for chunk_id, similarity, content in search_results[:10]:
            context_parts.append(f"[Relevance: {similarity:.3f}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    async def _local_retrieval(self, query: str, param: QueryParam, query_analysis: Dict[str, Any]) -> str:
        """Enhanced local retrieval with query-aware entity extraction"""
        # Extract entities from query with context
        query_entities, _ = await self.knowledge_extractor.extract_entities_and_relationships(query, "query")
        
        # Add entities from query analysis
        entity_names = [e.name for e in query_entities]
        entity_names.extend(query_analysis.get("key_entities", []))
        
        if not entity_names:
            return await self._naive_retrieval(query, param)
        
        # Get local subgraph with enhanced depth based on query specificity
        depth = 2 if query_analysis.get("specificity") == "specific" else 1
        local_subgraph = self.knowledge_graph.get_local_subgraph(entity_names, depth=depth)
        
        context_parts = []
        
        # Add entity information with relevance ranking
        if local_subgraph["entities"]:
            context_parts.append("=== RELEVANT ENTITIES ===")
            sorted_entities = sorted(
                local_subgraph["entities"].values(),
                key=lambda x: x.get("relevance", 0),
                reverse=True
            )
            
            for entity in sorted_entities[:8]:
                context_parts.append(
                    f"• {entity['name']} ({entity['type']}): {entity['description']}"
                )
        
        # Add relationships with relevance
        if local_subgraph["relationships"]:
            context_parts.append("\n=== KEY RELATIONSHIPS ===")
            for rel in list(local_subgraph["relationships"].values())[:6]:
                context_parts.append(f"• {rel['description']}")
        
        # Add relevant chunks with entity context
        entity_ids = list(local_subgraph["entities"].keys())
        if entity_ids:
            relevant_chunks = self.vector_storage.get_chunks_by_entities(entity_ids)
            if relevant_chunks:
                context_parts.append("\n=== SUPPORTING CONTENT ===")
                for chunk_id, content in relevant_chunks[:5]:
                    context_parts.append(f"{content}\n")
        
        return "\n".join(context_parts)
    
    async def _global_retrieval(self, query: str, param: QueryParam, query_analysis: Dict[str, Any]) -> str:
        """Enhanced global retrieval with thematic analysis"""
        # Extract themes from query and analysis
        themes = await self._extract_themes(query)
        themes.extend(query_analysis.get("financial_concepts", []))
        
        # Remove duplicates
        themes = list(set(themes))
        
        if not themes:
            return await self._naive_retrieval(query, param)
        
        # Get global subgraph
        global_subgraph = self.knowledge_graph.get_global_subgraph(themes, max_entities=param.top_k)
        
        context_parts = []
        
        # Add thematic entities
        if global_subgraph["entities"]:
            context_parts.append("=== THEMATIC ENTITIES ===")
            for entity in list(global_subgraph["entities"].values())[:8]:
                context_parts.append(
                    f"• {entity['name']} ({entity['type']}): {entity['description']}"
                )
        
        # Add thematic relationships
        if global_subgraph["relationships"]:
            context_parts.append("\n=== THEMATIC RELATIONSHIPS ===")
            for rel in list(global_subgraph["relationships"].values())[:6]:
                context_parts.append(f"• {rel['description']}")
        
        # Add vector search results with theme filtering
        search_results = self.vector_storage.similarity_search(query, param.top_k // 2)
        if search_results:
            context_parts.append("\n=== RELEVANT CONTENT ===")
            for chunk_id, similarity, content in search_results[:6]:
                context_parts.append(f"[Similarity: {similarity:.3f}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    async def _hybrid_retrieval(self, query: str, param: QueryParam, query_analysis: Dict[str, Any]) -> str:
        """Enhanced hybrid retrieval with intelligent weighting"""
        # Determine retrieval weights based on query analysis
        if query_analysis.get("specificity") == "specific":
            local_weight = 0.7
            global_weight = 0.3
        else:
            local_weight = 0.4
            global_weight = 0.6
        
        # Get local context
        local_context = await self._local_retrieval(
            query, 
            QueryParam(mode="local", top_k=int(param.top_k * local_weight)),
            query_analysis
        )
        
        # Get global context
        global_context = await self._global_retrieval(
            query,
            QueryParam(mode="global", top_k=int(param.top_k * global_weight)),
            query_analysis
        )
        
        # Combine with intelligent weighting
        combined_context = f"""
=== LOCAL CONTEXT (Entity-Focused) ===
{local_context}

=== GLOBAL CONTEXT (Thematic) ===
{global_context}
"""
        
        return combined_context
    
    async def _mix_retrieval(self, query: str, param: QueryParam, query_analysis: Dict[str, Any]) -> str:
        """Enhanced mix retrieval integrating all sources"""
        context_parts = []
        
        # Vector search with financial filtering
        filters = {"has_currency": True} if "financial" in query.lower() else {}
        vector_results = self.vector_storage.similarity_search(query, param.top_k // 2, filters)
        
        if vector_results:
            context_parts.append("=== VECTOR SEARCH RESULTS ===")
            for chunk_id, similarity, content in vector_results[:4]:
                context_parts.append(f"[Similarity: {similarity:.3f}]\n{content}\n")
        
        # Entity-based results
        query_entities, _ = await self.knowledge_extractor.extract_entities_and_relationships(query, "query")
        entity_names = [e.name for e in query_entities]
        entity_names.extend(query_analysis.get("key_entities", []))
        
        if entity_names:
            local_subgraph = self.knowledge_graph.get_local_subgraph(entity_names, depth=1)
            
            if local_subgraph["entities"]:
                context_parts.append("\n=== KNOWLEDGE GRAPH RESULTS ===")
                for entity in list(local_subgraph["entities"].values())[:4]:
                    context_parts.append(
                        f"• {entity['name']} ({entity['type']}): {entity['description']}"
                    )
                
                for rel in list(local_subgraph["relationships"].values())[:4]:
                    context_parts.append(f"• Relationship: {rel['description']}")
        
        # Thematic results
        themes = await self._extract_themes(query)
        if themes:
            global_subgraph = self.knowledge_graph.get_global_subgraph(themes[:3], max_entities=6)
            
            if global_subgraph["entities"]:
                context_parts.append("\n=== THEMATIC ANALYSIS ===")
                for entity in list(global_subgraph["entities"].values())[:3]:
                    context_parts.append(f"• {entity['name']}: {entity['description']}")
        
        return "\n".join(context_parts)
    
    async def _extract_themes(self, query: str) -> List[str]:
        """Enhanced theme extraction with financial focus"""
        theme_prompt = f"""
Extract main financial themes from this query. Focus on business and financial concepts:

Query: {query}

Return JSON list of 3-5 key themes relevant to financial analysis:
["theme1", "theme2", "theme3"]

Financial theme categories:
- Financial Performance: revenue, profit, growth, margins, earnings
- Business Operations: market share, competition, efficiency, costs
- Financial Position: debt, equity, cash, assets, liquidity
- Market Conditions: industry trends, economic factors, regulations
- Risk Factors: business risks, market risks, operational risks
- Strategic Actions: mergers, acquisitions, expansions, investments
"""
        
        try:
            response = await self.knowledge_extractor._llm_call(theme_prompt, max_tokens=300)
            themes = self.knowledge_extractor._parse_json_response(response)
            return themes if isinstance(themes, list) else []
        except:
            # Fallback keyword extraction
            financial_keywords = [
                "revenue", "profit", "earnings", "growth", "margin", "performance",
                "market", "competition", "risk", "debt", "equity", "cash", "assets",
                "investment", "acquisition", "merger", "strategy", "business"
            ]
            words = query.lower().split()
            return [word for word in words if word in financial_keywords]
    
    async def _generate_enhanced_response(self, query: str, context: str, param: QueryParam, query_analysis: Dict[str, Any]) -> str:
        """Generate enhanced response with query-aware analysis"""
        
        # Customize system prompt based on query analysis
        intent = query_analysis.get("intent", "information_seeking")
        
        if intent == "comparison":
            system_prompt = """You are a financial analyst expert specializing in comparative analysis. 
            Provide detailed comparisons with specific metrics, identify key differences, and explain implications."""
        elif intent == "prediction":
            system_prompt = """You are a financial analyst expert specializing in forecasting and trend analysis. 
            Analyze trends, identify patterns, and provide informed projections based on available data."""
        elif intent == "analysis":
            system_prompt = """You are a financial analyst expert specializing in deep financial analysis. 
            Provide comprehensive analysis with supporting data, identify key insights, and explain business implications."""
        else:
            system_prompt = """You are a financial analyst expert. Provide clear, accurate information 
            based on the context. Focus on financial metrics, business implications, and actionable insights."""
        
        # Limit context size intelligently
        context_limited = self._optimize_context(context, param.max_token_for_text_unit)
        
        user_prompt = f"""
Query: {query}

Financial Context:
{context_limited}

Analysis Requirements:
- Query Intent: {intent}
- Response Type: {param.response_type.lower()}
- Focus Areas: {', '.join(query_analysis.get('financial_concepts', []))}

Please provide a comprehensive response that:
1. Directly addresses the query using the provided context
2. Includes specific financial data when available
3. Explains business implications and significance
4. Maintains analytical rigor and accuracy

Response should be in {param.response_type.lower()} format.
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1200,  # Increased for better responses
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I encountered an error generating the response. Please try rephrasing your query."
    
    def _optimize_context(self, context: str, max_tokens: int) -> str:
        """Optimize context length while preserving important information"""
        if len(context.split()) <= max_tokens:
            return context
        
        # Split into sections and prioritize
        sections = context.split("===")
        prioritized_sections = []
        
        # Prioritize sections with financial data
        financial_indicators = ["$", "%", "revenue", "profit", "earnings", "growth"]
        
        for section in sections:
            score = sum(1 for indicator in financial_indicators if indicator in section.lower())
            prioritized_sections.append((score, section))
        
        # Sort by score and rebuild
        prioritized_sections.sort(key=lambda x: x[0], reverse=True)
        
        result = ""
        for score, section in prioritized_sections:
            if len((result + section).split()) <= max_tokens:
                result += "===" + section if result else section
            else:
                break
        
        return result
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        memory_info = self.memory_monitor.check_and_cleanup()
        
        # Calculate processing efficiency
        total_docs = len(self.processed_docs)
        total_processing_time = sum(
            doc.get("processing_time_seconds", 0) 
            for doc in self.processed_docs.values()
        )
        
        return {
            "system_info": {
                "version": "2.0-advanced",
                "total_documents": total_docs,
                "total_processing_time": total_processing_time,
                "avg_processing_time": total_processing_time / total_docs if total_docs > 0 else 0,
                "working_directory": str(self.working_dir)
            },
            "memory_usage": {
                "current_mb": memory_info["current_mb"],
                "max_mb": self.memory_monitor.max_memory_mb,
                "usage_percent": memory_info["usage_percent"],
                "cleanup_count": memory_info["cleanup_count"]
            },
            "knowledge_graph": self.knowledge_graph.get_stats(),
            "vector_storage": self.vector_storage.get_stats(),
            "chunking_stats": {
                "base_chunk_size": self.chunker.base_chunk_size,
                "overlap_ratio": self.chunker.overlap_ratio,
                "max_chunks_per_doc": self.chunker.max_chunks_per_doc
            },
            "recent_documents": list(self.processed_docs.values())[-5:],
            # Fixed: Use the correct key name that terminal interface expects
            "processed_documents": total_docs
        }

# =============================================================================
# Example Usage and Testing
# =============================================================================

async def main():
    """Advanced example usage"""
    
    # Initialize with advanced settings
    rag = AdvancedMemoryEfficientFinancialRAG(
        working_dir="./advanced_financial_rag",
        max_memory_mb=2048,
        chunk_size=1200,  # Optimized chunk size
        chunk_overlap_ratio=0.15,  # 15% overlap
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Enhanced sample document with more complexity
    sample_doc = """
    Apple Inc. Quarterly Financial Results - Q1 2024
    Executive Summary
    
    Apple delivered exceptional results in Q1 2024, demonstrating the strength of our diversified product portfolio 
    and services ecosystem. Total net sales reached $119.6 billion, representing a 2% year-over-year increase 
    despite challenging macroeconomic conditions.
    
    Financial Performance Highlights
    
    Revenue Performance:
    Total net sales for Q1 2024 were $119.6 billion, up 2% year-over-year from $117.2 billion in Q1 2023.
    iPhone revenue was $69.7 billion, down 1% from $70.2 billion in the prior year quarter.
    Services revenue reached a new all-time high of $23.1 billion, up 11% from $20.8 billion.
    Mac revenue was $7.8 billion, up 1% year-over-year from $7.7 billion.
    iPad revenue was $7.0 billion, down 25% year-over-year from $9.3 billion.
    Wearables, Home and Accessories revenue was $12.0 billion, down 11% from $13.5 billion.
    
    Profitability Analysis:
    Gross margin was 45.9%, compared to 44.5% in the prior year quarter, representing a 140 basis point improvement.
    Operating margin was 30.0%, compared to 29.4% in the prior year quarter.
    Net income was $33.9 billion, or $2.18 per diluted share, up 16% year-over-year.
    Return on invested capital (ROIC) remained strong at 29.5%.
    
    Geographic Performance:
    Americas revenue was $50.4 billion, up 4% year-over-year, driven by strong iPhone and Services performance.
    Europe revenue was $23.3 billion, down 1% year-over-year, impacted by foreign exchange headwinds.
    Greater China revenue was $20.8 billion, down 13% year-over-year, reflecting challenging market conditions.
    Japan revenue was $8.1 billion, up 2% year-over-year
    
    Revenue Performance:
    Total net sales for Q1 2024 were $119.6 billion, up 2% year-over-year.
    iPhone revenue was $69.7 billion, down 1% from prior year quarter.
    Services revenue reached a new all-time high of $23.1 billion, up 11%.
    
    Product Performance:
    Mac revenue was $7.8 billion, up 1% year-over-year.
    iPad revenue was $7.0 billion, down 25% year-over-year.
    Wearables revenue was $12.0 billion, down 11% year-over-year.
    
    Geographic Performance:
    Americas revenue was $50.4 billion, up 4% year-over-year.
    Europe revenue was $23.3 billion, down 1% year-over-year.
    Greater China revenue was $20.8 billion, down 13% year-over-year.
    
    Key Metrics:
    Gross margin was 45.9%, compared to 44.5% in the prior year quarter.
    Operating margin was 30.0%, compared to 29.4% in the prior year quarter.
    Diluted earnings per share were $2.18, up 16% year-over-year.
    """
    
    print("🏦 Memory-Efficient FinancialLightRAG Demo")
    print("=" * 50)
    
    # Process document
    print("📄 Processing Apple Q1 2024 results...")
    result = await rag.ainsert(sample_doc, "apple_q1_2024")
    print(f"✅ Result: {result}")
    
    # Test queries
    queries = [
        ("What was Apple's Q1 2024 revenue?", "local"),
        ("How did different product lines perform?", "global"),
        ("What were the key financial metrics?", "hybrid")
    ]
    
    print("\n🔍 Testing queries...")
    for query, mode in queries:
        print(f"\n--- Query: {query} [Mode: {mode}] ---")
        response = await rag.aquery(query, QueryParam(mode=mode))
        print(f"Response: {response[:200]}...")
    
    # Show stats
    print(f"\n📊 System Statistics:")
    stats = rag.get_comprehensive_stats()
    print(f"Memory Usage: {stats['memory_usage']['current_mb']:.1f}MB")
    print(f"ChromaDB Chunks: {stats['vector_storage']['total_chunks']}")
    print(f"Knowledge Graph: {stats['knowledge_graph']['total_entities']} entities, {stats['knowledge_graph']['total_relationships']} relationships")

if __name__ == "__main__":
    asyncio.run(main())