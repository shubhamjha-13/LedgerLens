# 🏦 FinancialLightRAG Core

**Advanced Graph-Enhanced RAG System for Financial Documents**

This is a sophisticated implementation following LightRAG's architecture, specialized for financial document analysis with advanced graph strategies, dual-level retrieval, and financial domain expertise.

## ✨ Key Features

### 🧠 Advanced Graph Intelligence
- **Entity & Relationship Extraction**: LLM-powered extraction of financial entities and their relationships
- **Knowledge Graph Construction**: NetworkX-based graph with financial relationship types
- **Deduplication**: Intelligent merging of identical entities and relationships
- **Key-Value Storage**: Efficient retrieval following LightRAG's approach

### 🔍 Sophisticated Retrieval Modes
- **Local Mode**: Entity-focused retrieval for specific financial data
- **Global Mode**: Theme-based retrieval exploring broader financial concepts
- **Hybrid Mode**: Combines local precision with global context
- **Mix Mode**: Integrates knowledge graph and vector retrieval
- **Naive Mode**: Simple vector similarity search

### 📊 Financial Domain Specialization
- **Entity Types**: Companies, metrics, products, people, locations, events, risk factors
- **Relationship Types**: Owns, reports, competes_with, supplies_to, affects, etc.
- **Financial Chunking**: Smart text segmentation respecting financial document structure
- **Table Awareness**: Enhanced processing for financial tables and structured data

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-openai-api-key"
```

### Basic Usage
```python
import asyncio
from financial_light_rag_core import FinancialLightRAG, QueryParam

async def main():
    # Initialize system
    rag = FinancialLightRAG(working_dir="./data")
    
    # Process financial document
    financial_text = "Apple Inc. reported revenue of $383.3 billion..."
    result = await rag.ainsert(financial_text, "apple_10k")
    
    # Query with different modes
    response = await rag.aquery(
        "What are Apple's main revenue sources?",
        QueryParam(mode="local")
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

### Run Demo
```bash
python demo_financial_rag.py
```

## 🎯 Advanced Usage

### Different Retrieval Modes

#### Local Retrieval (Entity-Focused)
```python
# Best for specific company/metric queries
response = await rag.aquery(
    "What was Apple's iPhone revenue?", 
    QueryParam(mode="local")
)
```

#### Global Retrieval (Theme-Focused)
```python
# Best for thematic analysis across companies
response = await rag.aquery(
    "How do supply chain issues affect tech companies?",
    QueryParam(mode="global")
)
```

#### Hybrid Retrieval (Comprehensive)
```python
# Best for complex analysis requiring both specific data and context
response = await rag.aquery(
    "Compare Apple and Microsoft's cloud strategies and revenue impact",
    QueryParam(mode="hybrid")
)
```

#### Mix Retrieval (Graph + Vector)
```python
# Integrates knowledge graph and vector search
response = await rag.aquery(
    "What are the emerging risks in the technology sector?",
    QueryParam(mode="mix")
)
```

### Context-Only Retrieval
```python
# Get raw context without LLM generation
context = await rag.aquery(
    "What metrics are discussed?",
    QueryParam(mode="hybrid", only_need_context=True)
)
```

### Batch Processing
```python
# Process multiple documents
documents = [
    ("Apple 10-K content", "apple_10k"),
    ("Microsoft earnings", "msft_earnings"),
    ("Tesla risks", "tesla_risks")
]

for content, doc_id in documents:
    result = await rag.ainsert(content, doc_id)
    print(f"Processed {doc_id}: {result}")
```

## 🏗️ Architecture Overview

### Core Components

#### 1. FinancialKnowledgeExtractor
- Extracts entities and relationships using LLM
- Financial domain-specific entity types and relationships
- Advanced prompt engineering for financial analysis

#### 2. FinancialKnowledgeGraph
- NetworkX-based graph storage
- Key-value approach for efficient retrieval
- Deduplication and merging strategies
- Local and global subgraph extraction

#### 3. FinancialVectorStorage
- Vector embeddings for text chunks
- Similarity search capabilities
- Integration with knowledge graph

#### 4. FinancialLightRAG (Main System)
- Orchestrates all components
- Implements dual-level retrieval strategies
- Financial-aware text chunking
- Async API following LightRAG patterns

### Data Models

#### FinancialEntity
```python
@dataclass
class FinancialEntity:
    id: str
    name: str
    type: str  # COMPANY, FINANCIAL_METRIC, PRODUCT, etc.
    description: str
    properties: Dict[str, Any]
    source_chunks: List[str]
    confidence: float
```

#### FinancialRelationship
```python
@dataclass
class FinancialRelationship:
    id: str
    source_id: str
    target_id: str
    relation_type: str  # OWNS, REPORTS, COMPETES_WITH, etc.
    description: str
    keywords: str
    weight: float
    context: str
    confidence: float
```

## 📊 Financial Entity Types

- **COMPANY**: Public/private companies, subsidiaries, competitors
- **FINANCIAL_METRIC**: Revenue, profit, margins, ratios, KPIs
- **PRODUCT**: Business segments, product lines, services
- **PERSON**: Executives, analysts, board members
- **LOCATION**: Markets, regions, headquarters, facilities
- **EVENT**: Mergers, acquisitions, earnings releases, regulatory changes
- **RISK_FACTOR**: Business risks, market risks, operational risks
- **REGULATION**: Laws, standards, compliance requirements

## 🔗 Financial Relationship Types

- **OWNS**: Ownership relationships (parent-subsidiary)
- **REPORTS**: Reporting relationships (company reports metric)
- **COMPETES_WITH**: Competitive relationships
- **SUPPLIES_TO**: Supply chain relationships
- **AFFECTS**: Impact relationships (factor affects performance)
- **LOCATED_IN**: Geographic relationships
- **EMPLOYED_BY**: Employment relationships
- **REGULATED_BY**: Regulatory oversight
- **CAUSES**: Causal relationships
- **CORRELATES_WITH**: Correlation relationships
- **DEPENDS_ON**: Dependency relationships

## 🎛️ Query Parameters

```python
@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "hybrid"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    top_k: int = 60
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000
```

## 📈 Performance & Features

### Advanced Capabilities
- **Graph-Based Indexing**: Entities and relationships stored in NetworkX graph
- **Dual-Level Retrieval**: Local entity-focused + Global theme-based
- **Smart Deduplication**: Merges identical entities and relationships
- **Financial Chunking**: Respects financial document structure
- **Async Support**: Full async API for scalable operations
- **Persistent Storage**: JSON-based storage with automatic save/load

### Memory & Performance
- **Incremental Processing**: Add documents without rebuilding
- **Efficient Retrieval**: Key-value storage for fast access
- **Graph Algorithms**: NetworkX for sophisticated graph operations
- **Vector Search**: Sentence transformers for semantic similarity

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
export FINANCIAL_RAG_WORKING_DIR="./data"  # Optional
```

### Working Directory Structure
```
working_dir/
├── entities.json          # Entity storage
├── relationships.json     # Relationship storage
├── entity_kv.json        # Entity key-value descriptions
├── relation_kv.json      # Relationship key-value descriptions
├── chunks.json           # Text chunks metadata
└── embeddings.npy        # Vector embeddings
```

## 🧪 Testing

Run the comprehensive demo to see all features:
```bash
python run.py
```

The demo will:
1. Process multiple financial documents
2. Build knowledge graph with entities and relationships
3. Test all retrieval modes (local, global, hybrid, mix)
4. Show knowledge graph statistics
5. Demonstrate context extraction

##  When to Use Each Mode

### Local Mode
- **Use for**: Specific company data, exact metrics, entity-focused queries
- **Example**: "What was Apple's iPhone revenue in Q3?"
- **Retrieval**: Focuses on specific entities and their direct relationships

### Global Mode  
- **Use for**: Thematic analysis, industry trends, cross-company patterns
- **Example**: "How do supply chain issues affect tech companies?"
- **Retrieval**: Explores themes and relationships across multiple entities

### Hybrid Mode
- **Use for**: Complex analysis requiring both specific data and broader context
- **Example**: "Compare Apple and Microsoft's cloud strategies and revenue impact"
- **Retrieval**: Combines local precision with global context

### Mix Mode
- **Use for**: Comprehensive analysis integrating multiple information sources
- **Example**: "What are the emerging competitive dynamics in cloud computing?"
- **Retrieval**: Integrates knowledge graph traversal with vector similarity

### Naive Mode
- **Use for**: Simple similarity search without graph intelligence
- **Example**: Quick keyword-based search
- **Retrieval**: Basic vector similarity search

## 🚀 Next Steps

This core system provides the foundation for:
1. **Web Interface**: Build Gradio/Streamlit UI
2. **API Service**: FastAPI wrapper for production deployment
3. **Advanced Analytics**: Graph visualization and analysis tools
4. **Domain Extensions**: Extend to other financial document types
5. **Performance Optimization**: Caching, indexing, and scaling improvements

## 📝 License

This project demonstrates advanced RAG architecture patterns for financial document analysis.

---

**Built with sophisticated graph strategies following LightRAG's advanced architecture** 