#!/usr/bin/env python3
"""
FastAPI Backend for Financial RAG System
Provides REST API endpoints for document processing and querying
"""

import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
import logging
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import your RAG system
try:
    from financial_rag import AdvancedMemoryEfficientFinancialRAG, QueryParam
    from pdf_processor import FinancialPDFProcessor
except ImportError as e:
    print(f"❌ Error importing RAG modules: {e}")
    print("Make sure financial_rag.py and pdf_processor.py are in the same directory.")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial RAG API",
    description="Advanced Financial Document Analysis and Query System",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
rag_system: Optional[AdvancedMemoryEfficientFinancialRAG] = None
pdf_processor = FinancialPDFProcessor()
processing_status = {}
temp_dir = Path("./temp_uploads")
temp_dir.mkdir(exist_ok=True)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    top_k: int = 20

class ProcessingStatus(BaseModel):
    status: str
    message: str
    progress: float
    documents_processed: int
    total_documents: int
    current_document: Optional[str] = None
    error: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    processing_time: float
    mode: str
    sources_used: int

class SystemStats(BaseModel):
    total_documents: int
    total_chunks: int
    total_entities: int
    total_relationships: int
    memory_usage_mb: float
    system_ready: bool

# Initialize RAG system
async def initialize_rag_system():
    """Initialize the RAG system on startup"""
    global rag_system
    
    try:
        logger.info("🏦 Initializing Financial RAG System...")
        
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("❌ OPENAI_API_KEY environment variable not set")
            return False
        
        # Initialize RAG system
        rag_system = AdvancedMemoryEfficientFinancialRAG(
            working_dir="./rag_data",
            max_memory_mb=4096,  # 4GB limit for server
            openai_api_key=api_key
        )
        
        logger.info("✅ Financial RAG System initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    await initialize_rag_system()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Financial RAG API is running",
        "system_ready": rag_system is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status", response_model=SystemStats)
async def get_system_status():
    """Get system status and statistics"""
    if not rag_system:
        return SystemStats(
            total_documents=0,
            total_chunks=0,
            total_entities=0,
            total_relationships=0,
            memory_usage_mb=0,
            system_ready=False
        )
    
    try:
        stats = rag_system.get_comprehensive_stats()
        return SystemStats(
            total_documents=stats.get("processed_documents", 0),
            total_chunks=stats.get("vector_storage", {}).get("total_chunks", 0),
            total_entities=stats.get("knowledge_graph", {}).get("total_entities", 0),
            total_relationships=stats.get("knowledge_graph", {}).get("total_relationships", 0),
            memory_usage_mb=stats.get("memory_usage", {}).get("current_mb", 0),
            system_ready=True
        )
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return SystemStats(
            total_documents=0,
            total_chunks=0,
            total_entities=0,
            total_relationships=0,
            memory_usage_mb=0,
            system_ready=False
        )

@app.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload and process documents"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Generate processing ID
    process_id = str(uuid.uuid4())
    
    # Initialize processing status
    processing_status[process_id] = ProcessingStatus(
        status="uploading",
        message="Uploading documents...",
        progress=0.0,
        documents_processed=0,
        total_documents=len(files)
    )
    
    # Validate files
    valid_files = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            processing_status[process_id].error = f"Invalid file type: {file.filename}. Only PDF files are supported."
            processing_status[process_id].status = "error"
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")
        valid_files.append(file)
    
    # Save uploaded files temporarily
    saved_files = []
    try:
        for file in valid_files:
            file_path = temp_dir / f"{process_id}_{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            saved_files.append(file_path)
    except Exception as e:
        processing_status[process_id].error = f"Failed to save files: {str(e)}"
        processing_status[process_id].status = "error"
        raise HTTPException(status_code=500, detail="Failed to save uploaded files")
    
    # Start background processing
    background_tasks.add_task(process_documents_background, process_id, saved_files)
    
    return {"process_id": process_id, "message": f"Started processing {len(files)} documents"}

async def process_documents_background(process_id: str, file_paths: List[Path]):
    """Background task to process documents"""
    try:
        # Update status
        processing_status[process_id].status = "processing"
        processing_status[process_id].message = "Processing documents..."
        processing_status[process_id].progress = 10.0
        
        total_files = len(file_paths)
        processed_count = 0
        
        for i, file_path in enumerate(file_paths):
            try:
                # Update current document
                processing_status[process_id].current_document = file_path.name
                processing_status[process_id].message = f"Processing {file_path.name}..."
                
                # Process PDF
                logger.info(f"Processing PDF: {file_path.name}")
                doc = pdf_processor.process_pdf_file(str(file_path))
                
                if not doc:
                    logger.warning(f"Failed to process PDF: {file_path.name}")
                    continue
                
                # Insert into RAG system
                result = await rag_system.ainsert(doc.text_content, doc.file_name)
                
                if result["status"] == "success":
                    processed_count += 1
                    logger.info(f"Successfully processed: {file_path.name}")
                else:
                    logger.warning(f"Failed to insert {file_path.name}: {result.get('error', 'Unknown error')}")
                
                # Update progress
                progress = 10.0 + (i + 1) / total_files * 85.0  # 10% to 95%
                processing_status[process_id].progress = progress
                processing_status[process_id].documents_processed = processed_count
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                continue
            
            finally:
                # Clean up file
                try:
                    file_path.unlink()
                except:
                    pass
        
        # Final status
        if processed_count > 0:
            processing_status[process_id].status = "completed"
            processing_status[process_id].message = f"Successfully processed {processed_count} out of {total_files} documents"
            processing_status[process_id].progress = 100.0
        else:
            processing_status[process_id].status = "error"
            processing_status[process_id].message = "No documents were successfully processed"
            processing_status[process_id].error = "Processing failed for all documents"
        
    except Exception as e:
        logger.error(f"Background processing error: {e}")
        processing_status[process_id].status = "error"
        processing_status[process_id].message = "Processing failed"
        processing_status[process_id].error = str(e)

@app.get("/processing/{process_id}", response_model=ProcessingStatus)
async def get_processing_status(process_id: str):
    """Get processing status for a specific process"""
    if process_id not in processing_status:
        raise HTTPException(status_code=404, detail="Process ID not found")
    
    return processing_status[process_id]

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the processed documents"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Validate mode
        valid_modes = ["local", "global", "hybrid", "mix", "naive"]
        if request.mode not in valid_modes:
            request.mode = "hybrid"
        
        # Create query parameters
        param = QueryParam(
            mode=request.mode,
            top_k=min(request.top_k, 50),  # Limit top_k for performance
            max_token_for_text_unit=3000,
            max_token_for_global_context=3000,
            max_token_for_local_context=3000
        )
        
        # Process query
        start_time = datetime.now()
        response = await rag_system.aquery(request.query, param)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        return QueryResponse(
            response=response,
            processing_time=processing_time,
            mode=request.mode,
            sources_used=min(request.top_k, 20)  # Approximate
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.delete("/reset")
async def reset_system():
    """Reset the system and clear all data"""
    global rag_system, processing_status
    
    try:
        # Clear processing status
        processing_status.clear()
        
        # Clear temp files
        for file_path in temp_dir.glob("*"):
            try:
                file_path.unlink()
            except:
                pass
        
        # Reinitialize RAG system
        await initialize_rag_system()
        
        return {"message": "System reset successfully"}
        
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "rag_system_ready": rag_system is not None,
        "active_processes": len(processing_status),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )