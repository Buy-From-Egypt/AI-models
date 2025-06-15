"""
Buy from Egypt Chatbot API

A FastAPI implementation of the Buy from Egypt chatbot API,
providing endpoints for chat, health checks, and knowledge retrieval.
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import logging
from chatbot import Chatbot
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize chatbot instance
chatbot = None

# Define request and response models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message to the chatbot")
    user_type: Optional[str] = Field(None, description="Type of user (buyer or seller)")
    business_context: Optional[Dict[str, Any]] = Field(None, description="Optional business-specific context")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation tracking")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The chatbot's response")
    sources: Optional[List[str]] = Field(None, description="Sources of information used in the response")
    session_id: str = Field(..., description="Session ID for conversation tracking")
    processing_time: float = Field(..., description="Processing time in seconds")

class StatusResponse(BaseModel):
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Additional information")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")

# Create FastAPI application
app = FastAPI(
    title="Buy from Egypt Chatbot API",
    description="""
    # Buy from Egypt Chatbot API
    
    An AI-powered chatbot API for the Buy from Egypt platform that provides
    assistance on Egyptian economy, business challenges, and customer support
    for both buyers and sellers.
    
    ## Features
    
    * Egyptian economy and industry knowledge
    * Business challenges and solutions
    * Platform navigation assistance
    * Buyer and seller support
    * Regional business information
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot on startup
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    global chatbot
    logger.info("Initializing Buy from Egypt chatbot...")
    try:
        chatbot = Chatbot()
        logger.info("Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}")

# Middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers and log requests"""
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"Request to {request.url.path} completed in {process_time:.4f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error processing request to {request.url.path}: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error", "details": str(e)},
        )

# Helper to check if chatbot is initialized
def get_chatbot():
    """Get the chatbot instance or raise an exception if not initialized"""
    if chatbot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chatbot service is not initialized yet. Please try again later."
        )
    return chatbot

# Main chat endpoint
@app.post(
    "/chat", 
    response_model=ChatResponse,
    responses={
        200: {"description": "Successful response from the chatbot"},
        400: {"description": "Bad request", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    },
    tags=["Chat"]
)
async def chat(request: ChatRequest, bot: Chatbot = Depends(get_chatbot)):
    """
    Get a response from the Buy from Egypt chatbot.
    
    This endpoint processes a user message and returns a response.
    You can include user type and business context to get more tailored responses.
    
    - **message**: User's message to the chatbot (required)
    - **user_type**: Type of user ('buyer' or 'seller')
    - **business_context**: Optional context about the business
    - **session_id**: Optional session ID for continuing conversations
    
    If you provide a session_id that exists, the conversation history will be maintained.
    If you don't provide a session_id, a new one will be created for you.
    """
    try:
        # Validate request
        if not request.message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No message provided in the request"
            )
        
        # Get response from chatbot
        response = await bot.get_response(
            query=request.message,
            user_type=request.user_type,
            business_context=request.business_context,
            session_id=request.session_id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )

# Reset conversation endpoint
@app.post(
    "/chat/reset",
    response_model=StatusResponse,
    responses={
        200: {"description": "Conversation history reset successfully"},
        404: {"description": "Session not found", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    },
    tags=["Chat"]
)
async def reset_conversation(
    session_id: str, 
    bot: Chatbot = Depends(get_chatbot)
):
    """
    Reset the conversation history for a specific session.
    
    This endpoint clears the conversation history for the specified session ID.
    
    - **session_id**: The session ID to reset (required)
    """
    result = bot.reset_conversation(session_id)
    
    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["message"]
        )
    
    return result

# Health check endpoint
@app.get(
    "/", 
    response_model=StatusResponse,
    tags=["Health"]
)
async def health_check():
    """
    Check if the Buy from Egypt chatbot API is running.
    
    This endpoint can be used to verify that the API is operational.
    """
    return {
        "status": "ok",
        "message": "Buy from Egypt Chatbot API is running"
    }

# Detailed health check endpoint
@app.get(
    "/health", 
    response_model=Dict[str, Any],
    tags=["Health"]
)
async def detailed_health(bot: Chatbot = Depends(get_chatbot)):
    """
    Get detailed health information about the chatbot.
    
    This endpoint provides information about the chatbot's configuration,
    API availability, and other relevant health metrics.
    """
    return {
        "status": "ok",
        "api_available": bot.api_available,
        "model_initialized": bot.model is not None,
        "active_sessions": len(bot.conversations),
        "version": "1.0.0"
    }

# Get industries endpoint
@app.get(
    "/industries", 
    response_model=Dict[str, Any],
    tags=["Knowledge"]
)
async def get_industries(bot: Chatbot = Depends(get_chatbot)):
    """
    Get information about Egyptian industries.
    
    This endpoint returns a list of industries and their details.
    """
    return {"industries": bot.knowledge["industries"]}

# Get industry details endpoint
@app.get(
    "/industry/{industry_name}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Industry information retrieved successfully"},
        404: {"description": "Industry not found", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    },
    tags=["Knowledge"]
)
async def get_industry_info(
    industry_name: str,
    bot: Chatbot = Depends(get_chatbot)
):
    """
    Get detailed information about a specific Egyptian industry.
    
    This endpoint returns detailed information about the specified industry.
    
    - **industry_name**: The name of the industry to get information about (required)
    """
    industries = bot.knowledge["industries"]
    
    if industry_name not in industries:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Industry '{industry_name}' not found"
        )
    
    return {"industry": industry_name, "details": industries[industry_name]}

# Get regions endpoint
@app.get(
    "/regions", 
    response_model=Dict[str, Any],
    tags=["Knowledge"]
)
async def get_regions(bot: Chatbot = Depends(get_chatbot)):
    """
    Get information about Egyptian regions.
    
    This endpoint returns a list of regions and their details.
    """
    return {"regions": bot.knowledge["regions"]}

# Get region details endpoint
@app.get(
    "/region/{region_name}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Region information retrieved successfully"},
        404: {"description": "Region not found", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    },
    tags=["Knowledge"]
)
async def get_region_info(
    region_name: str,
    bot: Chatbot = Depends(get_chatbot)
):
    """
    Get detailed information about a specific Egyptian region.
    
    This endpoint returns detailed information about the specified region.
    
    - **region_name**: The name of the region to get information about (required)
    """
    regions = bot.knowledge["regions"]
    
    if region_name not in regions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Region '{region_name}' not found"
        )
    
    return {"region": region_name, "details": regions[region_name]}

# Get economic indicators endpoint
@app.get(
    "/economy", 
    response_model=Dict[str, Any],
    tags=["Knowledge"]
)
async def get_economy(bot: Chatbot = Depends(get_chatbot)):
    """
    Get information about Egyptian economic indicators.
    
    This endpoint returns economic indicators and their details.
    """
    return {"economy": bot.knowledge["economy"]}

# Get business challenges endpoint
@app.get(
    "/challenges", 
    response_model=Dict[str, Any],
    tags=["Knowledge"]
)
async def get_challenges(bot: Chatbot = Depends(get_chatbot)):
    """
    Get information about common business challenges in Egypt.
    
    This endpoint returns business challenges and their details.
    """
    return {"challenges": bot.knowledge["business_challenges"]}

# Message queue to store pending requests
message_queue = {}

# Send message endpoint
@app.post(
    "/send", 
    response_model=Dict[str, str],
    responses={
        200: {"description": "Message received successfully"},
        400: {"description": "Bad request", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    },
    tags=["Chat"]
)
async def send_message(request: ChatRequest, bot: Chatbot = Depends(get_chatbot)):
    """
    Send a message to the chatbot and get a request ID.
    
    This endpoint accepts a message and returns a request ID that can be used
    to retrieve the response later.
    
    - **message**: User's message to the chatbot (required)
    - **user_type**: Type of user ('buyer' or 'seller')
    - **business_context**: Optional context about the business
    - **session_id**: Optional session ID for continuing conversations
    """
    try:
        # Validate request
        if not request.message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No message provided in the request"
            )
        
        # Generate a request ID
        request_id = str(uuid.uuid4())
        
        # Store the request in the queue
        message_queue[request_id] = {
            "request": request.dict(),
            "status": "pending",
            "response": None
        }
        
        # Process the message asynchronously
        import asyncio
        asyncio.create_task(process_message(request_id, bot))
        
        return {"request_id": request_id, "status": "pending"}
        
    except Exception as e:
        logger.error(f"Error in send endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )

# Get response endpoint
@app.get(
    "/response/{request_id}", 
    responses={
        200: {"description": "Response retrieved successfully"},
        202: {"description": "Response not ready yet"},
        404: {"description": "Request ID not found", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    },
    tags=["Chat"]
)
async def get_response(request_id: str):
    """
    Get the response for a previously sent message.
    
    This endpoint retrieves the response for a given request ID.
    
    - **request_id**: The request ID returned by the /send endpoint
    """
    try:
        # Check if the request ID exists
        if request_id not in message_queue:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Request ID {request_id} not found"
            )
        
        # Check if the response is ready
        if message_queue[request_id]["status"] == "pending":
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={"status": "pending", "message": "Response not ready yet"}
            )
        
        # Return the response
        return message_queue[request_id]["response"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_response endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving response: {str(e)}"
        )

async def process_message(request_id: str, bot: Chatbot):
    """Process a message asynchronously"""
    try:
        # Get the request from the queue
        request_data = message_queue[request_id]["request"]
        
        # Get response from chatbot
        response = await bot.get_response(
            query=request_data["message"],
            user_type=request_data["user_type"],
            business_context=request_data["business_context"],
            session_id=request_data["session_id"]
        )
        
        # Update the queue
        message_queue[request_id]["status"] = "completed"
        message_queue[request_id]["response"] = response
        
    except Exception as e:
        logger.error(f"Error processing message {request_id}: {e}")
        message_queue[request_id]["status"] = "error"
        message_queue[request_id]["response"] = {
            "error": str(e),
            "status": "error"
        }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle generic exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "details": str(exc)}
    )

# Main function to run the API
def main():
    """Run the API using uvicorn"""
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )

if __name__ == "__main__":
    main() 