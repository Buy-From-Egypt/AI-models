#!/usr/bin/env python3
"""
Streamlit UI for the Buy from Egypt chatbot.

This Streamlit app provides a user-friendly interface for interacting with the chatbot.
"""
import os
import json
import uuid
import time
import requests
import streamlit as st
from typing import Dict, List, Any, Optional

# Configure page
st.set_page_config(
    page_title="Buy from Egypt Chatbot",
    page_icon="🇪🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8080")
DEFAULT_USER_TYPE = "buyer"

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "user_type" not in st.session_state:
    st.session_state.user_type = DEFAULT_USER_TYPE
    
if "business_context" not in st.session_state:
    st.session_state.business_context = {}
    
if "api_available" not in st.session_state:
    st.session_state.api_available = True

# Helper functions
def check_api_health() -> bool:
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def reset_conversation() -> None:
    """Reset the conversation history"""
    try:
        if st.session_state.api_available:
            requests.post(
                f"{API_URL}/chat/reset",
                params={"session_id": st.session_state.session_id},
                timeout=5
            )
        st.session_state.messages = []
        # Generate a new session ID
        st.session_state.session_id = str(uuid.uuid4())
        st.success("Conversation reset successfully!")
    except Exception as e:
        st.error(f"Error resetting conversation: {e}")

def send_message(message: str) -> Dict[str, Any]:
    """Send a message to the chatbot API and get a response"""
    try:
        if not st.session_state.api_available:
            st.error("API is not available. Please check the API server.")
            return {
                "response": "I'm sorry, but I'm currently unable to connect to the API server. Please make sure it's running.",
                "sources": None,
                "session_id": st.session_state.session_id,
                "processing_time": 0
            }
            
        payload = {
            "message": message,
            "user_type": st.session_state.user_type,
            "business_context": st.session_state.business_context,
            "session_id": st.session_state.session_id
        }
        
        with st.spinner("Thinking..."):
            start_time = time.time()
            response = requests.post(f"{API_URL}/chat", json=payload, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Make sure we update the session ID for future requests
                if "session_id" in response_data:
                    st.session_state.session_id = response_data["session_id"]
                    
                return response_data
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                return {
                    "response": f"Error: {response.status_code} - {response.text}",
                    "sources": None,
                    "session_id": st.session_state.session_id,
                    "processing_time": response_time
                }
    except requests.exceptions.ConnectionError:
        st.session_state.api_available = False
        st.error("Cannot connect to the API server. Please check if it's running.")
        return {
            "response": "I'm sorry, but I'm currently unable to connect to the API server. Please make sure it's running.",
            "sources": None,
            "session_id": st.session_state.session_id,
            "processing_time": 0
        }
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {
            "response": f"Error: {str(e)}",
            "sources": None,
            "session_id": st.session_state.session_id,
            "processing_time": 0
        }

def get_industries() -> List[str]:
    """Get the list of industries from the API"""
    try:
        if not st.session_state.api_available:
            return ["Textiles", "Agriculture", "Food Processing", "Handicrafts", 
                    "Tourism", "Information Technology", "Pharmaceuticals", "Furniture"]
            
        response = requests.get(f"{API_URL}/industries", timeout=5)
        if response.status_code == 200:
            return list(response.json()["industries"].keys())
        else:
            return ["Textiles", "Agriculture", "Food Processing", "Handicrafts", 
                    "Tourism", "Information Technology", "Pharmaceuticals", "Furniture"]
    except Exception:
        return ["Textiles", "Agriculture", "Food Processing", "Handicrafts", 
                "Tourism", "Information Technology", "Pharmaceuticals", "Furniture"]

def get_regions() -> List[str]:
    """Get the list of regions from the API"""
    try:
        if not st.session_state.api_available:
            return ["Greater Cairo", "Alexandria", "Delta Region", 
                    "Suez Canal Zone", "Upper Egypt", "Red Sea Coast"]
            
        response = requests.get(f"{API_URL}/regions", timeout=5)
        if response.status_code == 200:
            return list(response.json()["regions"].keys())
        else:
            return ["Greater Cairo", "Alexandria", "Delta Region", 
                    "Suez Canal Zone", "Upper Egypt", "Red Sea Coast"]
    except Exception:
        return ["Greater Cairo", "Alexandria", "Delta Region", 
                "Suez Canal Zone", "Upper Egypt", "Red Sea Coast"]

# Check API health
st.session_state.api_available = check_api_health()

# UI Layout
st.title("🇪🇬 Buy from Egypt Chatbot")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # API status indicator
    if st.session_state.api_available:
        st.success("API Connected ✅")
    else:
        st.error("API Disconnected ❌")
        st.info("Make sure the API is running with: `python run.py api`")
        if st.button("Retry Connection"):
            st.session_state.api_available = check_api_health()
            st.rerun()
    
    # User type selection
    st.subheader("User Type")
    user_type = st.radio(
        "Select your user type:",
        options=["buyer", "seller"],
        index=0 if st.session_state.user_type == "buyer" else 1,
        help="Select whether you are a buyer or seller to get more relevant responses."
    )
    
    if user_type != st.session_state.user_type:
        st.session_state.user_type = user_type
    
    # Business context
    st.subheader("Business Context")
    
    # Get industries and regions
    industries = get_industries()
    regions = get_regions()
    
    # Industry selection
    industry = st.selectbox(
        "Industry:",
        options=[""] + industries,
        index=0,
        help="Select your industry to get more specific information."
    )
    
    # Region selection
    region = st.selectbox(
        "Region:",
        options=[""] + regions,
        index=0,
        help="Select your region to get location-specific information."
    )
    
    # Business name
    business_name = st.text_input(
        "Business Name (optional):",
        value=st.session_state.business_context.get("name", ""),
        help="Enter your business name for more personalized responses."
    )
    
    # Update business context
    business_context = {}
    if industry:
        business_context["industry"] = industry
    if region:
        business_context["region"] = region
    if business_name:
        business_context["name"] = business_name
    
    st.session_state.business_context = business_context
    
    # Reset conversation button
    if st.button("Reset Conversation"):
        reset_conversation()
    
    # About section
    st.subheader("About")
    st.markdown("""
    This chatbot provides information about Egyptian economy, 
    business challenges, and customer support for the Buy from Egypt platform.
    
    **Topics you can ask about:**
    - Egyptian industries and economy
    - Business challenges and solutions
    - Platform navigation assistance
    - Buyer and seller support
    - Regional business information
    """)

# Chat interface
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    st.write(", ".join(message["sources"]))
            
            # Display processing time if available
            if message["role"] == "assistant" and "processing_time" in message and message["processing_time"] > 0:
                st.caption(f"Response time: {message['processing_time']:.2f}s")

# Chat input
user_input = st.chat_input("Ask about Egyptian business, economy, or platform help...")

if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get chatbot response
    response_data = send_message(user_input)
    
    # Add assistant message to chat
    assistant_message = {
        "role": "assistant", 
        "content": response_data["response"],
        "sources": response_data.get("sources"),
        "processing_time": response_data.get("processing_time", 0)
    }
    st.session_state.messages.append(assistant_message)
    
    # Display assistant message
    with st.chat_message("assistant"):
        st.write(response_data["response"])
        
        # Display sources if available
        if response_data.get("sources"):
            with st.expander("Sources"):
                st.write(", ".join(response_data["sources"]))
        
        # Display processing time
        if "processing_time" in response_data:
            st.caption(f"Response time: {response_data['processing_time']:.2f}s")

# First-time welcome message
if not st.session_state.messages:
    with st.chat_message("assistant"):
        welcome_message = (
            "Welcome to Buy from Egypt. I'm your specialized assistant for Egyptian business information. "
            "I can provide insights on Egyptian industries, economic trends, business opportunities, "
            "and platform support for both buyers and sellers. How may I assist you today?"
        )
        st.write(welcome_message)
        
        # Add welcome message to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_message
        })
