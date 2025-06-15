#!/usr/bin/env python3
"""
Run script for the Buy from Egypt chatbot.

This script can start either the FastAPI server or the Streamlit UI.
"""

import os
import sys
import argparse
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_api():
    """Run the FastAPI server"""
    logger.info("Starting the Buy from Egypt chatbot API...")
    try:
        import uvicorn
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=8080,
            reload=True
        )
    except ImportError:
        logger.error("uvicorn not found. Please install it with 'pip install uvicorn'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        sys.exit(1)

def run_ui():
    """Run the Streamlit UI"""
    logger.info("Starting the Buy from Egypt chatbot UI...")
    try:
        subprocess.run(["streamlit", "run", "streamlit_app.py"])
    except FileNotFoundError:
        logger.error("streamlit not found. Please install it with 'pip install streamlit'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting Streamlit UI: {e}")
        sys.exit(1)

def main():
    """Parse arguments and run the appropriate component"""
    parser = argparse.ArgumentParser(description="Run the Buy from Egypt chatbot")
    parser.add_argument(
        "component",
        choices=["api", "ui", "all"],
        help="Which component to run: api, ui, or all (both)"
    )
    
    args = parser.parse_args()
    
    if args.component == "api":
        run_api()
    elif args.component == "ui":
        run_ui()
    elif args.component == "all":
        # Run API in a separate process
        import threading
        api_thread = threading.Thread(target=run_api)
        api_thread.daemon = True
        api_thread.start()
        
        # Run UI in the main process
        run_ui()

if __name__ == "__main__":
    main() 