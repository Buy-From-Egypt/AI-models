#!/usr/bin/env python3
"""
Warm-up script for the Buy From Egypt AI API

This script helps with pre-loading the recommendation model to avoid cold start delays
when first accessing the API. It sends a series of API requests to load the model into memory,
improving response times for subsequent user requests.
"""

import sys
import time
import logging
import requests
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 60  # Generous timeout for first model load
TEST_USER_IDS = ["1000", "1001", "1002"]  # Sample user IDs for warming up
TEST_BUSINESS = "Tech Egypt"  # Sample business for warming up

def check_api_health():
    """Check if the API is up and running"""
    try:
        logger.info("🔍 Checking API health...")
        response = requests.get(f"{API_BASE_URL}/", timeout=10)
        if response.status_code == 200:
            logger.info("✅ API is running")
            return True
        else:
            logger.error(f"❌ API returned status code {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        return False

def warm_up_customer_recommendations():
    """Warm up the customer recommendation endpoints"""
    logger.info("🔄 Warming up customer recommendation endpoints...")
    
    for user_id in TEST_USER_IDS:
        try:
            start_time = time.time()
            response = requests.get(
                f"{API_BASE_URL}/recommend/customer/{user_id}", 
                params={"num_recommendations": 3},
                timeout=TIMEOUT
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                recommendations = response.json().get("recommended_products", [])
                logger.info(f"✅ Loaded recommendations for user {user_id} in {elapsed:.2f}s ({len(recommendations)} items)")
            else:
                logger.warning(f"⚠️ Failed to load recommendations for user {user_id}: {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"❌ Error warming up customer recommendations: {e}")

def warm_up_business_recommendations():
    """Warm up the business recommendation endpoints"""
    logger.info("🔄 Warming up business recommendation endpoints...")
    
    try:
        start_time = time.time()
        response = requests.get(
            f"{API_BASE_URL}/recommend/business/{TEST_BUSINESS}", 
            params={
                "num_product_recommendations": 3,
                "num_partner_recommendations": 3
            },
            timeout=TIMEOUT
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            products = data.get("recommended_products", [])
            partners = data.get("recommended_partners", [])
            logger.info(f"✅ Loaded business recommendations in {elapsed:.2f}s ({len(products)} products, {len(partners)} partners)")
        else:
            logger.warning(f"⚠️ Failed to load business recommendations: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"❌ Error warming up business recommendations: {e}")

def warm_up_economic_context():
    """Warm up the economic context endpoint"""
    logger.info("🔄 Warming up economic context...")
    
    try:
        start_time = time.time()
        response = requests.get(
            f"{API_BASE_URL}/egyptian-economic-context",
            timeout=TIMEOUT
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            logger.info(f"✅ Loaded economic context in {elapsed:.2f}s")
        else:
            logger.warning(f"⚠️ Failed to load economic context: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"❌ Error warming up economic context: {e}")

def main():
    """Main entry point for the warm-up script"""
    logger.info("🚀 Starting API warm-up process...")
    
    # First check if API is available
    if not check_api_health():
        logger.error("❌ API is not available. Please start the API server first.")
        return 1
    
    # Warm up each endpoint type
    warm_up_customer_recommendations()
    warm_up_business_recommendations()
    warm_up_economic_context()
    
    logger.info("✅ API warm-up completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
