# Buy-From-Egypt API Integration Guide

This guide provides detailed information for developers on how to integrate with the Buy-From-Egypt recommendation system API.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Recommendation API](#recommendation-api)
   - [API Base URL](#api-base-url)
   - [Core Endpoints](#core-endpoints)
   - [Data Models](#data-models)
   - [Example Requests](#example-requests)
4. [Integration Best Practices](#integration-best-practices)
5. [Error Handling](#error-handling)
6. [Performance Considerations](#performance-considerations)
7. [Deployment Notes](#deployment-notes)

## Overview

The Buy-From-Egypt recommendation system API provides developers with access to an advanced recommendation engine specifically designed for Egyptian businesses and products. The system leverages:

1. **Collaborative Filtering**: Recommends items based on user interaction patterns
2. **Content-Based Filtering**: Uses business and product metadata to find relevant matches
3. **Dwell Time Analysis**: Considers user engagement time to identify compelling content
4. **Hybrid Approach**: Combines multiple methods for optimal recommendations

## Authentication

Currently, no authentication is required for development. In production, the API will integrate with the main platform's authentication mechanism.

## Recommendation API

### API Base URL

Development: `http://localhost:8000`  
Production: `https://api.buyfromegypt.com/recommendations`

### Core Endpoints

#### Health Check

```
GET /
```

**Response:**
```json
{
### Core Endpoints

#### API Health Check

```
GET /
GET /api/health
```

**Description:** Verify if the API is running and check the status of the recommendation engine.

**Response:**
```json
{
  "status": "success",
  "message": "Buy from Egypt Recommendation API is running",
  "data": {
    "version": "1.0.0"
  }
}
```

#### Post Recommendations

```
POST /api/recommendations/posts
```

**Description:** Get personalized post recommendations for users.

**Query Parameters:**
- `user_id` (optional): User ID for personalized recommendations
- `num_recommendations` (optional, default: 10): Number of recommendations
- `include_similar_rated` (optional, default: false): Include posts similar to ones rated highly
- `force_refresh` (optional, default: false): Force refresh the recommendations cache

**Request Body:**
```json
{
  "preferred_industries": ["Textiles", "Handicrafts"],
  "preferred_supplier_type": "Manufacturer",
  "business_size": "Small",
  "location": "Cairo",
  "keywords": ["handmade", "traditional"]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Post recommendations generated successfully",
  "data": {
    "recommendations": [
      {
        "PostID": "123",
        "PostTitle": "Traditional Egyptian Textiles",
        "Industry": "Textiles",
        "CompanyName": "Cairo Crafts",
        "Score": 0.95,
        "RecommendationReason": "Based on your preferences and browsing history"
      }
    ],
    "user_id": "user_123",
    "recommendation_type": "post",
    "recommendation_reason": "Based on your preferences and browsing history",
    "generated_at": "2023-05-10T12:30:45.123456"
  }
}
```

#### Product Recommendations

```
POST /api/recommendations/products
```

**Description:** Get personalized product recommendations for marketplace.

**Query Parameters:**
- `user_id` (optional): User ID for personalized recommendations
- `num_recommendations` (optional, default: 10): Number of recommendations
- `force_refresh` (optional, default: false): Force refresh the recommendations cache

**Request Body:**
```json
{
  "preferred_industries": ["Food", "Handicrafts"],
  "price_range": "medium",
  "keywords": ["spices", "traditional"]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Product recommendations generated successfully",
  "data": {
    "recommendations": [
      {
        "ProductID": "1001",
        "Description": "Egyptian Food Product 1",
        "Category": "Food",
        "UnitPrice": 25.99,
        "Score": 0.9,
        "RecommendationReason": "Matches your selected criteria"
      }
    ],
    "user_id": "user_123",
    "recommendation_type": "product",
    "recommendation_reason": "Based on your preferences and purchase history",
    "generated_at": "2023-05-10T12:30:45.123456"
  }
}
```

#### Record User Interactions

```
POST /api/interactions
```

**Description:** Record a user interaction with a post or product.

**Request Body:**
```json
{
  "user_id": "user_123",
  "item_id": "post_456",
  "item_type": "post",
  "interaction_type": "view",
  "value": 1.0,
  "dwell_time_seconds": 45,
  "timestamp": "2023-05-10T12:30:45.123456"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Interaction recorded successfully",
  "data": {
    "user_id": "user_123",
    "item_id": "post_456",
    "type": "view"
  }
}
```

#### Get User Interaction History

```
GET /api/interactions/user/{user_id}
```

**Description:** Get interaction history for a specific user.

**Path Parameters:**
- `user_id`: The unique user identifier

**Query Parameters:**
- `item_type` (optional): Filter by item type (post or product)
- `interaction_type` (optional): Filter by interaction type
- `limit` (optional, default: 50): Maximum number of interactions to return

**Response:**
```json
{
  "status": "success",
  "message": "Retrieved 2 interactions for user user_123",
  "data": {
    "interactions": [
      {
        "UserID": "user_123",
        "ItemID": "post_456",
        "ItemType": "post",
        "InteractionType": "view",
        "Value": 1.0,
        "DwellTimeSeconds": 45,
        "Timestamp": "2023-05-10T12:30:45.123456"
      }
    ]
  }
}
```

### Data Models

#### UserInput Model

The `UserInput` model is used to pass user preferences for recommendation requests:

```json
{
  "preferred_industries": ["Textiles", "Handicrafts"],
  "preferred_supplier_type": "Manufacturer",
  "business_size": "Small",
  "location": "Cairo", 
  "price_range": "medium", 
  "keywords": ["handmade", "traditional"]
}
```

#### UserInteraction Model

The `UserInteraction` model is used to record user interactions:

```json
{
  "user_id": "user_123",  
  "item_id": "post_456",  
  "item_type": "post",  // or "product"
  "interaction_type": "view",  // "view", "like", "rate", "save", "share", "comment"
  "value": 1.0,  // optional, value for ratings
  "dwell_time_seconds": 45,  // optional, time spent viewing
  "timestamp": "2023-05-10T12:30:45.123456"  // optional, defaults to current time
}
```

## Integration Best Practices

### Recommendation Implementation

1. **Progressive Enhancement**:
   - Start with basic recommendations without user IDs
   - Enhance with user IDs for personalized recommendations once users are logged in
   - Further enhance with explicit user preferences and interaction tracking

2. **Client-Side Implementation**:
   - Cache recommendations for better performance
   - Update recommendations after significant user interactions
   - Implement client-side fallbacks for offline usage

3. **Server-Side Implementation**:
   - Batch recommendations for multiple users when possible
   - Cache recommendations server-side
   - Implement retry logic with exponential backoff

### Interaction Tracking Implementation

1. **Timing and Frequency**:
   - Record views after minimum engagement time (e.g., 5+ seconds)
   - Batch interaction records to reduce API calls
   - Track dwell time accurately on both mobile and desktop

2. **Privacy Considerations**:
   - Inform users about data collection
   - Allow users to opt out of personalized recommendations
   - Follow data protection regulations

## Error Handling

The API uses standard HTTP status codes:

- 200 OK: Successful request
- 400 Bad Request: Invalid input data
- 404 Not Found: Resource not found
- 500 Internal Server Error: Server-side error
- 503 Service Unavailable: Recommendation engine unavailable

Error responses follow the format:

```json
{
  "status_code": 400,
  "detail": "Error message describing the issue"
}
```

## Performance Considerations

- Use the force_refresh parameter sparingly to leverage the built-in caching
- Implement client-side caching for recommendations
- Batch interaction records when possible
- Consider using debouncing for dwell time tracking

## Deployment Notes

### Development Environment

```bash
# Start the API server locally
cd /path/to/buy-from-egypt-api
python -m api.main
```

The API will be available at http://localhost:8000

### Production Deployment

For production deployment, consider:

1. Using a proper WSGI server like Gunicorn
2. Setting up proper authentication
3. Implementing rate limiting
4. Using a reverse proxy like Nginx

Example production start command:

```bash
gunicorn api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
        "Timestamp": "2023-05-10T12:30:45.123456"
      }
    ]
  }
}
```

**Response:**
```json
{
  "gdp_growth": 4.35,
  "inflation": 5.04,
  "population_growth": 1.73,
  "tourism_sensitivity": 0.85,
  "economic_sentiment": "positive",
  "major_exports": [
    "textiles",
    "agricultural products",
    "petroleum products",
    "furniture"
  ],
  "seasonal_factors": {
    "current_season": "winter",
    "high_season_industries": [
      "textiles",
      "citrus exports"
    ]
  }
}
```

#### Export Recommendations as CSV

```
GET /export/recommendations/customer/{customer_id}?format=csv
```

**Parameters:**
- `customer_id` (path): The ID of the customer to export recommendations for
- `format` (query): Export format, either 'json' or 'csv'

**Response:**  
CSV file download or JSON response

### Data Models

#### Customer

```json
{
  "userId": "string",
  "name": "string",
  "email": "string",
  "type": "string",
  "industrySector": "string",
  "country": "string",
  "active": true
}
```

#### Product

```json
{
  "productId": "string",
  "name": "string",
  "description": "string",
  "price": 0,
  "currencyCode": "string",
  "categoryId": "string",
  "ownerId": "string",
  "rating": 0,
  "reviewCount": 0,
  "active": true,
  "available": true
}
```

#### Order

```json
{
  "orderId": "string",
  "importerId": "string",
  "exporterId": "string",
  "products": [
    "string"
  ],
  "totalPrice": 0,
  "currencyCode": "string",
  "createdAt": "2023-01-01T00:00:00Z"
}
```

### Example Requests

#### cURL

```bash
# Get customer recommendations
curl -X GET "http://localhost:8000/recommend/customer/10001" -H "accept: application/json" -H "X-API-Key: your_api_key_here"

# Get business recommendations
curl -X GET "http://localhost:8000/recommend/business/International%20Fruits%20Agriculture%20Egypt" -H "accept: application/json" -H "X-API-Key: your_api_key_here"
```

#### Python

```python
import requests

# Base URL
base_url = "http://localhost:8000"
headers = {"X-API-Key": "your_api_key_here"}

# Get customer recommendations
customer_id = "10001"
response = requests.get(f"{base_url}/recommend/customer/{customer_id}", headers=headers)
print(response.json())

# Get business recommendations
business_name = "International Fruits Agriculture Egypt"
response = requests.get(f"{base_url}/recommend/business/{business_name}", headers=headers)
print(response.json())
```

## Data Synchronization

### Sync User Data

```
POST /sync/user
```

**Request Body:**
```json
{
  "userId": "user123",
  "name": "Test User",
  "email": "test@example.com",
  "type": "IMPORTER",
  "industrySector": "Textiles",
  "country": "Egypt",
  "active": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "User Test User (user123) synced successfully",
  "syncedItem": {
    "userId": "user123",
    "name": "Test User",
    "type": "IMPORTER"
  }
}
```

### Sync Product Data

```
POST /sync/product
```

**Request Body:**
```json
{
  "productId": "prod123",
  "name": "Egyptian Cotton Fabric",
  "description": "High quality Egyptian cotton fabric for clothing",
  "price": 25.99,
  "currencyCode": "USD",
  "categoryId": "textiles",
  "ownerId": "business1",
  "rating": 4.8,
  "reviewCount": 24,
  "active": true,
  "available": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Product Egyptian Cotton Fabric (prod123) synced successfully",
  "syncedItem": {
    "productId": "prod123",
    "name": "Egyptian Cotton Fabric",
    "ownerId": "business1"
  }
}
```

### Sync Order Data

```
POST /sync/order
```

**Request Body:**
```json
{
  "orderId": "order123",
  "importerId": "user123",
  "exporterId": "business1",
  "products": ["prod123", "prod456"],
  "totalPrice": 149.99,
  "currencyCode": "USD",
  "createdAt": "2023-01-15T14:30:00Z"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Order order123 synced successfully",
  "syncedItem": {
    "orderId": "order123",
    "importerId": "user123",
    "productCount": 2
  }
}
```

## Chatbot API

### Chatbot API Base URL

Development: `http://localhost:8080`  
Production: `https://api.buyfromegypt.com/chatbot`

### Chatbot Endpoints

#### Health Check

```
GET /
```

**Response:**
```json
{
  "status": "Egyptian Business Chatbot API is running"
}
```

#### Chat Conversation

```
POST /chat
```

**Request Body:**
```json
{
  "user_id": "user123",
  "message": "What are the best seasons for exporting Egyptian citrus?",
  "conversation_id": "conv789"
}
```

**Response:**
```json
{
  "response": "The best seasons for exporting Egyptian citrus are from December to April. During this period, navel oranges, mandarins, and grapefruits are harvested and available for export. Egypt is one of the world's largest citrus exporters, with main markets in Russia, Saudi Arabia, Netherlands, and UAE. The Mediterranean climate in the Nile Delta and North Coast provides ideal growing conditions for high-quality citrus fruits.",
  "conversation_id": "conv789",
  "sources": [
    {
      "title": "Egyptian Agricultural Export Guide",
      "url": "https://example.com/egyptian-agriculture"
    }
  ]
}
```

#### Get Industry Information

```
GET /industry/{industry_name}
```

**Parameters:**
- `industry_name` (path): The name of the industry to get information about

**Response:**
```json
{
  "industry": "Textiles",
  "description": "Egypt's textile industry is one of its oldest and most established sectors...",
  "key_regions": ["Alexandria", "Greater Cairo", "Mahalla al-Kubra"],
  "annual_export_value": "3.8 billion USD",
  "growth_rate": 5.7,
  "key_products": ["Cotton fabrics", "Ready-made garments", "Home textiles"],
  "seasonal_factors": {
    "peak_seasons": ["Winter", "Spring"],
    "notes": "Higher demand for Egyptian cotton products typically occurs in Q4 and Q1..."
  }
}
```

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication failure
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error

Error responses follow this format:

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested resource was not found",
    "details": "Customer ID 99999 does not exist in the system"
  }
}
```

## Deployment Notes

### Requirements

- Python 3.8 or higher
- FastAPI
- PostgreSQL database (for production)
- Redis (optional, for caching)

### Environment Variables

The following environment variables need to be set:

- `API_KEY`: API key for authentication
- `DATABASE_URL`: Connection string for PostgreSQL (production only)
- `REDIS_URL`: Connection string for Redis (optional)
- `ENVIRONMENT`: Either "development" or "production"
- `GOOGLE_API_KEY`: API key for the Gemini chatbot

### Deployment Options

1. **Docker**: Docker images are available for both APIs
2. **Kubernetes**: Helm charts are provided for Kubernetes deployment
3. **Serverless**: AWS Lambda configurations are available

Contact the Buy-From-Egypt team for detailed deployment instructions. 