# Buy-From-Egypt Recommendation API Documentation

## Overview

The Buy-From-Egypt Recommendation API provides personalized recommendations for Egyptian businesses and customers, leveraging machine learning models that incorporate collaborative filtering, content-based filtering, and dwell time tracking.

## Base URL

```
http://localhost:8000
```

## Authentication

The API is designed to integrate with the authentication mechanism of the main Buy-From-Egypt platform. Currently, no authentication is required for development purposes.

## API Endpoints

### Health Check

```
GET /
```

**Description:** Verify if the API is running.

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

```
GET /api/health
```

**Description:** Detailed health check for the API and recommendation engine.

**Response:**
```json
{
  "status": "success",
  "message": "Recommendation system health status",
  "data": {
    "api_available": true,
    "engine_initialized": true,
    "version": "1.0.0",
    "timestamp": "2023-05-10T12:30:45.123456"
  }
}
```

### Post Recommendations

```
POST /api/recommendations/posts
```

**Description:** Get personalized post recommendations for users.

**Query Parameters:**
- `user_id` (optional, string) - User ID for personalized recommendations
- `num_recommendations` (optional, integer, default: 10) - Number of recommendations to return (min: 1, max: 50)
- `include_similar_rated` (optional, boolean, default: false) - Include posts similar to ones the user rated highly
- `force_refresh` (optional, boolean, default: false) - Force refresh the recommendations cache

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

### Product Recommendations

```
POST /api/recommendations/products
```

**Description:** Get personalized product recommendations for marketplace.

**Query Parameters:**
- `user_id` (optional, string) - User ID for personalized recommendations
- `num_recommendations` (optional, integer, default: 10) - Number of recommendations to return (min: 1, max: 50)
- `force_refresh` (optional, boolean, default: false) - Force refresh the recommendations cache

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
    }
  ],
  "egyptian_context": {
    "gdp_growth": 4.35,
    "inflation": 5.04,
    "population_growth": 1.73,
    "tourism_sensitivity": 0.85,
    "economic_stability_index": 0.65,
    "trade_balance": -0.12,
    "is_winter_tourism_season": 1,
    "is_ramadan_season": 0,
    "current_date": "2025-05-03"
  },
  "industry_weights": {
    "Textiles": 0.15,
    "Agriculture": 0.18,
    "Spices": 0.12,
    "Fruits & Vegetables": 0.15,
    "Chemicals": 0.08,
    "Pharmaceuticals": 0.07,
    "Electronics": 0.06,
    "Machinery": 0.05,
    "Metals": 0.08,
    "Automobiles": 0.03,
    "Seafood": 0.06,
    "Manufacturing": 0.10
  }
}
```

### Egyptian Economic Context

```
GET /egyptian-economic-context
```

**Description:** Get current Egyptian economic context data used by the recommendation system.

**Response:**
```json
{
  "gdp_growth": 4.35,
  "inflation": 5.04,
  "population_growth": 1.73,
  "tourism_sensitivity": 0.85,
  "economic_stability_index": 0.65,
  "trade_balance": -0.12,
  "is_winter_tourism_season": 1,
  "is_ramadan_season": 0,
  "current_date": "2025-05-03"
}
```

### Export Customer Recommendations

```
GET /export/recommendations/customer/{customer_id}
```

**Description:** Export product recommendations for a customer in JSON or CSV format.

**Path Parameters:**
- `customer_id` - The unique customer identifier

**Query Parameters:**
- `num_recommendations` (optional, integer, default: 20) - Number of recommendations to export (min: 1, max: 100)
- `format` (optional, string, default: "json") - Export format, either 'json' or 'csv'

**Response:**
A file download containing the recommendations in the specified format.

### Export Business Recommendations

```
GET /export/recommendations/business/{business_name}
```

**Description:** Export product and business partnership recommendations in JSON or CSV format.

**Path Parameters:**
- `business_name` - The name of the business

**Query Parameters:**
- `num_product_recommendations` (optional, integer, default: 20) - Number of product recommendations (min: 1, max: 100)
- `num_partner_recommendations` (optional, integer, default: 10) - Number of business partner recommendations (min: 1, max: 50)
- `format` (optional, string, default: "json") - Export format, either 'json' or 'csv'

**Response:**
A file download containing the recommendations in the specified format.

## Data Synchronization Endpoints

These endpoints allow synchronizing data from the Buy-From-Egypt platform with the recommendation system.

### Sync User

```
POST /sync/user
```

**Description:** Synchronize user data with the recommendation system.

**Request Body:**
```json
{
  "userId": "user123",
  "name": "Cairo Textiles",
  "email": "info@cairotextiles.com",
  "type": "EXPORTER",
  "industrySector": "Textiles",
  "country": "Egypt",
  "active": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "User Cairo Textiles (user123) synced successfully",
  "syncedItem": {
    "userId": "user123",
    "name": "Cairo Textiles",
    "type": "EXPORTER"
  }
}
```

### Sync Product

```
POST /sync/product
```

**Description:** Synchronize product data with the recommendation system.

**Request Body:**
```json
{
  "productId": "prod123",
  "name": "Egyptian Cotton Shirt",
  "description": "High-quality cotton shirt made in Egypt",
  "price": 29.99,
  "currencyCode": "USD",
  "categoryId": "category123",
  "ownerId": "user456",
  "rating": 4.5,
  "reviewCount": 120,
  "active": true,
  "available": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Product Egyptian Cotton Shirt (prod123) synced successfully",
  "syncedItem": {
    "productId": "prod123",
    "name": "Egyptian Cotton Shirt",
    "ownerId": "user456"
  }
}
```

### Sync Order

```
POST /sync/order
```

**Description:** Synchronize order data with the recommendation system.

**Request Body:**
```json
{
  "orderId": "order123",
  "importerId": "importer123",
  "exporterId": "exporter456",
  "products": ["prod1", "prod2", "prod3"],
  "totalPrice": 150.75,
  "currencyCode": "USD",
  "createdAt": "2025-05-03T12:30:45Z"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Order order123 synced successfully",
  "syncedItem": {
    "orderId": "order123",
    "importerId": "importer123",
    "productCount": 3
  }
}
```

### Retrain Models

```
POST /admin/retrain
```

**Description:** Trigger a full retraining of all recommendation models.

**Response:**
```json
{
  "success": true,
  "message": "Model retraining initiated. This may take some time to complete.",
  "syncedItem": null
}
```

## Error Handling

The API returns appropriate HTTP status codes:

- 200: Successful operation
- 400: Bad request (missing or invalid parameters)
- 404: Resource not found (customer ID or business name not in database)
- 500: Server error

Error responses include a detailed message:

```json
{
  "detail": "Customer ID 99999 not found in model data."
}
```

## Integration with Buy-From-Egypt Platform

The API integrates with the Buy-From-Egypt platform's database schema:

- Customer recommendations use the `User.userId` as the customer ID
- Business recommendations use the `User.name` as the business name
- Product recommendations include `Product.productId` as the stock code and `Product.name` as the description
- Orders are used to build the interaction matrix for collaborative filtering

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `/api/docs`
- ReDoc: `/api/redoc` 