# Backend Implementation Guide for Buy-From-Egypt Recommendation System

This document provides detailed guidance for integrating the Buy-From-Egypt recommendation system with your backend services.

## System Overview

The Buy-From-Egypt recommendation system is built on a hybrid approach that combines:

1. **Collaborative filtering** based on user interactions
2. **Content-based recommendations** using business and product metadata
3. **User interaction and dwell time tracking** to identify engaging content

## API Integration

### Core API Endpoints

The following endpoints are essential for basic integration:

#### Post Recommendations
```
POST /api/recommendations/posts
```
Use this endpoint to get personalized post recommendations for users based on their preferences and browsing history.

#### Product Recommendations
```
POST /api/recommendations/products
```
Use this endpoint to get personalized product recommendations for the marketplace.

#### User Interactions
```
POST /api/interactions
```
Use this endpoint to record user interactions with posts or products.

### Health Check Endpoints

```
GET /
GET /api/health
```
Use these endpoints to verify the API is running and check the status of the recommendation engine.

## Authentication

Currently, no authentication is required for development purposes. In production, the API will be integrated with the main Buy-From-Egypt platform's authentication mechanism.

## Data Streaming and Events

To maintain up-to-date recommendations:

1. **Track user interactions** by sending data to `/api/interactions` for each view, like, rate, save, share, or comment
2. **Include dwell time** whenever possible to improve recommendation quality

## Implementation Steps

### 1. Initial Setup

1. Deploy the recommendation API using the provided Docker configuration or by running the server directly
2. Verify the API is running with a health check (`GET /`)

### 2. Basic Integration

Implement API calls for the core recommendation endpoints:

```javascript
// Example in Node.js with axios
const axios = require('axios');

async function getPostRecommendations(userId) {
  try {
    const response = await axios.post('http://localhost:8000/api/recommendations/posts', 
      {
        // Optional user preferences
        preferred_industries: ['Textiles', 'Handicrafts']
      },
      {
        params: { user_id: userId }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    return null;
  }
}
```

### 3. User Interaction Tracking

Implement event tracking for user interactions:

```javascript
// Example in Node.js with axios
async function trackUserInteraction(userId, itemId, interactionType, dwellTime = null) {
  try {
    await axios.post('http://localhost:8000/api/interactions', {
      user_id: userId,
      item_id: itemId,
      item_type: 'post',  // or 'product'
      interaction_type: interactionType, // 'view', 'like', 'rate', 'save', 'share', 'comment'
      dwell_time_seconds: dwellTime,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error tracking interaction:', error);
  }
}
```

### 4. Recommended UI Integration

For an optimal user experience, implement the following UI elements:

1. **Recommendations Widget**
   - Display personalized post and product recommendations
   - Include a reason why items are being recommended 
   - Update recommendations based on user interactions

2. **Interaction Tracking**
   - Track post/product views (client-side)
   - Implement dwell time tracking for measuring engagement
   - Capture likes, ratings, shares, and other engagement metrics

### 5. Dwell Time Measurement

Implement dwell time tracking to improve recommendation quality:

```javascript
// Client-side dwell time tracking example
let startViewTime;
let itemId;

function startDwellTimeTracking(id) {
  itemId = id;
  startViewTime = Date.now();
}

function endDwellTimeTracking(userId) {
  if (!startViewTime || !itemId) return;
  
  const dwellTimeSeconds = Math.round((Date.now() - startViewTime) / 1000);
  
  // Only track meaningful engagement (e.g., > 5 seconds)
  if (dwellTimeSeconds > 5) {
    trackUserInteraction(userId, itemId, 'view', dwellTimeSeconds);
  }
  
  // Reset tracking
  startViewTime = null;
  itemId = null;
}
```

## Error Handling

### Error Responses

All API errors return a standard format:

```json
{
  "status_code": 400,
  "detail": "Error message describing the issue"
}
```

Common error codes:
- 400: Bad Request - invalid input
- 404: Not Found - resource not found
- 500: Internal Server Error - server-side issue
- 503: Service Unavailable - recommendation engine not available

### Fallback Strategy

Implement fallbacks in case the recommendation service is unavailable:

1. Cache previous recommendation results client-side
2. Display popular or featured items as a default
3. Track failed API calls and retry with exponential backoff

## Performance Considerations

### Caching

The recommendation API uses internal caching with a 5-minute expiration. You can force a cache refresh by adding `?force_refresh=true` to any recommendation request.

### Batch Processing

Consider consolidating multiple user interaction events before sending them to the API to reduce network traffic.

## Monitoring Integration

Monitor the recommendation system performance by:

1. Tracking API response times
2. Monitoring recommendation click-through rates
3. Comparing user engagement with and without personalized recommendations

## Next Steps

Contact the Buy-From-Egypt development team for:

1. Production API keys and quota increases
2. Custom recommendation algorithm tuning
3. Advanced integration support
```
      "score": 0.85
    },
    {
      "business_id": 67,
      "name": "Alexandria Apparel",
      "category": "Clothing",
      "score": 0.79
    }
  ]
}
```

## Error Handling

The API uses standard HTTP status codes:

- 200: Success
- 400: Bad request
- 401: Unauthorized 
- 404: Resource not found
- 500: Server error

Each error response includes a message and error code:

```json
{
  "error": {
    "code": "INVALID_USER_ID",
    "message": "The provided user ID does not exist"
  }
}
```

## Rate Limiting

API requests are rate-limited to 100 requests per minute per API key. The response headers include rate limiting information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1621872000
```

## Testing with Postman

A Postman collection is available in `/docs/postman/buyFromEgypt.json`. Import this collection to test all API endpoints.

## Performance Considerations

- The recommendation API typically responds within 200ms
- For high-traffic scenarios, implement caching of recommendation results (TTL: 1 hour recommended)
- Batch synchronization requests when possible to reduce API calls 