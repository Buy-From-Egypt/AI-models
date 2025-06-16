# Buy from Egypt Chatbot API Documentation

## Overview

The Buy from Egypt Chatbot API provides a set of endpoints for interacting with an AI-powered assistant specialized in Egyptian business knowledge. The API offers chat functionality, conversation management, and access to structured knowledge about Egyptian industries, economy, business challenges, and regional characteristics.

## Base URL

```
http://localhost:8080
```

## Authentication

Currently, the API does not require authentication. In a production environment, appropriate authentication mechanisms should be implemented.

## API Endpoints

### Health Check Endpoints

#### Check API Status

```
GET /
```

Returns basic health information about the API.

**Response:**
```json
{
  "status": "ok",
  "message": "Buy from Egypt Chatbot API is running"
}
```

#### Detailed Health Check

```
GET /health
```

Returns detailed health information about the chatbot service.

**Response:**
```json
{
  "status": "ok",
  "api_available": true,
  "model_initialized": true,
  "version": "1.0.0",
  "uptime": "3h 24m 15s"
}
```

### Chat Endpoints

#### Send Chat Message

```
POST /chat
```

Send a message to the chatbot and receive a response.

**Request Body:**
```json
{
  "message": "Tell me about the textile industry in Egypt",
  "user_type": "buyer",
  "business_context": {
    "industry": "Textiles",
    "region": "Greater Cairo",
    "name": "My Business"
  },
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| message | string | Yes | User's message to the chatbot |
| user_type | string | No | Type of user ("buyer" or "seller") |
| business_context | object | No | Optional business-specific context |
| session_id | string | No | Session ID for conversation tracking |

**Response:**
```json
{
  "response": "Egypt's textile industry is renowned for high-quality cotton production, with major exports to Europe and MENA regions. The industry is concentrated in Greater Cairo, the Nile Delta, and Alexandria. Some challenges faced by the industry include global competition, raw material price fluctuations, and modernization needs. However, there are opportunities in European market expansion, sustainable textile production, and value-added garment manufacturing.",
  "sources": ["Industry: Textiles"],
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "processing_time": 0.856
}
```

| Field | Type | Description |
|-------|------|-------------|
| response | string | The chatbot's response |
| sources | array | Sources of information used in the response |
| session_id | string | Session ID for conversation tracking |
| processing_time | number | Processing time in seconds |

#### Reset Conversation

```
POST /chat/reset
```

Reset the conversation history for a specific session.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| session_id | string | Yes | The session ID to reset |

**Response:**
```json
{
  "status": "success",
  "message": "Conversation reset successfully"
}
```

#### Asynchronous Message Processing

```
POST /send
```

Send a message for asynchronous processing.

**Request Body:**
Same as the `/chat` endpoint.

**Response:**
```json
{
  "request_id": "7f4b3a21-c6e8-4c19-9f8a-b12d456e7890",
  "status": "processing"
}
```

#### Retrieve Async Response

```
GET /response/{request_id}
```

Retrieve the response for an asynchronously processed message.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| request_id | string | The ID of the request |

**Response (when processing):**
```json
{
  "status": "processing"
}
```

**Response (when complete):**
```json
{
  "response": "Egypt's textile industry is renowned for high-quality cotton production...",
  "sources": ["Industry: Textiles"],
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "processing_time": 0.856
}
```

### Knowledge Endpoints

#### Get All Industries

```
GET /industries
```

Get information about all Egyptian industries.

**Response:**
```json
{
  "industries": {
    "Textiles": {
      "description": "Egypt's textile industry is renowned for high-quality cotton production...",
      "key_regions": ["Greater Cairo", "Nile Delta", "Alexandria"],
      "challenges": ["Global competition", "Raw material price fluctuations", "Modernization needs"],
      "opportunities": ["European market expansion", "Sustainable textile production", "Value-added garment manufacturing"],
      "seasonal_factors": ["Winter tourism increases demand for textile souvenirs", "Ramadan increases demand for home textiles"]
    },
    "Agriculture": {
      "description": "Agricultural businesses focus on fruits, vegetables, and cotton...",
      "key_regions": ["Nile Delta", "Upper Egypt", "Fayoum"],
      "challenges": ["Water scarcity", "Climate change impacts", "Cold chain logistics"],
      "opportunities": ["Organic farming certification", "Export to premium markets", "Agricultural technology adoption"],
      "seasonal_factors": ["Harvest seasons vary by crop", "Ramadan affects food consumption patterns"]
    },
    // Other industries...
  }
}
```

#### Get Specific Industry

```
GET /industry/{industry_name}
```

Get information about a specific Egyptian industry.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| industry_name | string | The name of the industry |

**Response:**
```json
{
  "industry": "Textiles",
  "description": "Egypt's textile industry is renowned for high-quality cotton production...",
  "key_regions": ["Greater Cairo", "Nile Delta", "Alexandria"],
  "challenges": ["Global competition", "Raw material price fluctuations", "Modernization needs"],
  "opportunities": ["European market expansion", "Sustainable textile production", "Value-added garment manufacturing"],
  "seasonal_factors": ["Winter tourism increases demand for textile souvenirs", "Ramadan increases demand for home textiles"]
}
```

#### Get All Regions

```
GET /regions
```

Get information about all Egyptian regions.

**Response:**
```json
{
  "regions": {
    "Greater Cairo": {
      "business_density": "Very High",
      "infrastructure": "Well-developed with occasional congestion challenges",
      "key_industries": ["Services", "Manufacturing", "Technology", "Retail"],
      "business_advantages": "Access to largest consumer market, government offices, and business services",
      "challenges": "High competition, property costs, and traffic congestion"
    },
    "Alexandria": {
      "business_density": "High",
      "infrastructure": "Good port facilities and transportation links",
      "key_industries": ["Shipping", "Manufacturing", "Tourism", "Petrochemicals"],
      "business_advantages": "Mediterranean port access, industrial zones, and lower costs than Cairo",
      "challenges": "Seasonal tourism fluctuations and infrastructure modernization needs"
    },
    // Other regions...
  }
}
```

#### Get Specific Region

```
GET /region/{region_name}
```

Get information about a specific Egyptian region.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| region_name | string | The name of the region |

**Response:**
```json
{
  "region": "Greater Cairo",
  "business_density": "Very High",
  "infrastructure": "Well-developed with occasional congestion challenges",
  "key_industries": ["Services", "Manufacturing", "Technology", "Retail"],
  "business_advantages": "Access to largest consumer market, government offices, and business services",
  "challenges": "High competition, property costs, and traffic congestion"
}
```

#### Get Economic Indicators

```
GET /economy
```

Get information about Egyptian economic indicators.

**Response:**
```json
{
  "GDP_Growth": {
    "current": "5.6% (2023)",
    "trend": "Positive growth despite global economic challenges",
    "sectors_driving_growth": ["Tourism", "Construction", "Natural Gas", "ICT"],
    "challenges": ["Inflation", "Currency fluctuations", "Public debt management"]
  },
  "Inflation": {
    "current": "Approximately 30% (2023)",
    "impact_on_business": "Increased operational costs, pricing challenges, inventory management difficulties",
    "consumer_impact": "Reduced purchasing power, shift to essential goods, price sensitivity",
    "mitigation_strategies": ["Hedging currency exposure", "Local sourcing", "Value-based pricing"]
  },
  "Foreign_Investment": {
    "trend": "Increasing in targeted sectors",
    "key_sectors": ["Energy", "Infrastructure", "Manufacturing", "ICT"],
    "incentives": ["Special economic zones", "Tax benefits", "Repatriation guarantees"],
    "challenges": ["Regulatory complexity", "Currency convertibility", "Bureaucratic procedures"]
  },
  "Export_Markets": {
    "primary_destinations": ["EU countries", "Arab states", "United States", "African markets"],
    "growing_markets": ["East Asia", "Eastern Europe", "Sub-Saharan Africa"],
    "export_challenges": ["Quality certification", "Logistics costs", "Trade barriers"],
    "export_support": ["Export councils", "Trade agreements", "Export financing programs"]
  }
}
```

#### Get Business Challenges

```
GET /challenges
```

Get information about common business challenges in Egypt.

**Response:**
```json
{
  "Regulatory": {
    "licensing": "Complex business licensing procedures requiring multiple approvals",
    "taxation": "Evolving tax regulations and digital tax reporting requirements",
    "customs": "Import/export documentation and customs clearance procedures",
    "solutions": ["Regulatory consultants", "Digital compliance tools", "Industry association support"]
  },
  "Financing": {
    "access_to_credit": "Challenges in securing business loans with favorable terms",
    "working_capital": "Managing cash flow with extended payment terms",
    "investment": "Finding investors for business expansion",
    "solutions": ["SME loan programs", "Invoice factoring", "Business angel networks"]
  },
  "Operations": {
    "supply_chain": "Reliability and cost of domestic and international logistics",
    "workforce": "Finding skilled labor and managing retention",
    "technology": "Digital transformation and technology adoption",
    "solutions": ["Supply chain optimization services", "Technical training programs", "Technology implementation partners"]
  },
  "Market_Access": {
    "customer_acquisition": "Reaching target customers cost-effectively",
    "competition": "Differentiating from local and international competitors",
    "pricing": "Setting competitive yet profitable pricing in inflationary environment",
    "solutions": ["Digital marketing strategies", "Value proposition development", "Market research services"]
  }
}
```

## Error Handling

The API uses standard HTTP status codes to indicate the success or failure of a request:

- 200: Success
- 400: Bad request (e.g., missing required parameters)
- 404: Not found (e.g., industry or region not found)
- 500: Internal server error
- 503: Service unavailable (e.g., chatbot not initialized)

Error responses follow this format:

```json
{
  "error": "Error message",
  "details": "Additional error details"
}
```

## Rate Limiting

Currently, the API does not implement rate limiting. In a production environment, appropriate rate limiting should be implemented to prevent abuse.

## Versioning

The current API version is 1.0.0. The version is included in the detailed health check response.

## Examples

### Example 1: Chat Request

```bash
curl -X POST "http://localhost:8080/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about the textile industry in Egypt",
    "user_type": "buyer"
  }'
```

### Example 2: Get Industry Information

```bash
curl -X GET "http://localhost:8080/industry/Textiles"
```

### Example 3: Reset Conversation

```bash
curl -X POST "http://localhost:8080/chat/reset?session_id=550e8400-e29b-41d4-a716-446655440000"
``` 