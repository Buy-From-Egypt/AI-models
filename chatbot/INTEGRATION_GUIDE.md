# Buy from Egypt Chatbot Integration Guide

This guide provides step-by-step instructions for integrating the Buy from Egypt chatbot into your backend system. It covers all necessary steps from initial setup to advanced usage patterns.

## Table of Contents

1. [System Overview](#system-overview)
2. [API Architecture](#api-architecture)
3. [Knowledge Base Structure](#knowledge-base-structure)
4. [Integration Approaches](#integration-approaches)
5. [Basic Integration](#basic-integration)
6. [Advanced Integration](#advanced-integration)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)
9. [Security Best Practices](#security-best-practices)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)

## System Overview

The Buy from Egypt chatbot is a specialized AI assistant that provides information about Egyptian industries, economy, business challenges, and customer support. It's designed to be integrated into e-commerce platforms, business websites, or mobile applications to provide users with relevant information about doing business in Egypt.

The chatbot system consists of three main components:

1. **Knowledge Base**: A structured repository of Egyptian business information stored in `knowledge.py`
2. **Chatbot Engine**: The core AI engine in `chatbot.py` that processes queries and generates responses
3. **API Layer**: A FastAPI implementation in `api.py` that provides RESTful endpoints

## API Architecture

The API is built using FastAPI and provides the following endpoint categories:

### Chat Endpoints
- **POST /chat**: Send a message and receive a synchronous response
- **POST /chat/reset**: Reset a conversation session
- **POST /send**: Send a message for asynchronous processing
- **GET /response/{request_id}**: Retrieve an asynchronous response

### Knowledge Endpoints
- **GET /industries**: Get information about all Egyptian industries
- **GET /industry/{industry_name}**: Get information about a specific industry
- **GET /regions**: Get information about all Egyptian regions
- **GET /region/{region_name}**: Get information about a specific region
- **GET /economy**: Get Egyptian economic indicators
- **GET /challenges**: Get information about business challenges

### Health Endpoints
- **GET /**: Basic health check
- **GET /health**: Detailed health information

## Knowledge Base Structure

The chatbot's knowledge base (`knowledge.py`) is organized into five main categories:

1. **Industry Sectors**: Information about 8 key Egyptian industries
   - Textiles, Agriculture, Food Processing, Handicrafts, Tourism, IT, Pharmaceuticals, Furniture
   - Each industry includes: description, key regions, challenges, opportunities, seasonal factors

2. **Economic Indicators**: Current economic data
   - GDP Growth, Inflation, Foreign Investment, Export Markets
   - Each indicator includes relevant metrics and impact information

3. **Business Challenges**: Common challenges faced by businesses
   - Regulatory, Financing, Operations, Market Access
   - Each challenge includes specific issues and potential solutions

4. **Customer Support**: Platform usage information
   - Platform Navigation, Buyer Support, Seller Support, Common Issues
   - Each area includes specific guidance and instructions

5. **Regional Characteristics**: Business information for 6 key Egyptian regions
   - Greater Cairo, Alexandria, Delta Region, Suez Canal Zone, Upper Egypt, Red Sea Coast
   - Each region includes: business density, infrastructure, key industries, advantages, challenges

## Integration Approaches

There are three main approaches to integrating the chatbot:

1. **Direct API Integration**: Your frontend communicates directly with the chatbot API
2. **Backend Proxy Integration**: Your backend proxies requests to the chatbot API
3. **Embedded Integration**: Run the chatbot as part of your application

This guide focuses primarily on the Backend Proxy Integration approach, which provides the most flexibility and control.

## Basic Integration

### Step 1: Set Up API Client

First, create an API client to handle communication with the chatbot API. Here's an example in Node.js:

```javascript
// chatbotClient.js
const axios = require('axios');

class ChatbotClient {
  constructor(baseUrl = 'http://localhost:8080') {
    this.baseUrl = baseUrl;
    this.axios = axios.create({
      baseURL: this.baseUrl,
      timeout: 30000, // 30 seconds timeout
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  async sendMessage(message, sessionId = null, userType = 'buyer', businessContext = null) {
    try {
      const response = await this.axios.post('/chat', {
        message,
        session_id: sessionId,
        user_type: userType,
        business_context: businessContext,
      });
      return response.data;
    } catch (error) {
      console.error('Error sending message to chatbot:', error);
      throw error;
    }
  }

  async resetConversation(sessionId) {
    try {
      const response = await this.axios.post(`/chat/reset?session_id=${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('Error resetting conversation:', error);
      throw error;
    }
  }
  
  async getIndustryInfo(industryName) {
    try {
      const response = await this.axios.get(`/industry/${encodeURIComponent(industryName)}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching information about ${industryName}:`, error);
      throw error;
    }
  }
  
  async getRegionInfo(regionName) {
    try {
      const response = await this.axios.get(`/region/${encodeURIComponent(regionName)}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching information about ${regionName}:`, error);
      throw error;
    }
  }
  
  async getEconomicIndicators() {
    try {
      const response = await this.axios.get('/economy');
      return response.data;
    } catch (error) {
      console.error('Error fetching economic indicators:', error);
      throw error;
    }
  }
  
  async getBusinessChallenges() {
    try {
      const response = await this.axios.get('/challenges');
      return response.data;
    } catch (error) {
      console.error('Error fetching business challenges:', error);
      throw error;
    }
  }
}

module.exports = ChatbotClient;
```

### Step 2: Create API Routes in Your Backend

Next, create routes in your backend that will proxy requests to the chatbot API:

```javascript
// chatbotRoutes.js
const express = require('express');
const router = express.Router();
const ChatbotClient = require('./chatbotClient');

const chatbot = new ChatbotClient(process.env.CHATBOT_API_URL || 'http://localhost:8080');

// Route to send a message to the chatbot
router.post('/chat', async (req, res) => {
  try {
    const { message, sessionId, userType, businessContext } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    const response = await chatbot.sendMessage(
      message,
      sessionId,
      userType || 'buyer',
      businessContext
    );
    
    res.json(response);
  } catch (error) {
    console.error('Error in chat endpoint:', error);
    res.status(500).json({ 
      error: 'Failed to get response from chatbot',
      details: error.message
    });
  }
});

// Route to reset a conversation
router.post('/chat/reset', async (req, res) => {
  try {
    const { sessionId } = req.body;
    
    if (!sessionId) {
      return res.status(400).json({ error: 'Session ID is required' });
    }
    
    const response = await chatbot.resetConversation(sessionId);
    res.json(response);
  } catch (error) {
    console.error('Error in reset conversation endpoint:', error);
    res.status(500).json({ 
      error: 'Failed to reset conversation',
      details: error.message
    });
  }
});

// Route to get industry information
router.get('/industry/:name', async (req, res) => {
  try {
    const industryName = req.params.name;
    const response = await chatbot.getIndustryInfo(industryName);
    res.json(response);
  } catch (error) {
    console.error('Error in industry info endpoint:', error);
    res.status(500).json({ 
      error: 'Failed to get industry information',
      details: error.message
    });
  }
});

// Route to get region information
router.get('/region/:name', async (req, res) => {
  try {
    const regionName = req.params.name;
    const response = await chatbot.getRegionInfo(regionName);
    res.json(response);
  } catch (error) {
    console.error('Error in region info endpoint:', error);
    res.status(500).json({ 
      error: 'Failed to get region information',
      details: error.message
    });
  }
});

// Route to get economic indicators
router.get('/economy', async (req, res) => {
  try {
    const response = await chatbot.getEconomicIndicators();
    res.json(response);
  } catch (error) {
    console.error('Error in economic indicators endpoint:', error);
    res.status(500).json({ 
      error: 'Failed to get economic indicators',
      details: error.message
    });
  }
});

// Route to get business challenges
router.get('/challenges', async (req, res) => {
  try {
    const response = await chatbot.getBusinessChallenges();
    res.json(response);
  } catch (error) {
    console.error('Error in business challenges endpoint:', error);
    res.status(500).json({ 
      error: 'Failed to get business challenges',
      details: error.message
    });
  }
});

module.exports = router;
```

### Step 3: Integrate Routes into Your Express App

```javascript
// app.js
const express = require('express');
const chatbotRoutes = require('./chatbotRoutes');

const app = express();
app.use(express.json());

// Mount chatbot routes
app.use('/api/chatbot', chatbotRoutes);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

### Step 4: Handle Sessions

Implement session management to maintain conversation context:

```javascript
// sessionManager.js
const sessions = new Map();

function getOrCreateSession(userId) {
  if (!sessions.has(userId)) {
    sessions.set(userId, {
      sessionId: null,
      lastActivity: Date.now()
    });
  }
  return sessions.get(userId);
}

function updateSession(userId, sessionId) {
  const session = getOrCreateSession(userId);
  session.sessionId = sessionId;
  session.lastActivity = Date.now();
  return session;
}

// Clean up inactive sessions (run periodically)
function cleanupSessions() {
  const now = Date.now();
  const inactivityThreshold = 24 * 60 * 60 * 1000; // 24 hours
  
  for (const [userId, session] of sessions.entries()) {
    if (now - session.lastActivity > inactivityThreshold) {
      sessions.delete(userId);
    }
  }
}

module.exports = {
  getOrCreateSession,
  updateSession,
  cleanupSessions
};
```

## Advanced Integration

### Using Business Context for Tailored Responses

The chatbot can provide more relevant responses when given business context:

```javascript
// Example of including business context
const businessContext = {
  industry: 'Textiles',
  region: 'Greater Cairo',
  name: 'Cairo Textiles Ltd'
};

const response = await chatbot.sendMessage(
  'What opportunities are there for my business?',
  sessionId,
  'seller',
  businessContext
);
```

The business context can include:
- `industry`: One of the industries in the knowledge base
- `region`: One of the regions in the knowledge base
- `name`: Your business name (for personalized responses)

### Using Asynchronous Processing for Complex Queries

For complex queries that might take longer to process:

```javascript
// asyncChatClient.js
async function sendAsyncMessage(message, sessionId, userType, businessContext) {
  try {
    // Send the message for async processing
    const response = await axios.post(`${baseUrl}/send`, {
      message,
      session_id: sessionId,
      user_type: userType,
      business_context: businessContext
    });
    
    const requestId = response.data.request_id;
    
    // Poll for the response
    let result = null;
    let attempts = 0;
    const maxAttempts = 10;
    
    while (!result && attempts < maxAttempts) {
      attempts++;
      await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
      
      const statusResponse = await axios.get(`${baseUrl}/response/${requestId}`);
      
      if (statusResponse.status === 200 && statusResponse.data.status !== 'pending') {
        result = statusResponse.data;
        break;
      }
    }
    
    return result;
  } catch (error) {
    console.error('Error in async message processing:', error);
    throw error;
  }
}
```

### Building a Domain-Specific Interface

For specialized use cases, you can create domain-specific interfaces:

```javascript
// egyptianBusinessAdvisor.js
class EgyptianBusinessAdvisor {
  constructor(chatbotClient) {
    this.chatbot = chatbotClient;
  }
  
  async getExportOpportunities(industry) {
    const industryInfo = await this.chatbot.getIndustryInfo(industry);
    return {
      industry: industry,
      opportunities: industryInfo.opportunities,
      exportMarkets: await this.chatbot.getEconomicIndicators().then(data => data.Export_Markets)
    };
  }
  
  async getBusinessChallengesByRegion(region) {
    const [regionInfo, challenges] = await Promise.all([
      this.chatbot.getRegionInfo(region),
      this.chatbot.getBusinessChallenges()
    ]);
    
    return {
      region: region,
      regionalChallenges: regionInfo.challenges,
      commonChallenges: challenges
    };
  }
  
  async getIndustryRecommendationsByRegion(region) {
    const regionInfo = await this.chatbot.getRegionInfo(region);
    const recommendedIndustries = [];
    
    for (const industry of regionInfo.key_industries) {
      try {
        const industryInfo = await this.chatbot.getIndustryInfo(industry);
        recommendedIndustries.push({
          name: industry,
          description: industryInfo.description,
          opportunities: industryInfo.opportunities
        });
      } catch (error) {
        console.error(`Error fetching info for ${industry}:`, error);
      }
    }
    
    return {
      region: region,
      recommendedIndustries: recommendedIndustries
    };
  }
}
```

## Error Handling

Implement robust error handling to manage API failures:

```javascript
async function sendMessageWithRetry(message, sessionId, userType, businessContext) {
  const maxRetries = 3;
  let retries = 0;
  
  while (retries < maxRetries) {
    try {
      return await chatbot.sendMessage(message, sessionId, userType, businessContext);
    } catch (error) {
      retries++;
      
      // Check if it's a recoverable error
      if (error.response && error.response.status >= 500) {
        console.log(`Retry ${retries}/${maxRetries} after server error`);
        await new Promise(resolve => setTimeout(resolve, 1000 * retries));
      } else {
        // Non-recoverable error, rethrow
        throw error;
      }
    }
  }
  
  throw new Error(`Failed after ${maxRetries} retries`);
}
```

## Performance Considerations

### Connection Pooling

For high-traffic applications, implement connection pooling:

```javascript
const http = require('http');
const https = require('https');

const axiosInstance = axios.create({
  baseURL: baseUrl,
  timeout: 30000,
  httpAgent: new http.Agent({ keepAlive: true }),
  httpsAgent: new https.Agent({ keepAlive: true }),
  maxSockets: 100 // Adjust based on your needs
});
```

### Caching Knowledge Base Responses

The knowledge base data is relatively static, so caching can improve performance:

```javascript
const NodeCache = require('node-cache');
const cache = new NodeCache({ stdTTL: 3600 }); // 1 hour TTL

async function getIndustryInfoWithCache(industryName) {
  const cacheKey = `industry:${industryName}`;
  
  // Check if data is in cache
  const cachedData = cache.get(cacheKey);
  if (cachedData) {
    return cachedData;
  }
  
  // If not in cache, fetch from API
  const data = await chatbot.getIndustryInfo(industryName);
  
  // Store in cache
  cache.set(cacheKey, data);
  
  return data;
}
```

## Security Best Practices

### Request Validation

Validate all input before sending to the chatbot API:

```javascript
const Joi = require('joi');

const chatSchema = Joi.object({
  message: Joi.string().required().max(500),
  userType: Joi.string().valid('buyer', 'seller').default('buyer'),
  businessContext: Joi.object({
    industry: Joi.string(),
    region: Joi.string(),
    name: Joi.string().max(100)
  }).optional(),
  sessionId: Joi.string().optional()
});

router.post('/chat', async (req, res) => {
  // Validate request
  const { error, value } = chatSchema.validate(req.body);
  if (error) {
    return res.status(400).json({ error: error.details[0].message });
  }
  
  // Proceed with validated data
  const { message, userType, businessContext, sessionId } = value;
  // ...
});
```

### Rate Limiting

Implement rate limiting to prevent abuse:

```javascript
const rateLimit = require('express-rate-limit');

const chatbotLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again after 15 minutes'
});

router.use(chatbotLimiter);
```

## Testing

### Unit Testing

Example unit test for the chatbot client using Jest:

```javascript
// chatbotClient.test.js
const axios = require('axios');
const ChatbotClient = require('./chatbotClient');

jest.mock('axios');

describe('ChatbotClient', () => {
  let client;
  
  beforeEach(() => {
    client = new ChatbotClient('http://test-api.com');
    axios.create.mockReturnValue(axios);
  });
  
  test('sendMessage should call the API correctly', async () => {
    // Mock response
    const mockResponse = {
      data: {
        response: 'Test response',
        session_id: '123',
        sources: ['Test'],
        processing_time: 0.1
      }
    };
    
    axios.post.mockResolvedValue(mockResponse);
    
    // Call the method
    const result = await client.sendMessage('Hello', '123', 'buyer', { industry: 'Test' });
    
    // Check if axios was called correctly
    expect(axios.post).toHaveBeenCalledWith('/chat', {
      message: 'Hello',
      session_id: '123',
      user_type: 'buyer',
      business_context: { industry: 'Test' }
    });
    
    // Check if the result is correct
    expect(result).toEqual(mockResponse.data);
  });
});
```

### Integration Testing

Example integration test:

```javascript
// integration.test.js
const request = require('supertest');
const app = require('./app');

describe('Chatbot Integration', () => {
  test('POST /api/chatbot/chat should return a response', async () => {
    const response = await request(app)
      .post('/api/chatbot/chat')
      .send({
        message: 'Tell me about Egyptian textiles',
        userType: 'buyer'
      })
      .set('Accept', 'application/json');
    
    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('response');
    expect(response.body).toHaveProperty('session_id');
  });
  
  test('GET /api/chatbot/industry/Textiles should return industry info', async () => {
    const response = await request(app)
      .get('/api/chatbot/industry/Textiles')
      .set('Accept', 'application/json');
    
    expect(response.status).toBe(200);
    expect(response.body).toHaveProperty('description');
    expect(response.body).toHaveProperty('key_regions');
  });
});
```

## Troubleshooting

### Common Issues and Solutions

1. **Connection Refused**
   - Check if the chatbot API is running with `python run.py api`
   - Verify the API URL is correct (default: http://localhost:8080)
   - Ensure network connectivity between your backend and the API server

2. **Slow Responses**
   - Consider using the asynchronous endpoints for complex queries
   - Implement caching for frequently accessed knowledge base information
   - Check if the Gemini API is available (the chatbot will fall back to rule-based responses if not)

3. **Session Management Issues**
   - Ensure you're storing and reusing the session_id correctly
   - Check if sessions are being expired prematurely
   - Verify that session IDs are being correctly associated with users

4. **Arabic Text Handling**
   - The chatbot detects Arabic text and responds with a message to use English
   - If you need Arabic support, you'll need to implement your own translation layer

### Logging and Monitoring

Implement comprehensive logging:

```javascript
const winston = require('winston');

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  defaultMeta: { service: 'chatbot-integration' },
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

// Log all requests to the chatbot
async function sendMessage(message, sessionId, userType, businessContext) {
  try {
    logger.info('Sending message to chatbot', { 
      message, 
      sessionId, 
      userType,
      hasBusinessContext: !!businessContext 
    });
    
    const startTime = Date.now();
    const response = await chatbot.sendMessage(message, sessionId, userType, businessContext);
    const duration = Date.now() - startTime;
    
    logger.info('Received response from chatbot', { 
      sessionId: response.session_id,
      processingTime: response.processing_time,
      totalDuration: duration,
      hasSources: !!response.sources
    });
    
    return response;
  } catch (error) {
    logger.error('Error sending message to chatbot', {
      message,
      sessionId,
      error: error.message,
      stack: error.stack
    });
    throw error;
  }
}
```

## Conclusion

By following this guide, you should now have a robust integration of the Buy from Egypt chatbot into your backend system. The integration provides your users with access to specialized knowledge about Egyptian business and economy while maintaining conversation context across interactions.

Remember to monitor the performance of your integration and adjust caching, connection pooling, and other optimizations as needed based on your specific usage patterns.