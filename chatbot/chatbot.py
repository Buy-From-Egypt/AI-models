import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
import google.generativeai as genai
from dotenv import load_dotenv
from knowledge import EGYPTIAN_KNOWLEDGE
import re

# Load environment variables
load_dotenv()

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

# Configure the Gemini API
# Set API key directly since we can't use .env file
GOOGLE_API_KEY = "AIzaSyDCTrudTetYrvxtICEAafDK2BnraKiRoEg"
# Force fallback mode to avoid API issues
USE_FALLBACK_MODE = False
if not GOOGLE_API_KEY:
    logger.warning("No GOOGLE_API_KEY found in environment. Please set a valid API key.")
    logger.warning("You can get a Gemini API key from https://aistudio.google.com/")

class Chatbot:
    """
    Buy from Egypt Chatbot that provides information about Egyptian economy,
    business challenges, and customer support for both buyers and sellers.
    """
    
    def __init__(self):
        """Initialize the chatbot with API configuration and knowledge base."""
        self.knowledge = EGYPTIAN_KNOWLEDGE
        self.conversations = {}
        self.api_available = False
        self.model = None
        
        # Skip API initialization if in fallback mode
        if USE_FALLBACK_MODE:
            logger.info("Using fallback mode - skipping API initialization")
            return
            
        # Try to initialize the Gemini model
        if GOOGLE_API_KEY:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                
                # Try to get the best available model
                model_priority = [
                    'gemini-1.5-pro',
                    'gemini-1.5-flash',
                    'gemini-1.0-pro',
                    'gemini-pro',
                ]
                
                for model_name in model_priority:
                    try:
                        self.model = genai.GenerativeModel(
                            model_name,
                            generation_config={
                                "temperature": 0.2,
                                "top_p": 0.9,
                                "top_k": 40,
                                "max_output_tokens": 1024,
                            }
                        )
                        # Test the model with a simple query to ensure it works
                        test_response = self.model.generate_content("Hello")
                        if test_response:
                            self.api_available = True
                            logger.info(f"Successfully initialized model: {model_name}")
                            break
                    except Exception as e:
                        logger.warning(f"Could not initialize {model_name}: {e}")
                        continue
                
                if not self.api_available:
                    logger.error("Failed to initialize any Gemini model")
            except Exception as e:
                logger.error(f"Error configuring Gemini API: {e}")
    
    def is_arabic(self, text):
        """Check if the text contains Arabic characters"""
        # Arabic Unicode range
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        return bool(arabic_pattern.search(text))
    
    def generate_system_prompt(self, user_type=None, business_context=None):
        """
        Generate a system prompt based on user type and business context.
        
        Args:
            user_type: 'buyer', 'seller', or None
            business_context: Optional dictionary with business information
        
        Returns:
            System prompt string
        """
        base_prompt = """You are the Buy from Egypt chatbot, an AI assistant specialized in Egyptian economy, business, and e-commerce.
Your purpose is to help users with any questions related to Egyptian business, economy, and trade.

Welcome users warmly, mentioning you are the "Buy from Egypt" chatbot.
Be concise, helpful, and knowledgeable about Egyptian industries, economy, and business practices.
Provide practical advice for business challenges and platform usage.

IMPORTANT: If asked a general question outside your specific knowledge base, try to relate it to Egyptian business context.
For example, if asked about technology trends, discuss how they impact Egyptian businesses.
If asked about global events, discuss their implications for Egyptian trade and economy.

Always maintain a professional, friendly tone appropriate for business users.

If you detect Arabic language in the query, respond with: "I currently focus on English language business inquiries. Please ask your question in English for the most accurate information about Egyptian business and economy."
"""
        
        if user_type == "buyer":
            base_prompt += """
Focus on helping buyers find Egyptian products, understand seller verification, payment options, and order tracking.
Provide guidance on product categories, quality indicators, and how to communicate with Egyptian sellers effectively.
"""
        elif user_type == "seller":
            base_prompt += """
Focus on helping Egyptian sellers optimize their listings, use promotional tools, understand shipping options, and interpret analytics.
Provide guidance on reaching international buyers, pricing strategies, and managing orders efficiently.
"""
        
        if business_context:
            industry = business_context.get("industry")
            region = business_context.get("region")
            
            if industry and industry in self.knowledge["industries"]:
                industry_info = self.knowledge["industries"][industry]
                base_prompt += f"\nThe user is in the {industry} industry. Some relevant information: {industry_info['description']}"
                
                if "challenges" in industry_info:
                    challenges = ", ".join(industry_info["challenges"])
                    base_prompt += f"\nCommon challenges in this industry include: {challenges}"
                
                if "opportunities" in industry_info:
                    opportunities = ", ".join(industry_info["opportunities"])
                    base_prompt += f"\nOpportunities in this industry include: {opportunities}"
            
            if region and region in self.knowledge["regions"]:
                region_info = self.knowledge["regions"][region]
                base_prompt += f"\nThe user is in the {region} region. Business density: {region_info['business_density']}. Infrastructure: {region_info['infrastructure']}"
        
        return base_prompt
    
    def search_knowledge_base(self, query):
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with relevant information and sources
        """
        relevant_info = []
        sources = []
        
        # Check for specific industry mentions in the query
        mentioned_industry = None
        for industry in self.knowledge["industries"]:
            if industry.lower() in query.lower():
                mentioned_industry = industry
                break
        
        # If a specific industry is mentioned, prioritize information about that industry
        if mentioned_industry:
            industry_details = self.knowledge["industries"][mentioned_industry]
            relevant_info.append(f"Industry: {mentioned_industry} - {industry_details['description']}")
            
            if "challenges" in industry_details:
                challenges = ", ".join(industry_details["challenges"])
                relevant_info.append(f"Challenges in {mentioned_industry}: {challenges}")
            
            if "opportunities" in industry_details:
                opportunities = ", ".join(industry_details["opportunities"])
                relevant_info.append(f"Opportunities in {mentioned_industry}: {opportunities}")
            
            if "key_regions" in industry_details:
                key_regions = ", ".join(industry_details["key_regions"])
                relevant_info.append(f"Key regions for {mentioned_industry}: {key_regions}")
                
            sources.append(f"Industry: {mentioned_industry}")
            
            # Return early with just this industry's information
            return {
                "relevant_info": relevant_info,
                "sources": sources
            }
        
        # General business and economy keywords
        general_business_keywords = ["business", "economy", "trade", "market", "commerce", "industry", "export", "import"]
        
        # Check if the query is very general about Egyptian business or economy
        is_general_query = any(keyword in query.lower() for keyword in general_business_keywords) and "egypt" in query.lower()
        
        # If it's a general query about Egyptian business, provide overview information
        if is_general_query and not any(industry.lower() in query.lower() for industry in self.knowledge["industries"]):
            # Add general economic information
            gdp_info = self.knowledge["economy"]["GDP_Growth"]
            relevant_info.append(f"Egyptian Economy Overview: {gdp_info['current']} GDP growth. {gdp_info['trend']}")
            relevant_info.append(f"Key sectors driving growth: {', '.join(gdp_info['sectors_driving_growth'])}")
            sources.append("Economy: GDP Growth")
            
            # Add export market information
            export_info = self.knowledge["economy"]["Export_Markets"]
            relevant_info.append(f"Main export markets: {', '.join(export_info['primary_destinations'])}")
            relevant_info.append(f"Growing export markets: {', '.join(export_info['growing_markets'])}")
            sources.append("Economy: Export Markets")
            
            # Add key industry overview
            relevant_info.append("Major Egyptian industries include:")
            for industry in ["Textiles", "Agriculture", "Tourism", "Information Technology"]:
                relevant_info.append(f"- {industry}: {self.knowledge['industries'][industry]['description']}")
            sources.append("Industries: Overview")
        
        # Search in industries
        for industry, details in self.knowledge["industries"].items():
            if industry.lower() in query.lower() or any(keyword in query.lower() for keyword in [industry.lower(), "sector", "industry"]):
                relevant_info.append(f"Industry: {industry} - {details['description']}")
                if "challenges" in details:
                    challenges = ", ".join(details["challenges"])
                    relevant_info.append(f"Challenges in {industry}: {challenges}")
                if "opportunities" in details:
                    opportunities = ", ".join(details["opportunities"])
                    relevant_info.append(f"Opportunities in {industry}: {opportunities}")
                sources.append(f"Industry: {industry}")
        
        # Search in economic indicators
        for indicator, details in self.knowledge["economy"].items():
            if indicator.lower().replace("_", " ") in query.lower() or any(keyword in query.lower() for keyword in ["economy", "economic", "gdp", "inflation", "investment", "export"]):
                if indicator == "GDP_Growth":
                    relevant_info.append(f"GDP Growth: {details['current']} - {details['trend']}")
                    sources.append("Economy: GDP Growth")
                elif indicator == "Inflation":
                    relevant_info.append(f"Inflation: {details['current']} - Impact on business: {details['impact_on_business']}")
                    sources.append("Economy: Inflation")
                elif indicator == "Foreign_Investment":
                    relevant_info.append(f"Foreign Investment: {details['trend']} in sectors like {', '.join(details['key_sectors'])}")
                    sources.append("Economy: Foreign Investment")
                elif indicator == "Export_Markets":
                    relevant_info.append(f"Export Markets: Primary destinations include {', '.join(details['primary_destinations'])}")
                    sources.append("Economy: Export Markets")
        
        # Search in business challenges
        for challenge_type, details in self.knowledge["business_challenges"].items():
            if challenge_type.lower() in query.lower() or "challenge" in query.lower() or "problem" in query.lower() or "issue" in query.lower():
                if challenge_type == "Regulatory" and any(keyword in query.lower() for keyword in ["regulation", "license", "tax", "customs"]):
                    relevant_info.append(f"Regulatory Challenges: {details['licensing']} - Solutions: {', '.join(details['solutions'])}")
                    sources.append("Business Challenges: Regulatory")
                elif challenge_type == "Financing" and any(keyword in query.lower() for keyword in ["finance", "loan", "credit", "investment", "capital"]):
                    relevant_info.append(f"Financing Challenges: {details['access_to_credit']} - Solutions: {', '.join(details['solutions'])}")
                    sources.append("Business Challenges: Financing")
                elif challenge_type == "Operations" and any(keyword in query.lower() for keyword in ["operation", "supply chain", "workforce", "technology"]):
                    relevant_info.append(f"Operational Challenges: {details['supply_chain']} - Solutions: {', '.join(details['solutions'])}")
                    sources.append("Business Challenges: Operations")
                elif challenge_type == "Market_Access" and any(keyword in query.lower() for keyword in ["market", "customer", "competition", "pricing"]):
                    relevant_info.append(f"Market Access Challenges: {details['customer_acquisition']} - Solutions: {', '.join(details['solutions'])}")
                    sources.append("Business Challenges: Market Access")
        
        # Search in customer support
        for support_type, details in self.knowledge["customer_support"].items():
            if support_type.lower().replace("_", " ") in query.lower() or "help" in query.lower() or "support" in query.lower() or "how to" in query.lower():
                if support_type == "Platform_Navigation" and any(keyword in query.lower() for keyword in ["navigation", "account", "setup", "listing", "order", "payment"]):
                    for key, value in details.items():
                        relevant_info.append(f"{key.replace('_', ' ').title()}: {value}")
                    sources.append("Customer Support: Platform Navigation")
                elif support_type == "Buyer_Support" and any(keyword in query.lower() for keyword in ["buy", "buyer", "purchase", "find", "product"]):
                    for key, value in details.items():
                        relevant_info.append(f"{key.replace('_', ' ').title()}: {value}")
                    sources.append("Customer Support: Buyer Support")
                elif support_type == "Seller_Support" and any(keyword in query.lower() for keyword in ["sell", "seller", "listing", "promotion", "shipping"]):
                    for key, value in details.items():
                        relevant_info.append(f"{key.replace('_', ' ').title()}: {value}")
                    sources.append("Customer Support: Seller Support")
                elif support_type == "Common_Issues" and any(keyword in query.lower() for keyword in ["issue", "problem", "access", "payment", "shipping", "return"]):
                    for key, value in details.items():
                        relevant_info.append(f"{key.replace('_', ' ').title()}: {value}")
                    sources.append("Customer Support: Common Issues")
        
        # Search in regions
        for region, details in self.knowledge["regions"].items():
            if region.lower() in query.lower() or "region" in query.lower() or "area" in query.lower() or "location" in query.lower():
                relevant_info.append(f"Region: {region} - Business density: {details['business_density']}, Infrastructure: {details['infrastructure']}")
                relevant_info.append(f"Key industries in {region}: {', '.join(details['key_industries'])}")
                relevant_info.append(f"Business advantages in {region}: {details['business_advantages']}")
                sources.append(f"Region: {region}")
        
        return {
            "relevant_info": relevant_info,
            "sources": sources
        }
    
    async def get_response(self, query, user_type=None, business_context=None, session_id=None):
        """
        Get a response from the chatbot for a given query.
        
        Args:
            query: User query string
            user_type: Optional user type ('buyer' or 'seller')
            business_context: Optional dictionary with business information
            session_id: Optional session ID for conversation tracking
            
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        # Generate a session ID if not provided
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
        
        # Check if the query is in Arabic
        is_arabic_query = self.is_arabic(query)
        
        # Search knowledge base for relevant information
        knowledge_results = self.search_knowledge_base(query)
        relevant_info = knowledge_results["relevant_info"]
        sources = knowledge_results["sources"]
        
        # Get conversation history or create new conversation
        if session_id in self.conversations:
            conversation = self.conversations[session_id]
        else:
            conversation = []
            self.conversations[session_id] = conversation
        
        # Add user message to conversation
        conversation.append({"role": "user", "content": query})
        
        # Handle Arabic queries with a specific response
        if is_arabic_query:
            response_text = "Welcome to Buy from Egypt. I currently focus on English language business inquiries. Please ask your question in English for the most accurate information about Egyptian business and economy."
            conversation.append({"role": "assistant", "content": response_text})
            
            processing_time = time.time() - start_time
            
            return {
                "response": response_text,
                "sources": None,
                "session_id": session_id,
                "processing_time": processing_time
            }
        
        # If API is available, use Gemini to generate a response
        if self.api_available and self.model:
            try:
                # Prepare the prompt with system instructions and knowledge
                system_prompt = self.generate_system_prompt(user_type, business_context)
                
                # Add relevant information from knowledge base
                if relevant_info:
                    system_prompt += "\n\nRelevant information from knowledge base:\n"
                    system_prompt += "\n".join(relevant_info)
                else:
                    # For queries without specific knowledge matches, provide general guidance
                    system_prompt += "\n\nThe user has asked a question that doesn't directly match our knowledge base. "
                    system_prompt += "Please respond helpfully, relating the answer to Egyptian business context. "
                    system_prompt += "For example, if asked about global trends, discuss how they affect Egyptian businesses. "
                    system_prompt += "If asked about technologies or practices, explain their relevance to Egyptian commerce."
                
                # Create the chat session
                chat_session = self.model.start_chat(history=[])
                
                # Add the system prompt as the first message
                chat_session.send_message(system_prompt)
                
                # Add conversation history (limited to last 10 messages)
                for msg in conversation[-10:]:
                    chat_session.send_message(msg["content"])
                
                # Get the response
                response = chat_session.send_message(query)
                
                # Add assistant message to conversation
                conversation.append({"role": "assistant", "content": response.text})
                
                processing_time = time.time() - start_time
                
                return {
                    "response": response.text,
                    "sources": sources if sources else None,
                    "session_id": session_id,
                    "processing_time": processing_time
                }
                
            except Exception as e:
                logger.error(f"Error generating response with Gemini: {e}")
                # Fall back to knowledge-based response
                return self._generate_fallback_response(query, relevant_info, sources, session_id, start_time, conversation)
        else:
            # Use knowledge-based response if API is not available
            return self._generate_fallback_response(query, relevant_info, sources, session_id, start_time, conversation)
    
    def _generate_fallback_response(self, query, relevant_info, sources, session_id, start_time, conversation=None):
        """Generate a fallback response based on the knowledge base and conversation history."""
        # Check if we have conversation history to use for context
        has_context = conversation and len(conversation) > 1
        
        if not relevant_info:
            # More helpful default response for general questions
            response_text = "Welcome to Buy from Egypt. "
            
            if has_context:
                # Add context from previous conversation
                response_text += "Based on our conversation, "
            
            # Try to provide a helpful response even without specific knowledge
            if "how" in query.lower() and "help" in query.lower():
                response_text += "I can help you with information about Egyptian industries, economic indicators, business challenges, and platform usage. "
                response_text += "For example, I can tell you about the textile industry in Egypt, current economic trends, common business challenges, or how to use our platform effectively."
            elif any(word in query.lower() for word in ["industry", "sector", "business", "trade", "export", "import"]):
                response_text += "Egypt has diverse industries including textiles, agriculture, food processing, handicrafts, tourism, IT, pharmaceuticals, and furniture manufacturing. "
                response_text += "Each industry has unique opportunities and challenges in the Egyptian market. Could you specify which industry you're interested in learning more about?"
            elif any(word in query.lower() for word in ["economy", "economic", "gdp", "inflation", "invest"]):
                response_text += "Egypt's economy has been growing steadily with a GDP growth rate of about 5.6% (2023). Key economic factors affecting businesses include inflation (approximately 30% in 2023), foreign investment trends, and export market opportunities. "
                response_text += "Would you like more specific information about any particular economic indicator?"
            elif any(word in query.lower() for word in ["challenge", "problem", "issue", "difficulty"]):
                response_text += "Egyptian businesses face various challenges including regulatory complexities, financing access, operational issues, and market access. "
                response_text += "I can provide more specific information about any of these challenge areas if you're interested."
            elif any(word in query.lower() for word in ["platform", "website", "buy", "sell", "account", "listing"]):
                response_text += "Our platform connects Egyptian sellers with buyers worldwide. I can help with account setup, product listings, payment options, shipping, and more. "
                response_text += "What specific aspect of using the platform would you like assistance with?"
            else:
                response_text += "I'm here to answer your questions about Egyptian business, economy, industries, and our platform. "
                response_text += "Please feel free to ask about specific industries, economic indicators, business challenges, or how to use our platform."
        else:
            response_text = "Welcome to Buy from Egypt. "
            
            if has_context:
                # Add context from previous conversation
                response_text += "Based on our conversation, here's relevant information: "
            
            response_text += "\n\n" + "\n\n".join(relevant_info[:3])  # Limit to top 3 pieces of information
            
            if len(relevant_info) > 3:
                response_text += "\n\nI have additional information available if you'd like to know more specific details."
        
        # Add to conversation history
        if session_id in self.conversations:
            self.conversations[session_id].append({"role": "assistant", "content": response_text})
        
        processing_time = time.time() - start_time
        
        return {
            "response": response_text,
            "sources": sources if sources else None,
            "session_id": session_id,
            "processing_time": processing_time
        }
    
    def reset_conversation(self, session_id):
        """Reset the conversation history for a given session."""
        if session_id in self.conversations:
            self.conversations[session_id] = []
            return {"status": "success", "message": "Conversation reset successfully"}
        else:
            return {"status": "error", "message": "Session not found"} 