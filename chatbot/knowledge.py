"""
Egyptian business and economy knowledge base for the Buy from Egypt chatbot.
This file contains structured information about Egyptian industries, economy,
common business challenges, and customer support information.
"""

# Egyptian Industry Sectors
INDUSTRY_SECTORS = {
    "Textiles": {
        "description": "Egypt's textile industry is renowned for high-quality cotton production, with major exports to Europe and MENA regions.",
        "key_regions": ["Greater Cairo", "Nile Delta", "Alexandria"],
        "challenges": ["Global competition", "Raw material price fluctuations", "Modernization needs"],
        "opportunities": ["European market expansion", "Sustainable textile production", "Value-added garment manufacturing"],
        "seasonal_factors": ["Winter tourism increases demand for textile souvenirs", "Ramadan increases demand for home textiles"]
    },
    "Agriculture": {
        "description": "Agricultural businesses focus on fruits, vegetables, and cotton, with strong export potential to Europe and Gulf countries.",
        "key_regions": ["Nile Delta", "Upper Egypt", "Fayoum"],
        "challenges": ["Water scarcity", "Climate change impacts", "Cold chain logistics"],
        "opportunities": ["Organic farming certification", "Export to premium markets", "Agricultural technology adoption"],
        "seasonal_factors": ["Harvest seasons vary by crop", "Ramadan affects food consumption patterns"]
    },
    "Food Processing": {
        "description": "Egypt's food processing sector transforms local agricultural products into packaged foods, beverages, and export-ready items.",
        "key_regions": ["Greater Cairo", "Alexandria", "10th of Ramadan City"],
        "challenges": ["Quality control standards", "Packaging innovations", "Cold chain maintenance"],
        "opportunities": ["Ready-made meals market", "Export to Arab markets", "Health-conscious food products"],
        "seasonal_factors": ["Ramadan increases processed food demand", "Summer increases beverage consumption"]
    },
    "Handicrafts": {
        "description": "Egyptian handicrafts include traditional textiles, pottery, metalwork, and woodwork with cultural significance and tourism appeal.",
        "key_regions": ["Khan el-Khalili", "Luxor", "Aswan", "Siwa"],
        "challenges": ["Younger generation interest", "Quality standardization", "Export packaging"],
        "opportunities": ["E-commerce platforms", "Tourism industry partnerships", "Contemporary design integration"],
        "seasonal_factors": ["Tourism seasons drive demand", "Holiday gift-giving periods"]
    },
    "Tourism": {
        "description": "Egypt's tourism industry leverages historical sites, Red Sea resorts, and cultural experiences as major economic drivers.",
        "key_regions": ["Cairo", "Luxor", "Aswan", "Red Sea", "South Sinai"],
        "challenges": ["Political stability perceptions", "Service quality consistency", "Digital marketing"],
        "opportunities": ["Eco-tourism development", "Medical tourism", "Cultural experience packages"],
        "seasonal_factors": ["Winter peak season (October-March)", "European holiday periods"]
    },
    "Information Technology": {
        "description": "Egypt's growing IT sector provides software development, outsourcing services, and technology solutions regionally.",
        "key_regions": ["Smart Village (Cairo)", "Alexandria", "New Administrative Capital"],
        "challenges": ["Talent retention", "International competition", "Infrastructure reliability"],
        "opportunities": ["Fintech solutions", "Arabic content development", "Business process outsourcing"],
        "seasonal_factors": ["Relatively stable year-round demand"]
    },
    "Pharmaceuticals": {
        "description": "Egyptian pharmaceutical industry produces generic medications and medical supplies for domestic and regional markets.",
        "key_regions": ["Greater Cairo", "Alexandria", "10th of Ramadan City"],
        "challenges": ["Regulatory approvals", "Import dependencies for raw materials", "Research funding"],
        "opportunities": ["Export to African markets", "Contract manufacturing", "Natural and herbal medicines"],
        "seasonal_factors": ["Seasonal illness patterns affect demand"]
    },
    "Furniture": {
        "description": "Egyptian furniture manufacturing combines traditional craftsmanship with modern production, known for wood quality and design.",
        "key_regions": ["Damietta", "Cairo", "Alexandria"],
        "challenges": ["Wood sourcing", "Design innovation", "Export logistics"],
        "opportunities": ["High-end export markets", "Hotel and tourism sector supply", "Office furniture for Gulf markets"],
        "seasonal_factors": ["Housing market cycles", "Tourism development projects"]
    }
}

# Economic Indicators and Trends
ECONOMIC_INDICATORS = {
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

# Common Business Challenges
BUSINESS_CHALLENGES = {
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

# Customer Support Information
CUSTOMER_SUPPORT = {
    "Platform_Navigation": {
        "account_setup": "Step-by-step guidance for creating and verifying business accounts",
        "product_listing": "Instructions for adding products with effective descriptions and images",
        "order_management": "Process for receiving, confirming, and fulfilling orders",
        "payment_options": "Available payment methods and setup procedures"
    },
    "Buyer_Support": {
        "finding_products": "Search and filtering techniques to find specific Egyptian products",
        "verifying_sellers": "Understanding seller ratings and verification badges",
        "payment_security": "Secure payment options and buyer protection policies",
        "order_tracking": "Methods to track order status and delivery timeframes"
    },
    "Seller_Support": {
        "optimizing_listings": "Best practices for product titles, descriptions, and images",
        "promotion_tools": "Available marketing tools and promotional features",
        "shipping_options": "Domestic and international shipping partners and methods",
        "analytics": "Understanding seller dashboard metrics and performance indicators"
    },
    "Common_Issues": {
        "account_access": "Troubleshooting login problems and account recovery",
        "payment_delays": "Understanding payment processing timeframes and issues",
        "shipping_delays": "Managing and communicating delivery expectations",
        "return_process": "Policies and procedures for product returns and refunds"
    }
}

# Egyptian Regional Business Information
REGIONAL_CHARACTERISTICS = {
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
    "Delta Region": {
        "business_density": "Medium to High",
        "infrastructure": "Developing with agricultural focus",
        "key_industries": ["Agriculture", "Food Processing", "Textiles", "Furniture"],
        "business_advantages": "Agricultural resources, manufacturing tradition, and proximity to ports",
        "challenges": "Infrastructure limitations and environmental concerns"
    },
    "Suez Canal Zone": {
        "business_density": "Growing rapidly",
        "infrastructure": "Modern with significant recent investment",
        "key_industries": ["Logistics", "Manufacturing", "Services", "Maritime"],
        "business_advantages": "Special economic zone benefits, strategic location, and new facilities",
        "challenges": "Developing residential amenities and service sector"
    },
    "Upper Egypt": {
        "business_density": "Low to Medium",
        "infrastructure": "Developing with regional variations",
        "key_industries": ["Agriculture", "Mining", "Tourism", "Handicrafts"],
        "business_advantages": "Lower operating costs, government development focus, and tourism potential",
        "challenges": "Distance from major markets, infrastructure gaps, and skilled labor availability"
    },
    "Red Sea Coast": {
        "business_density": "Medium (concentrated in tourism zones)",
        "infrastructure": "Good in tourism areas, developing elsewhere",
        "key_industries": ["Tourism", "Hospitality", "Real Estate", "Marine Services"],
        "business_advantages": "International tourism market, development incentives, and quality of life",
        "challenges": "Seasonality, water resources, and supply chain logistics"
    }
}

# Combined Knowledge Base
EGYPTIAN_KNOWLEDGE = {
    "industries": INDUSTRY_SECTORS,
    "economy": ECONOMIC_INDICATORS,
    "business_challenges": BUSINESS_CHALLENGES,
    "customer_support": CUSTOMER_SUPPORT,
    "regions": REGIONAL_CHARACTERISTICS
} 