#!/usr/bin/env python3

"""
Comprehensive Data Processor for Egyptian Post Recommendation System
Handles all data extraction, cleaning, and feature engineering in one place.
"""

import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
import random
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Single comprehensive data processor for the entire system."""
    
    def __init__(self):
        self.data_dir = Path('data')
        self.processed_dir = Path('data/processed')
        self.processed_dir.mkdir(exist_ok=True)
        
        # Use CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Egyptian geographic data
        self.egyptian_cities = {
            'Cairo': 'Greater Cairo', 'Giza': 'Greater Cairo',
            '6th of October': 'Greater Cairo', 'New Cairo': 'Greater Cairo',
            'Alexandria': 'Mediterranean Coast', 'Port Said': 'Mediterranean Coast',
            'Damietta': 'Mediterranean Coast', 'Marsa Matruh': 'Mediterranean Coast',
            'Suez': 'Suez Canal', 'Ismailia': 'Suez Canal',
            'Sharm El Sheikh': 'Sinai', 'Dahab': 'Sinai', 'El Arish': 'Sinai',
            'Hurghada': 'Red Sea', 'Safaga': 'Red Sea', 'Marsa Alam': 'Red Sea',
            'Aswan': 'Upper Egypt', 'Luxor': 'Upper Egypt', 'Qena': 'Upper Egypt',
            'Sohag': 'Upper Egypt', 'Assiut': 'Upper Egypt',
            'Minya': 'Middle Egypt', 'Beni Suef': 'Middle Egypt',
            'Fayyum': 'Middle Egypt', 'Tanta': 'Delta', 'Mansoura': 'Delta',
            'Zagazig': 'Delta', 'Damanhur': 'Delta', 'Kafr El Sheikh': 'Delta'
        }
        
        # Industry categories for Egyptian market
        self.industries = [
            'Textiles & Garments', 'Agriculture & Food', 'Electronics',
            'Construction Materials', 'Automotive', 'Chemicals & Pharmaceuticals',
            'Tourism & Hospitality', 'Oil & Gas', 'Mining & Metals',
            'Furniture & Wood Products', 'Leather & Footwear'
        ]
    
    def process_all_data(self):
        """Main method to process all data from scratch."""
        logger.info("🚀 Starting comprehensive data processing...")
        
        try:
            # Step 1: Extract Egyptian companies
            companies_df = self._extract_egyptian_companies()
            
            # Step 2: Create user profiles
            users_df = self._create_user_profiles()
            
            # Step 3: Generate company posts
            posts_df = self._generate_company_posts(companies_df)
            
            # Step 4: Create user interactions
            interactions_df = self._create_user_interactions(users_df, posts_df)
            
            # Step 5: Add economic context
            self._add_economic_context()
            
            logger.info("✅ All data processing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in data processing: {e}")
            return False
    
    def _extract_egyptian_companies(self):
        """Extract Egyptian companies from large dataset."""
        logger.info("📊 Extracting Egyptian companies...")
        
        try:
            # Check if we already have processed companies
            companies_file = self.processed_dir / "egyptian_companies.csv"
            if companies_file.exists():
                logger.info("Using existing Egyptian companies data")
                return pd.read_csv(companies_file)
            
            # Extract from large dataset
            chunk_size = 50000
            egyptian_companies = []
            
            dataset_file = self.data_dir / "free_company_dataset.csv"
            if not dataset_file.exists():
                logger.warning("Large dataset not found, creating sample data")
                return self._create_sample_companies()
            
            logger.info("Processing large dataset in chunks...")
            for i, chunk in enumerate(pd.read_csv(
                dataset_file, 
                chunksize=chunk_size,
                on_bad_lines='skip',
                dtype=str
            )):
                if i % 10 == 0:
                    logger.info(f"   Processed {i * chunk_size:,} rows...")
                
                # Filter Egyptian companies
                egypt_mask = (
                    chunk['country'].str.contains('egypt', case=False, na=False) |
                    chunk['locality'].str.contains('cairo|alexandria|giza', case=False, na=False)
                )
                
                egyptian_chunk = chunk[egypt_mask].copy()
                if len(egyptian_chunk) > 0:
                    egyptian_companies.append(egyptian_chunk)
                
                # Limit processing for development
                if i > 200:  # Process ~10M rows
                    break
            
            if egyptian_companies:
                companies_df = pd.concat(egyptian_companies, ignore_index=True)
                companies_df = self._clean_company_data(companies_df)
                companies_df.to_csv(companies_file, index=False)
                logger.info(f"✅ Extracted {len(companies_df)} Egyptian companies")
                return companies_df
            else:
                return self._create_sample_companies()
                
        except Exception as e:
            logger.error(f"Error extracting companies: {e}")
            return self._create_sample_companies()
    
    def _clean_company_data(self, df):
        """Clean and standardize company data."""
        logger.info("🧹 Cleaning company data...")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['name'], keep='first')
        
        # Clean names
        df['name'] = df['name'].str.strip().str.title()
        
        # Standardize locations
        df['city'] = df['locality'].fillna(df['region']).str.title()
        df['region'] = df['city'].map(self.egyptian_cities).fillna('Other')
        
        # Assign industries
        df['industry'] = np.random.choice(self.industries, len(df))
        
        # Add business size
        df['size'] = np.random.choice(['Small', 'Medium', 'Large'], len(df), p=[0.6, 0.3, 0.1])
        
        # Keep only necessary columns
        columns_to_keep = ['name', 'city', 'region', 'industry', 'size', 'country']
        df = df[columns_to_keep].copy()
        
        return df
    
    def _create_sample_companies(self):
        """Create sample Egyptian companies if large dataset not available."""
        logger.info("Creating sample Egyptian companies...")
        
        sample_companies = []
        cities = list(self.egyptian_cities.keys())
        
        for i in range(200):
            city = random.choice(cities)
            industry = random.choice(self.industries)
            size = random.choice(['Small', 'Medium', 'Large'])
            
            company_types = ['Co.', 'Ltd.', 'Corp.', 'Industries', 'Trading', 'Group']
            company_name = f"{random.choice(['Nile', 'Cairo', 'Alexandria', 'Egyptian', 'Pharaoh', 'Golden', 'Modern', 'Premium'])} {random.choice(['Trade', 'Industries', 'Manufacturing', 'Export', 'Import', 'Business'])} {random.choice(company_types)}"
            
            sample_companies.append({
                'name': company_name,
                'city': city,
                'region': self.egyptian_cities[city],
                'industry': industry,
                'size': size,
                'country': 'Egypt'
            })
        
        df = pd.DataFrame(sample_companies)
        df.to_csv(self.processed_dir / "egyptian_companies.csv", index=False)
        return df
    
    def _create_user_profiles(self):
        """Create diverse user profiles with preferences."""
        logger.info("👥 Creating user profiles...")
        
        users = []
        for i in range(1000):
            user_id = f"user_{i+1}"
            
            # User demographics
            age_group = random.choice(['18-25', '26-35', '36-45', '46-55', '55+'])
            location = random.choice(list(self.egyptian_cities.keys()))
            business_type = random.choice(['Retailer', 'Wholesaler', 'Manufacturer', 'Service Provider'])
            
            # Preferences
            preferred_industries = random.sample(self.industries, k=random.randint(1, 3))
            preferred_regions = random.sample(list(set(self.egyptian_cities.values())), k=random.randint(1, 2))
            preferred_supplier_size = random.choice(['Small', 'Medium', 'Large', 'Any'])
            
            # Behavior patterns
            activity_level = random.choice(['Low', 'Medium', 'High'])
            price_sensitivity = random.choice(['Low', 'Medium', 'High'])
            
            users.append({
                'UserID': user_id,
                'AgeGroup': age_group,
                'Location': location,
                'BusinessType': business_type,
                'PreferredIndustries': ','.join(preferred_industries),
                'PreferredRegions': ','.join(preferred_regions),
                'PreferredSupplierSize': preferred_supplier_size,
                'ActivityLevel': activity_level,
                'PriceSensitivity': price_sensitivity
            })
        
        users_df = pd.DataFrame(users)
        users_df.to_csv(self.processed_dir / "user_profiles.csv", index=False)
        logger.info(f"✅ Created {len(users_df)} user profiles")
        return users_df
    
    def _generate_company_posts(self, companies_df):
        """Generate posts for companies."""
        logger.info("📝 Generating company posts...")
        
        posts = []
        post_id = 1
        
        # Generate 2-5 posts per company
        for _, company in companies_df.iterrows():
            num_posts = random.randint(2, 5)
            
            for _ in range(num_posts):
                post_types = ['Product Launch', 'Service Announcement', 'Partnership Opportunity', 'Special Offer']
                post_type = random.choice(post_types)
                
                title = f"{post_type} from {company['name']}"
                
                # Engagement metrics
                views = random.randint(50, 2000)
                likes = random.randint(0, views // 5)
                comments = random.randint(0, views // 20)
                shares = random.randint(0, views // 30)
                
                posts.append({
                    'PostID': post_id,
                    'CompanyName': company['name'],
                    'PostTitle': title,
                    'PostType': post_type,
                    'Industry': company['industry'],
                    'Location': company['city'],
                    'Region': company['region'],
                    'CompanySize': company['size'],
                    'Views': views,
                    'Likes': likes,
                    'Comments': comments,
                    'Shares': shares,
                    'Engagement': likes + comments + shares,
                    'CreatedDate': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
                })
                post_id += 1
        
        posts_df = pd.DataFrame(posts)
        posts_df.to_csv(self.processed_dir / "company_posts.csv", index=False)
        logger.info(f"✅ Generated {len(posts_df)} company posts")
        return posts_df
    
    def _create_user_interactions(self, users_df, posts_df):
        """Create realistic user-post interactions."""
        logger.info("🔄 Creating user interactions...")
        
        interactions = []
        interaction_id = 1
        
        for _, user in users_df.iterrows():
            user_id = user['UserID']
            
            # Filter posts based on user preferences
            preferred_industries = user['PreferredIndustries'].split(',')
            preferred_regions = user['PreferredRegions'].split(',')
            
            relevant_posts = posts_df[
                (posts_df['Industry'].isin(preferred_industries)) |
                (posts_df['Region'].isin(preferred_regions))
            ]
            
            # If no relevant posts, sample randomly
            if len(relevant_posts) == 0:
                relevant_posts = posts_df.sample(n=min(20, len(posts_df)))
            
            # Determine user activity level
            activity_multiplier = {'Low': 0.3, 'Medium': 0.6, 'High': 1.0}[user['ActivityLevel']]
            num_interactions = int(random.randint(10, 50) * activity_multiplier)
            
            # Sample posts for interaction
            if len(relevant_posts) > 0:
                sampled_posts = relevant_posts.sample(n=min(num_interactions, len(relevant_posts)))
                
                for _, post in sampled_posts.iterrows():
                    # Different types of interactions
                    interaction_types = ['view', 'like', 'comment', 'share', 'rate']
                    weights = [0.7, 0.15, 0.08, 0.05, 0.02]  # View is most common
                    interaction_type = random.choices(interaction_types, weights=weights)[0]
                    
                    # Time spent (seconds)
                    time_spent = random.randint(5, 300)
                    
                    # Rating (1-5) if it's a rating interaction
                    rating = random.randint(1, 5) if interaction_type == 'rate' else None
                    
                    # Interaction strength
                    strength_map = {'view': 1, 'like': 3, 'comment': 5, 'share': 4, 'rate': rating or 3}
                    interaction_strength = strength_map[interaction_type]
                    
                    interactions.append({
                        'InteractionID': interaction_id,
                        'UserID': user_id,
                        'PostID': post['PostID'],
                        'InteractionType': interaction_type,
                        'InteractionStrength': interaction_strength,
                        'TimeSpent': time_spent,
                        'Rating': rating,
                        'Timestamp': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d %H:%M:%S')
                    })
                    interaction_id += 1
        
        interactions_df = pd.DataFrame(interactions)
        interactions_df.to_csv(self.processed_dir / "user_interactions.csv", index=False)
        logger.info(f"✅ Created {len(interactions_df)} user interactions")
        return interactions_df
    
    def _add_economic_context(self):
        """Add Egyptian economic context data."""
        logger.info("💰 Adding economic context...")
        
        economic_data = {
            'gdp_growth': 4.35,
            'inflation': 5.04,
            'population_growth': 1.73,
            'tourism_sensitivity': 0.8,
            'trade_balance': -2.5,
            'currency_stability': 0.7,
            'current_season': 'summer',
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }
        
        import json
        with open(self.processed_dir / "economic_context.json", 'w') as f:
            json.dump(economic_data, f, indent=2)
        
        logger.info("✅ Economic context added")

def main():
    """Main execution function."""
    processor = DataProcessor()
    success = processor.process_all_data()
    
    if success:
        print("🎉 Data processing completed successfully!")
        print("📁 Processed files saved in data/processed/")
    else:
        print("❌ Data processing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
