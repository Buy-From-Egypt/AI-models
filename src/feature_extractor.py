#!/usr/bin/env python3

"""
Advanced Feature Extractor for Post Recommendation System
Extracts and processes all user interaction features systematically.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFeatureExtractor:
    """
    Extracts and processes advanced features for post recommendation.
    """
    
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Feature processors
        self.scalers = {}
        self.encoders = {}
        
        # Data containers
        self.users_df = None
        self.posts_df = None
        self.interactions_df = None
        
    def load_base_data(self):
        """Load base datasets."""
        logger.info("Loading base datasets...")
        
        try:
            self.users_df = pd.read_csv("data/processed/user_preferences.csv")
            self.posts_df = pd.read_csv("data/processed/company_posts.csv")
            self.interactions_df = pd.read_csv("data/processed/user_post_interactions.csv")
            
            logger.info(f"Loaded {len(self.users_df)} users, {len(self.posts_df)} posts, {len(self.interactions_df)} interactions")
            return True
            
        except Exception as e:
            logger.error(f"Error loading base data: {e}")
            return False
    
    def extract_user_behavior_features(self):
        """
        Extract advanced user behavior features.
        """
        logger.info("Extracting user behavior features...")
        
        # 1. Rating Behavior Features
        rating_features = self._extract_rating_features()
        
        # 2. Time Behavior Features  
        time_features = self._extract_time_features()
        
        # 3. Engagement Features
        engagement_features = self._extract_engagement_features()
        
        # 4. Browse Pattern Features
        browse_features = self._extract_browse_features()
        
        # Combine all features
        user_features = pd.concat([
            rating_features,
            time_features, 
            engagement_features,
            browse_features
        ], axis=1)
        
        return user_features
    
    def _extract_rating_features(self):
        """Extract rating-based features."""
        logger.info("Processing rating features...")
        
        # Simulate rating data (in production, this comes from user actions)
        np.random.seed(42)
        
        rating_data = []
        for _, interaction in self.interactions_df.iterrows():
            user_id = interaction['UserID']
            post_id = interaction['PostID']
            
            # Simulate rating based on interaction score
            base_score = interaction['InteractionScore']
            if base_score > 0.8:
                rating = np.random.choice([4, 5], p=[0.3, 0.7])
            elif base_score > 0.6:
                rating = np.random.choice([3, 4], p=[0.6, 0.4])
            elif base_score > 0.4:
                rating = np.random.choice([2, 3], p=[0.7, 0.3])
            else:
                rating = np.random.choice([1, 2], p=[0.8, 0.2])
            
            rating_data.append({
                'UserID': user_id,
                'PostID': post_id,
                'Rating': rating,
                'Timestamp': interaction['InteractionDate']
            })
        
        ratings_df = pd.DataFrame(rating_data)
        
        # Aggregate rating features per user
        user_rating_features = ratings_df.groupby('UserID').agg({
            'Rating': ['mean', 'std', 'count', 'min', 'max'],
        }).round(3)
        
        user_rating_features.columns = [
            'avg_rating', 'rating_std', 'total_ratings', 'min_rating', 'max_rating'
        ]
        
        # Rating distribution
        rating_dist = ratings_df.groupby('UserID')['Rating'].value_counts().unstack(fill_value=0)
        rating_dist.columns = [f'rating_{int(col)}_count' for col in rating_dist.columns]
        
        return pd.concat([user_rating_features, rating_dist], axis=1).fillna(0)
    
    def _extract_time_features(self):
        """Extract time-based behavior features."""
        logger.info("Processing time behavior features...")
        
        # Simulate time spent data
        np.random.seed(42)
        
        time_data = []
        for _, interaction in self.interactions_df.iterrows():
            user_id = interaction['UserID']
            post_id = interaction['PostID']
            base_score = interaction['InteractionScore']
            
            # Simulate time spent based on interaction score
            if base_score > 0.8:
                time_spent = np.random.lognormal(4.0, 0.5)  # Higher engagement
            elif base_score > 0.6:
                time_spent = np.random.lognormal(3.5, 0.6)  # Medium engagement
            else:
                time_spent = np.random.lognormal(2.5, 0.8)  # Lower engagement
            
            time_spent = max(5, min(300, time_spent))  # 5 seconds to 5 minutes
            
            time_data.append({
                'UserID': user_id,
                'PostID': post_id,
                'TimeSpent': time_spent,
                'Timestamp': interaction['InteractionDate']
            })
        
        time_df = pd.DataFrame(time_data)
        
        # Aggregate time features per user
        user_time_features = time_df.groupby('UserID').agg({
            'TimeSpent': ['mean', 'std', 'sum', 'max', 'count'],
        }).round(3)
        
        user_time_features.columns = [
            'avg_time_spent', 'time_spent_std', 'total_time_spent', 'max_time_spent', 'session_count'
        ]
        
        return user_time_features
    
    def _extract_engagement_features(self):
        """Extract engagement-based features (likes, shares, comments)."""
        logger.info("Processing engagement features...")
        
        np.random.seed(42)
        
        engagement_data = []
        for _, interaction in self.interactions_df.iterrows():
            user_id = interaction['UserID']
            post_id = interaction['PostID']
            base_score = interaction['InteractionScore']
            
            # Simulate engagement actions based on interaction score
            like_prob = base_score * 0.3
            share_prob = base_score * 0.1
            comment_prob = base_score * 0.05
            
            liked = np.random.random() < like_prob
            shared = np.random.random() < share_prob
            commented = np.random.random() < comment_prob
            
            engagement_data.append({
                'UserID': user_id,
                'PostID': post_id,
                'Liked': liked,
                'Shared': shared,
                'Commented': commented,
                'Timestamp': interaction['InteractionDate']
            })
        
        engagement_df = pd.DataFrame(engagement_data)
        
        # Aggregate engagement features per user
        user_engagement = engagement_df.groupby('UserID').agg({
            'Liked': 'sum',
            'Shared': 'sum', 
            'Commented': 'sum'
        })
        
        user_engagement.columns = ['total_likes', 'total_shares', 'total_comments']
        
        # Engagement rates
        total_interactions = engagement_df.groupby('UserID').size()
        user_engagement['like_rate'] = (user_engagement['total_likes'] / total_interactions).fillna(0)
        user_engagement['share_rate'] = (user_engagement['total_shares'] / total_interactions).fillna(0)
        user_engagement['comment_rate'] = (user_engagement['total_comments'] / total_interactions).fillna(0)
        
        return user_engagement.round(3)
    
    def _extract_browse_features(self):
        """Extract marketplace browsing pattern features."""
        logger.info("Processing browse pattern features...")
        
        # Parse user preferences
        browse_features = []
        
        for _, user in self.users_df.iterrows():
            user_id = user['UserID']
            
            # Parse preferred industries
            industries = user['PreferredIndustries'].split(',')
            num_preferred_industries = len(industries)
            
            # Supplier type preference
            supplier_type = user['PreferredSupplierType']
            
            # Order patterns
            order_quantity = user['PreferredOrderQuantity']
            shipping_method = user['PreferredShippingMethod']
            
            # Calculate user's market exploration behavior
            user_interactions = self.interactions_df[self.interactions_df['UserID'] == user_id]
            
            if len(user_interactions) > 0:
                # Get industries of posts user interacted with
                interacted_posts = user_interactions['PostID'].values
                post_industries = self.posts_df[self.posts_df['PostID'].isin(interacted_posts)]['Industry'].values
                unique_industries_explored = len(set(post_industries))
                
                # Browse diversity score
                browse_diversity = unique_industries_explored / max(1, len(industries))
            else:
                unique_industries_explored = 0
                browse_diversity = 0
            
            browse_features.append({
                'UserID': user_id,
                'num_preferred_industries': num_preferred_industries,
                'unique_industries_explored': unique_industries_explored,
                'browse_diversity_score': browse_diversity,
                'prefers_large_corp': 1 if 'Large' in supplier_type else 0,
                'prefers_startup': 1 if 'Startup' in supplier_type else 0,
                'prefers_bulk_orders': 1 if 'Bulk' in order_quantity else 0,
                'prefers_sea_freight': 1 if 'Sea' in shipping_method else 0,
                'receives_alerts': 1 if user['ReceiveAlerts'] else 0
            })
        
        return pd.DataFrame(browse_features).set_index('UserID')
    
    def extract_post_features(self):
        """Extract advanced post features."""
        logger.info("Extracting post features...")
        
        post_features = []
        
        # Encode categorical features
        industries = self.posts_df['Industry'].unique()
        industry_encoder = {industry: idx for idx, industry in enumerate(industries)}
        
        for _, post in self.posts_df.iterrows():
            post_id = post['PostID']
            
            # Basic features (encoded)
            industry = post['Industry']
            industry_encoded = industry_encoder[industry]
            
            # Interaction statistics
            post_interactions = self.interactions_df[self.interactions_df['PostID'] == post_id]
            
            num_interactions = len(post_interactions)
            avg_interaction_score = post_interactions['InteractionScore'].mean() if num_interactions > 0 else 0
            unique_users = post_interactions['UserID'].nunique()
            
            # Engagement metrics (simulated)
            np.random.seed(post_id)
            engagement_score = np.random.beta(2, 5) * avg_interaction_score
            
            post_features.append({
                'PostID': post_id,
                'industry_encoded': industry_encoded,
                'num_interactions': num_interactions,
                'avg_interaction_score': avg_interaction_score,
                'unique_users': unique_users,
                'engagement_score': engagement_score,
                'post_popularity': num_interactions / max(1, self.users_df.shape[0])  # Popularity ratio
            })
        
        # Save industry encoder for later use
        self.encoders['industry'] = industry_encoder
        
        return pd.DataFrame(post_features).set_index('PostID')
    
    def create_interaction_matrix(self, interaction_type='rating'):
        """Create user-post interaction matrix for collaborative filtering."""
        logger.info(f"Creating {interaction_type} interaction matrix...")
        
        if interaction_type == 'rating':
            # Use simulated ratings
            matrix_data = []
            np.random.seed(42)
            
            for _, interaction in self.interactions_df.iterrows():
                user_id = interaction['UserID']
                post_id = interaction['PostID']
                base_score = interaction['InteractionScore']
                
                # Convert interaction score to rating
                if base_score > 0.8:
                    rating = np.random.choice([4, 5], p=[0.3, 0.7])
                elif base_score > 0.6:
                    rating = np.random.choice([3, 4], p=[0.6, 0.4])
                elif base_score > 0.4:
                    rating = np.random.choice([2, 3], p=[0.7, 0.3])
                else:
                    rating = np.random.choice([1, 2], p=[0.8, 0.2])
                
                matrix_data.append({
                    'UserID': user_id,
                    'PostID': post_id,
                    'Rating': rating
                })
            
            matrix_df = pd.DataFrame(matrix_data)
            interaction_matrix = matrix_df.pivot(index='UserID', columns='PostID', values='Rating').fillna(0)
            
        else:
            # Use interaction scores
            interaction_matrix = self.interactions_df.pivot(
                index='UserID', 
                columns='PostID', 
                values='InteractionScore'
            ).fillna(0)
        
        return interaction_matrix
    
    def save_features(self, user_features, post_features, interaction_matrix):
        """Save all extracted features."""
        logger.info("Saving extracted features...")
        
        # Create output directory
        output_dir = Path("data/features")
        output_dir.mkdir(exist_ok=True)
        
        # Save user features
        user_features.to_csv(output_dir / "user_features.csv")
        
        # Save post features  
        post_features.to_csv(output_dir / "post_features.csv")
        
        # Save interaction matrix
        interaction_matrix.to_csv(output_dir / "rating_matrix.csv")
        
        # Save encoders
        with open(output_dir / "encoders.pkl", 'wb') as f:
            pickle.dump(self.encoders, f)
        
        # Save feature info
        feature_info = {
            'extraction_date': datetime.now().isoformat(),
            'num_users': len(user_features),
            'num_posts': len(post_features),
            'num_interactions': interaction_matrix.sum().sum(),
            'feature_types': {
                'user_features': list(user_features.columns),
                'post_features': list(post_features.columns)
            }
        }
        
        with open(output_dir / "feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        logger.info(f"Features saved to {output_dir}")
        return output_dir

def main():
    """Main feature extraction pipeline."""
    print("🚀 Advanced Feature Extraction Pipeline")
    print("=" * 50)
    
    # Initialize extractor with CUDA support
    extractor = AdvancedFeatureExtractor(use_cuda=True)
    
    # Load base data
    if not extractor.load_base_data():
        print("❌ Failed to load base data")
        return
    
    # Extract features
    print("\n📊 Extracting User Behavior Features...")
    user_features = extractor.extract_user_behavior_features()
    
    print("\n📱 Extracting Post Features...")
    post_features = extractor.extract_post_features()
    
    print("\n🔢 Creating Interaction Matrix...")
    interaction_matrix = extractor.create_interaction_matrix('rating')
    
    # Save features
    output_dir = extractor.save_features(user_features, post_features, interaction_matrix)
    
    print("\n✅ Feature extraction completed successfully!")
    print(f"📁 Features saved to: {output_dir}")
    print(f"👥 User features: {user_features.shape}")
    print(f"📄 Post features: {post_features.shape}")
    print(f"🔢 Rating matrix: {interaction_matrix.shape}")

if __name__ == "__main__":
    main()
