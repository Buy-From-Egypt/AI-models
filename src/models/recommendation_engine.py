import pandas as pd
import numpy as np
import logging
import pickle
import joblib
from pathlib import Path
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

class HybridRecommendationModel(nn.Module):
    """Neural network model for hybrid recommendations."""
    
    def __init__(self, num_users, num_posts, num_companies, 
                 user_content_dim=16, post_content_dim=103, company_content_dim=4,
                 embedding_dim=64):
        super().__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.post_embedding = nn.Embedding(num_posts, embedding_dim)
        self.company_embedding = nn.Embedding(num_companies, embedding_dim)
        
        # Content feature networks
        self.user_content_net = nn.Sequential(
            nn.Linear(user_content_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )
        
        self.post_content_net = nn.Sequential(
            nn.Linear(post_content_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )
        
        self.company_content_net = nn.Sequential(
            nn.Linear(company_content_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim),
            nn.ReLU()
        )
        
        # Fusion layers
        fusion_input_dim = embedding_dim * 6  # 3 embeddings + 3 content features
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_ids, post_ids, company_ids, 
                user_content, post_content, company_content):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        post_emb = self.post_embedding(post_ids)
        company_emb = self.company_embedding(company_ids)
        
        # Process content features
        user_content_emb = self.user_content_net(user_content)
        post_content_emb = self.post_content_net(post_content)
        company_content_emb = self.company_content_net(company_content)
        
        # Concatenate all features
        combined = torch.cat([
            user_emb, post_emb, company_emb,
            user_content_emb, post_content_emb, company_content_emb
        ], dim=1)
        
        # Predict interaction probability
        output = self.fusion_layers(combined)
        return output.squeeze()

class PostRecommendationEngine:
    """
    Advanced recommendation engine using trained hybrid neural model.
    """
    
    def __init__(self):
        """
        Initialize the recommendation engine by loading the trained model.
        """
        logger.info("🚀 Initializing advanced hybrid recommendation engine...")
        
        try:
            # Create directories if they don't exist
            MODELS_DIR.mkdir(exist_ok=True)
            PROCESSED_DIR.mkdir(exist_ok=True)
            
            # Load the trained PyTorch model
            self._load_trained_model()
            
            # Load data
            self._load_data()
            
            logger.info("✅ Advanced recommendation engine initialized successfully!")
        
        except Exception as e:
            logger.error(f"❌ Error initializing recommendation engine: {e}")
            raise
    
    def _load_trained_model(self):
        """Load the trained PyTorch hybrid model."""
        model_path = MODELS_DIR / "hybrid_recommendation_model.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        
        logger.info("📦 Loading trained hybrid model...")
        
        # Load model data
        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Extract configuration
        config = model_data['model_config']
        self.num_users = config['num_users']
        self.num_posts = config['num_posts'] 
        self.num_companies = config['num_companies']
        
        # Initialize model architecture
        self.model = HybridRecommendationModel(
            num_users=self.num_users,
            num_posts=self.num_posts,
            num_companies=self.num_companies,
            user_content_dim=config['user_content_dim'],
            post_content_dim=config['post_content_dim'],
            company_content_dim=config['company_content_dim']
        )
        
        # Load trained weights
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.eval()
        
        # Load encoders
        self.user_encoder = model_data['user_encoder']
        self.post_encoder = model_data['post_encoder'] 
        self.company_encoder = model_data['company_encoder']
        self.encoders = model_data['encoders']
        
        # Store validation accuracy
        self.validation_accuracy = model_data['val_acc']
        
        logger.info(f"✅ Model loaded with {self.validation_accuracy:.1%} validation accuracy")
        logger.info(f"📊 Model covers {self.num_users:,} users, {self.num_posts:,} posts, {self.num_companies:,} companies")
    
    def _load_data(self):
        """Load supporting data files."""
        try:
            # Load economic context
            try:
                import json
                with open(PROCESSED_DIR / "egyptian_context.json", "r") as f:
                    self.economic_context = json.load(f)
                logger.info("✅ Loaded economic context")
            except:
                logger.warning("⚠️ Economic context not found, using defaults")
                self.economic_context = {
                    'gdp_growth': 4.35,
                    'inflation': 5.04,
                    'population_growth': 1.73
                }
            
            # Load company posts data
            try:
                # Load company posts
                self.posts_df = pd.read_csv(PROCESSED_DIR / "company_posts.csv")
                if 'PostID' in self.posts_df.columns:
                    self.posts_df.set_index('PostID', inplace=True)
                logger.info(f"✅ Loaded {len(self.posts_df)} company posts")
            except:
                logger.warning("⚠️ Company posts not found, creating sample data")
                self.posts_df = pd.DataFrame({
                    'PostID': [1, 2, 3],
                    'CompanyName': ["Tech Egypt", "Food Corp", "Textile Co"],
                    'PostTitle': ["Latest Electronics", "Fresh Produce", "Quality Fabrics"],
                    'Industry': ["Electronics", "Agriculture & Food", "Textiles & Garments"],
                    'Engagement': [100, 200, 150]
                }).set_index('PostID')
            
            # Load user preferences
            try:
                self.user_preferences_df = pd.read_csv(PROCESSED_DIR / "user_preferences.csv")
                if 'UserID' in self.user_preferences_df.columns:
                    self.user_preferences_df.set_index('UserID', inplace=True)
                logger.info(f"✅ Loaded {len(self.user_preferences_df)} user preferences")
            except:
                logger.warning("⚠️ User preferences not found, creating sample data")
                self.user_preferences_df = pd.DataFrame({
                    'UserID': ["1000", "1001", "1002"],
                    'PreferredIndustries': ["Electronics", "Agriculture & Food", "Textiles & Garments"],
                    'PreferredSupplierType': ["Small Businesses", "Medium Enterprises", "Large Corporations"]
                }).set_index('UserID')
            
            # Load business data
            try:
                self.business_df = pd.read_csv(PROCESSED_DIR / "business_features.csv")
                logger.info(f"✅ Loaded {len(self.business_df)} business profiles")
            except:
                logger.warning("⚠️ Business features not found, creating sample data")
                self.business_df = pd.DataFrame({
                    'BusinessID': [1, 2],
                    'Business Name': ["Business 1", "Business 2"],
                    'Category': ["Electronics", "Textiles"],
                    'Location': ["Cairo, Egypt", "Alexandria, Egypt"],
                    'Trade Type': ["Importer", "Exporter"]
                })
            
            # Load retail data for product information
            try:
                self.products_df = pd.read_csv(PROCESSED_DIR / "retail_cleaned.csv")
                if 'StockCode' in self.products_df.columns and 'Description' in self.products_df.columns:
                    self.products_df = self.products_df[['StockCode', 'Description']].drop_duplicates()
                    self.products_df.set_index('StockCode', inplace=True)
                    logger.info(f"✅ Loaded {len(self.products_df)} products")
                else:
                    raise FileNotFoundError("Required columns not found")
            except:
                logger.warning("⚠️ Product info not found, creating sample data")
                self.products_df = pd.DataFrame({
                    'StockCode': ["10000", "10001", "10002"],
                    'Description': ["Product 1", "Product 2", "Product 3"]
                }).set_index('StockCode')
            
            # Load interaction matrices for analysis
            try:
                self.user_post_matrix = pd.read_csv(PROCESSED_DIR / "user_post_matrix.csv", index_col=0)
                logger.info(f"✅ Loaded user-post interaction matrix: {self.user_post_matrix.shape}")
            except:
                logger.warning("⚠️ User-post matrix not found, creating sample data")
                self.user_post_matrix = pd.DataFrame({
                    '1': [1, 0, 1],
                    '2': [0, 1, 1], 
                    '3': [1, 1, 0]
                }, index=['1000', '1001', '1002'])
            
            logger.info("✅ Data loading completed successfully")
        
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def recommend_posts_for_user_neural(self, user_id, user_context=None, num_recommendations=10):
        """
        Generate post recommendations using the trained PyTorch hybrid model.
        
        Args:
            user_id (str): The user ID
            user_context (dict): Optional user context (demographics, preferences)
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended posts with scores
        """
        logger.info(f"🤖 Generating neural recommendations for user {user_id}")
        
        try:
            # Encode user ID
            if user_id not in self.user_encoder.classes_:
                logger.warning(f"User {user_id} not seen during training, using fallback recommendations")
                return self._get_fallback_recommendations(user_context, num_recommendations)
            
            user_encoded = self.user_encoder.transform([user_id])[0]
            
            # Prepare user content features (16 dimensions as per model config)
            user_content = self._prepare_user_content_features(user_id, user_context)
            
            # Get all posts and companies for scoring
            post_scores = []
            
            # Score all available posts
            for post_id in range(self.num_posts):
                for company_id in range(self.num_companies):
                    # Prepare content features
                    post_content = self._prepare_post_content_features(post_id)
                    company_content = self._prepare_company_content_features(company_id)
                    
                    # Prepare tensors
                    user_tensor = torch.tensor([user_encoded], dtype=torch.long)
                    post_tensor = torch.tensor([post_id], dtype=torch.long)
                    company_tensor = torch.tensor([company_id], dtype=torch.long)
                    user_content_tensor = torch.tensor([user_content], dtype=torch.float32)
                    post_content_tensor = torch.tensor([post_content], dtype=torch.float32)
                    company_content_tensor = torch.tensor([company_content], dtype=torch.float32)
                    
                    # Get prediction
                    with torch.no_grad():
                        score = self.model(
                            user_tensor, post_tensor, company_tensor,
                            user_content_tensor, post_content_tensor, company_content_tensor
                        ).item()
                    
                    post_scores.append((post_id, company_id, score))
            
            # Sort by score and get top recommendations
            post_scores.sort(key=lambda x: x[2], reverse=True)
            top_recommendations = post_scores[:num_recommendations]
            
            # Format results
            result = []
            for post_id, company_id, score in top_recommendations:
                # Get post information
                post_info = self._get_post_info(post_id, company_id)
                result.append({
                    "PostID": post_id,
                    "CompanyID": company_id,
                    "Score": float(score),
                    **post_info
                })
            
            logger.info(f"✅ Generated {len(result)} neural recommendations for user {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating neural recommendations: {e}")
            return self._get_fallback_recommendations(user_context, num_recommendations)
    
    def _prepare_user_content_features(self, user_id, user_context=None):
        """Prepare 16-dimensional user content features."""
        features = np.zeros(16)
        
        try:
            # Use user preferences if available
            if str(user_id) in self.user_preferences_df.index:
                user_prefs = self.user_preferences_df.loc[str(user_id)]
                # Encode categorical preferences (simplified)
                if 'PreferredIndustries' in user_prefs:
                    industry = str(user_prefs['PreferredIndustries'])
                    features[0] = hash(industry) % 10 / 10.0  # Normalized hash
                if 'PreferredSupplierType' in user_prefs:
                    supplier_type = str(user_prefs['PreferredSupplierType'])
                    features[1] = hash(supplier_type) % 10 / 10.0
            
            # Use provided context
            if user_context:
                if 'age' in user_context:
                    features[2] = min(user_context['age'] / 100.0, 1.0)
                if 'location' in user_context:
                    features[3] = hash(str(user_context['location'])) % 10 / 10.0
                if 'business_size' in user_context:
                    features[4] = hash(str(user_context['business_size'])) % 10 / 10.0
            
            # Economic context features
            if self.economic_context:
                features[5] = self.economic_context.get('gdp_growth', 0) / 10.0
                features[6] = self.economic_context.get('inflation', 0) / 20.0
                features[7] = self.economic_context.get('population_growth', 0) / 5.0
            
            # Fill remaining with random values (representing other user features)
            features[8:] = np.random.rand(8) * 0.1
            
        except Exception as e:
            logger.warning(f"Error preparing user features: {e}")
            features = np.random.rand(16) * 0.1
        
        return features
    
    def _prepare_post_content_features(self, post_id):
        """Prepare 103-dimensional post content features."""
        features = np.zeros(103)
        
        try:
            # Try to get real post data
            if post_id in self.posts_df.index:
                post_data = self.posts_df.loc[post_id]
                
                # Encode industry (simplified)
                if 'Industry' in post_data:
                    industry_hash = hash(str(post_data['Industry'])) % 50
                    features[industry_hash] = 1.0
                
                # Encode engagement
                if 'Engagement' in post_data:
                    features[50] = min(float(post_data['Engagement']) / 1000.0, 1.0)
                
                # Encode title features (simplified)
                if 'PostTitle' in post_data:
                    title_hash = hash(str(post_data['PostTitle'])) % 20
                    features[51 + title_hash] = 1.0
            
            # Fill remaining with small random values
            remaining_features = 103 - np.count_nonzero(features)
            if remaining_features > 0:
                random_indices = np.random.choice(
                    np.where(features == 0)[0], 
                    min(remaining_features, 10), 
                    replace=False
                )
                features[random_indices] = np.random.rand(len(random_indices)) * 0.1
                
        except Exception as e:
            logger.warning(f"Error preparing post features: {e}")
            features = np.random.rand(103) * 0.1
        
        return features
    
    def _prepare_company_content_features(self, company_id):
        """Prepare 4-dimensional company content features."""
        features = np.zeros(4)
        
        try:
            # Use business data if available
            if company_id < len(self.business_df):
                company_data = self.business_df.iloc[company_id]
                
                # Encode category
                if 'Category' in company_data:
                    features[0] = hash(str(company_data['Category'])) % 10 / 10.0
                
                # Encode location
                if 'Location' in company_data:
                    features[1] = hash(str(company_data['Location'])) % 10 / 10.0
                
                # Encode trade type
                if 'Trade Type' in company_data:
                    features[2] = hash(str(company_data['Trade Type'])) % 10 / 10.0
                
                # Company size (simplified)
                features[3] = np.random.rand() * 0.5 + 0.5
            else:
                features = np.random.rand(4) * 0.5
                
        except Exception as e:
            logger.warning(f"Error preparing company features: {e}")
            features = np.random.rand(4) * 0.5
        
        return features
    
    def _get_post_info(self, post_id, company_id):
        """Get post information for display."""
        try:
            if post_id in self.posts_df.index:
                post_data = self.posts_df.loc[post_id]
                return {
                    "PostTitle": post_data.get('PostTitle', f'Post {post_id}'),
                    "CompanyName": post_data.get('CompanyName', f'Company {company_id}'),
                    "Industry": post_data.get('Industry', 'Unknown'),
                    "Engagement": post_data.get('Engagement', 0)
                }
            else:
                return {
                    "PostTitle": f'Post {post_id}',
                    "CompanyName": f'Company {company_id}',
                    "Industry": 'Unknown',
                    "Engagement": 0
                }
        except:
            return {
                "PostTitle": f'Post {post_id}',
                "CompanyName": f'Company {company_id}',
                "Industry": 'Unknown',
                "Engagement": 0
            }
    
    def _get_fallback_recommendations(self, user_context=None, num_recommendations=10):
        """Get fallback recommendations when neural model can't be used."""
        logger.info("🔄 Using fallback recommendation strategy")
        
        try:
            # Try to import sample posts if available
            try:
                from src.models.sample_posts import SAMPLE_POSTS
                
                # Use sample posts for recommendations
                import random
                
                # Shuffle posts for variety
                sampled_posts = random.sample(SAMPLE_POSTS, min(len(SAMPLE_POSTS), num_recommendations))
                
                result = []
                for i, post in enumerate(sampled_posts):
                    # Assign a score that decreases slightly for each post
                    # to create a ranking effect (0.9 to 0.7)
                    score = 0.9 - (i * 0.02)
                    
                    result.append({
                        "PostID": post["post_id"],
                        "CompanyID": int(post["post_id"]) % 1000,  # Generate a company ID
                        "Score": score,
                        "PostTitle": post["title"],
                        "CompanyName": post["company"],
                        "Industry": post["industry"],
                        "Engagement": 100 - i*5,  # Decreasing engagement
                        "RecommendationReason": "Based on your interests"
                    })
                
                return result
                
            except (ImportError, ModuleNotFoundError):
                logger.warning("Sample posts not available, using generic fallback")
                pass
            
            # Get top posts by engagement
            if hasattr(self, 'company_posts_df') and not self.company_posts_df.empty:
                if 'Engagement' in self.company_posts_df.columns:
                    top_posts = self.company_posts_df.nlargest(num_recommendations, 'Engagement')
                    
                    result = []
                    for idx, post_data in top_posts.iterrows():
                        result.append({
                            "PostID": idx,
                            "CompanyID": 0,
                            "Score": 0.7,  # Default score
                            "PostTitle": post_data.get('PostTitle', f'Post {idx}'),
                            "CompanyName": post_data.get('CompanyName', 'Unknown'),
                            "Industry": post_data.get('Industry', 'Unknown'),
                            "Engagement": post_data.get('Engagement', 0),
                            "RecommendationReason": "Popular in Egyptian business community"
                        })
                    
                    return result
            
            # Ultimate fallback with more diverse industries
            industries = ["Technology", "Textiles & Garments", "Tourism & Hospitality", 
                         "Agriculture & Food", "Manufacturing", "Healthcare & Pharmaceuticals"]
            
            return [{
                "PostID": str(i+1000),
                "CompanyID": i+100,
                "Score": 0.85 - (i * 0.05),
                "PostTitle": f"Egyptian {industries[i % len(industries)]} Showcase",
                "CompanyName": f"Egypt {industries[i % len(industries)]} Ltd.",
                "Industry": industries[i % len(industries)],
                "Engagement": 100 - i*10,
                "RecommendationReason": "Popular in Egyptian business community"
            } for i in range(num_recommendations)]
            
        except Exception as e:
            logger.error(f"Error in fallback recommendations: {e}")
            return []
    
    def recommend(self, user_id, user_context=None, num_recommendations=10):
        """
        Enhanced recommendation method using hybrid neural model with dwell time adjustment.
        
        Args:
            user_id (str): The user ID
            user_context (dict): Optional user context (demographics, preferences)
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended posts with scores and metadata
        """
        logger.info(f"🔍 Generating enhanced recommendations for user {user_id}")
        
        try:
            # Get base recommendations from neural model
            base_recommendations = self.recommend_posts_for_user_neural(user_id, user_context, num_recommendations * 2)
            
            # Check for dwell time data to enhance recommendations
            dwell_time_path = Path("data/processed/dwell_time_log.csv")
            if dwell_time_path.exists():
                try:
                    # Load dwell time data
                    dwell_df = pd.read_csv(dwell_time_path)
                    
                    # Filter for this user's data
                    user_dwell = dwell_df[dwell_df['UserID'] == user_id]
                    
                    if not user_dwell.empty:
                        logger.info(f"Found dwell time data for user {user_id}, enhancing recommendations")
                        
                        # Identify posts with high dwell times
                        high_dwell_posts = user_dwell[user_dwell['AvgDwellTime'] > 30]  # Posts viewed for >30 seconds
                        
                        if not high_dwell_posts.empty:
                            # Get collaborative filtering recommendations based on users with similar viewing patterns
                            similar_users_recommendations = self._get_similar_users_recommendations(user_id, high_dwell_posts, num_recommendations)
                            
                            # Adjust scores of base recommendations based on dwell time patterns
                            for rec in base_recommendations:
                                post_id = rec.get('PostID')
                                if post_id in high_dwell_posts['PostID'].values:
                                    # Boost score for posts similar to those with high dwell time
                                    rec['Score'] = min(1.0, rec['Score'] * 1.2)  # 20% boost, capped at 1.0
                                    rec['RecommendationReason'] = "Based on content you've spent more time viewing"
                            
                            # Add recommendations from similar users if not in base set
                            for rec in similar_users_recommendations:
                                if not any(r.get('PostID') == rec.get('PostID') for r in base_recommendations):
                                    base_recommendations.append(rec)
                
                except Exception as e:
                    logger.warning(f"Error processing dwell time data: {e}")
            
            # Sort by score and limit to requested number
            base_recommendations.sort(key=lambda x: x.get('Score', 0), reverse=True)
            final_recommendations = base_recommendations[:num_recommendations]
            
            # Add variety to recommendations if we have extra capacity
            if len(final_recommendations) < num_recommendations:
                # Get some recommendations from different categories
                diverse_recs = self._get_diverse_recommendations(user_id, num_recommendations - len(final_recommendations))
                
                # Add them if not already in recommendations
                for rec in diverse_recs:
                    if not any(r.get('PostID') == rec.get('PostID') for r in final_recommendations):
                        final_recommendations.append(rec)
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Error in enhanced recommendations: {e}")
            return self.recommend_posts_for_user_neural(user_id, user_context, num_recommendations)
    
    def _get_similar_users_recommendations(self, user_id, high_dwell_posts, num_recommendations=5):
        """
        Get recommendations based on users with similar viewing patterns.
        
        Args:
            user_id (str): The user ID
            high_dwell_posts (DataFrame): Posts with high dwell times for this user
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended posts
        """
        try:
            # Load interaction data
            interactions_path = Path("data/processed/user_interactions_log.csv")
            if not interactions_path.exists():
                return []
            
            interactions_df = pd.read_csv(interactions_path)
            
            # Find users who have interacted with the same posts
            high_dwell_post_ids = high_dwell_posts['PostID'].values
            
            # Get users who also viewed these posts with high dwell time
            similar_users = interactions_df[
                (interactions_df['PostID'].isin(high_dwell_post_ids)) & 
                (interactions_df['UserID'] != user_id) & 
                (interactions_df['InteractionType'] == 'view')
            ]['UserID'].unique()
            
            if len(similar_users) == 0:
                return []
            
            # Find posts that similar users have interacted with positively
            similar_user_favs = interactions_df[
                (interactions_df['UserID'].isin(similar_users)) & 
                (interactions_df['InteractionType'].isin(['like', 'rate', 'save']))
            ]
            
            # Group by post and count interactions as a measure of popularity
            post_counts = similar_user_favs['PostID'].value_counts()
            
            # Get top posts that the current user hasn't seen
            user_seen_posts = interactions_df[interactions_df['UserID'] == user_id]['PostID'].unique()
            
            # Filter for posts the user hasn't seen yet
            new_posts = [pid for pid in post_counts.index if pid not in user_seen_posts]
            
            recommendations = []
            
            for post_id in new_posts[:num_recommendations]:
                # Get post details
                post_info = self._get_post_details(post_id)
                if post_info:
                    # Score is normalized popularity among similar users
                    score = min(0.95, post_counts[post_id] / post_counts.max())
                    
                    post_info['Score'] = score
                    post_info['RecommendationReason'] = "Popular with users who view similar content"
                    recommendations.append(post_info)
                
                if len(recommendations) >= num_recommendations:
                    break
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting similar users recommendations: {e}")
            return []
    
    def _get_diverse_recommendations(self, user_id, num_recommendations=3):
        """
        Get diverse recommendations from categories the user hasn't explored much.
        
        Args:
            user_id (str): The user ID
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of diverse recommended posts
        """
        try:
            # Default industries to consider
            industries = ["Technology", "Retail & Commerce", "Manufacturing", "Agriculture & Food", 
                         "Textiles & Garments", "Tourism & Hospitality", "Healthcare & Pharmaceuticals"]
            
            # Check user's interaction history
            interactions_path = Path("data/processed/user_interactions_log.csv")
            if interactions_path.exists():
                interactions_df = pd.read_csv(interactions_path)
                user_interactions = interactions_df[interactions_df['UserID'] == user_id]
                
                if not user_interactions.empty and 'PostID' in user_interactions.columns:
                    # Get post IDs the user has interacted with
                    user_post_ids = user_interactions['PostID'].unique()
                    
                    # Find the industries of these posts
                    user_post_industries = set()
                    for post_id in user_post_ids:
                        post_info = self._get_post_details(post_id)
                        if post_info and 'Industry' in post_info:
                            user_post_industries.add(post_info['Industry'])
                    
                    # Find industries the user hasn't explored
                    unexplored_industries = [ind for ind in industries if ind not in user_post_industries]
                    
                    if unexplored_industries:
                        industries = unexplored_industries
            
            # Get top posts from each unexplored industry
            recommendations = []
            
            for industry in industries:
                # Get posts from this industry
                industry_posts = self.posts_df[self.posts_df['Industry'] == industry].sample(min(2, len(self.posts_df)))
                
                for idx, post in industry_posts.iterrows():
                    recommendations.append({
                        'PostID': str(idx),
                        'PostTitle': post.get('PostTitle', post.get('Description', f"Post {idx}")),
                        'Industry': industry,
                        'Score': 0.7,  # Lower score for exploratory recommendations
                        'RecommendationReason': f"Explore new content from {industry}"
                    })
                    
                    if len(recommendations) >= num_recommendations:
                        return recommendations
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting diverse recommendations: {e}")
            return []
    
    def _get_post_details(self, post_id):
        """Get details for a specific post."""
        try:
            if hasattr(self, 'posts_df') and post_id in self.posts_df.index:
                post_data = self.posts_df.loc[post_id]
                return {
                    'PostID': str(post_id),
                    'PostTitle': post_data.get('PostTitle', post_data.get('Description', f"Post {post_id}")),
                    'Industry': post_data.get('Industry', 'Unknown'),
                    'CompanyName': post_data.get('CompanyName', 'Unknown Company'),
                }
            return None
        except Exception:
            return None
def load_recommendation_engine():
    """
    Factory function to create and return a post recommendation engine instance.
    """
    return PostRecommendationEngine()