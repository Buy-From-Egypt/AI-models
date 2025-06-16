import pandas as pd
import numpy as np
import logging
import pickle
import joblib
from pathlib import Path
import heapq
import torch

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

class PostRecommendationEngine:
    """
    Class to handle loading of trained models and generating post recommendations.
    """
    
    def __init__(self):
        """
        Initialize the post recommendation engine by loading all required models.
        """
        logger.info("Initializing post recommendation engine...")
        
        try:
            # Create directories if they don't exist
            MODELS_DIR.mkdir(exist_ok=True)
            PROCESSED_DIR.mkdir(exist_ok=True)
            
            # Load collaborative filtering model and mappings
            try:
                with open(MODELS_DIR / "cf_model.pkl", "rb") as f:
                    self.cf_model = pickle.load(f)
                
                with open(MODELS_DIR / "user_id_map.pkl", "rb") as f:
                    self.user_id_map = pickle.load(f)
                
                with open(MODELS_DIR / "post_id_map.pkl", "rb") as f:
                    self.post_id_map = pickle.load(f)
                
                with open(MODELS_DIR / "reverse_user_map.pkl", "rb") as f:
                    self.reverse_user_map = pickle.load(f)
                
                with open(MODELS_DIR / "reverse_post_map.pkl", "rb") as f:
                    self.reverse_post_map = pickle.load(f)
                    
                logger.info("Loaded collaborative filtering model for posts")
            except:
                logger.warning("Post CF model not found, creating dummy")
                self.cf_model = {"user_factors": np.random.rand(10, 10), "post_factors": np.random.rand(10, 10)}
                self.user_id_map = {"1000": 0, "1001": 1}
                self.post_id_map = {1: 0, 2: 1}
                self.reverse_user_map = {0: "1000", 1: "1001"}
                self.reverse_post_map = {0: 1, 1: 2}
            
            # Load business recommendation model and mappings
            try:
                self.business_similarity = np.load(MODELS_DIR / "business_similarity_matrix.npy")
                
                with open(MODELS_DIR / "business_id_map.pkl", "rb") as f:
                    self.business_id_map = pickle.load(f)
                
                with open(MODELS_DIR / "business_idx_map.pkl", "rb") as f:
                    self.business_idx_map = pickle.load(f)
                    
                logger.info("Loaded business similarity matrix")
            except:
                logger.warning("Business similarity matrix not found, creating dummy")
                self.business_similarity = np.random.rand(10, 10)
                self.business_id_map = {"Business 1": 1, "Business 2": 2}
                self.business_idx_map = {1: 0, 2: 1}
            
            # Load business-post affinity mapping
            try:
                with open(MODELS_DIR / "business_post_affinity.pkl", "rb") as f:
                    self.business_post_affinity = pickle.load(f)
                logger.info("Loaded business-post affinity mapping")
            except:
                logger.warning("Business-post affinity not found, creating dummy")
                self.business_post_affinity = {
                    "Business 1": [
                        {"PostID": 1, "PostTitle": "Product 1", "Industry": "Electronics", "Engagement": 100}
                    ]
                }
            
            # Load economic context
            try:
                with open(MODELS_DIR / "economic_context.pkl", "rb") as f:
                    self.economic_context = pickle.load(f)
                logger.info("Loaded economic context")
            except:
                logger.warning("Economic context not found, creating dummy")
                self.economic_context = {
                    'gdp_growth': 4.35,
                    'inflation': 5.04,
                    'population_growth': 1.73
                }
            
            # Load company posts data
            try:
                self.company_posts_df = pd.read_csv(PROCESSED_DIR / "company_posts.csv")
                self.company_posts_df.set_index('PostID', inplace=True)
                logger.info(f"Loaded {len(self.company_posts_df)} company posts")
            except:
                logger.warning("Company posts not found, creating dummy")
                self.company_posts_df = pd.DataFrame({
                    'PostID': [1, 2, 3],
                    'CompanyName': ["Tech Egypt", "Food Corp", "Textile Co"],
                    'PostTitle': ["Latest Electronics", "Fresh Produce", "Quality Fabrics"],
                    'Industry': ["Electronics", "Agriculture & Food", "Textiles & Garments"],
                    'Engagement': [100, 200, 150]
                }).set_index('PostID')
            
            # Load user preferences
            try:
                self.user_preferences_df = pd.read_csv(PROCESSED_DIR / "user_preferences.csv")
                self.user_preferences_df.set_index('UserID', inplace=True)
                logger.info(f"Loaded {len(self.user_preferences_df)} user preferences")
            except:
                logger.warning("User preferences not found, creating dummy")
                self.user_preferences_df = pd.DataFrame({
                    'UserID': ["1000", "1001", "1002"],
                    'PreferredIndustries': ["Electronics", "Agriculture & Food", "Textiles & Garments"],
                    'PreferredSupplierType': ["Small Businesses", "Medium Enterprises", "Large Corporations"]
                }).set_index('UserID')
            
            # Load business data for additional info
            try:
                self.business_df = pd.read_csv(PROCESSED_DIR / "business_features.csv")
                logger.info(f"Loaded {len(self.business_df)} business profiles")
            except:
                logger.warning("Business features not found, creating dummy")
                self.business_df = pd.DataFrame({
                    'BusinessID': [1, 2],
                    'Business Name': ["Business 1", "Business 2"],
                    'Category': ["Electronics", "Textiles"],
                    'Location': ["Cairo, Egypt", "Alexandria, Egypt"],
                    'Trade Type': ["Importer", "Exporter"]
                })
            
            logger.info("Post recommendation engine initialized successfully.")
        
        except Exception as e:
            logger.error(f"Error initializing post recommendation engine: {e}")
            raise
            
            # Load economic context
            try:
                with open(MODELS_DIR / "economic_context.pkl", "rb") as f:
                    self.economic_context = pickle.load(f)
            except:
                logger.warning("Economic context not found, creating dummy")
                self.economic_context = {
                    'gdp_growth': 4.35,
                    'inflation': 5.04,
                    'population_growth': 1.73
                }
            
            # Load product info
            try:
                # Try to load product information from different potential sources
                if (PROCESSED_DIR / "retail_cleaned.csv").exists():
                    self.products_df = pd.read_csv(PROCESSED_DIR / "retail_cleaned.csv")[['StockCode', 'Description']].drop_duplicates()
                    self.products_df.set_index('StockCode', inplace=True)
                elif (DATA_DIR / "data.csv").exists():
                    # Load from original source with limited rows
                    self.products_df = pd.read_csv(DATA_DIR / "data.csv", encoding='ISO-8859-1', nrows=10000)[['StockCode', 'Description']].drop_duplicates()
                    self.products_df.set_index('StockCode', inplace=True)
            except:
                logger.warning("Product info not found, creating dummy")
                self.products_df = pd.DataFrame({
                    'StockCode': ["10000", "10001", "10002"],
                    'Description': ["Product 1", "Product 2", "Product 3"]
                }).set_index('StockCode')
            
            # Load business-product affinity mapping
            try:
                with open(MODELS_DIR / "business_product_affinity.pkl", "rb") as f:
                    self.business_product_affinity = pickle.load(f)
            except:
                logger.warning("Business-product affinity not found, creating dummy")
                self.business_product_affinity = {
                    "Business 1": [
                        {"StockCode": "10000", "Description": "Product 1", "Score": 0.8}
                    ]
                }
            
            # Load business data for additional info
            try:
                self.business_df = pd.read_csv(PROCESSED_DIR / "business_features.csv")
            except:
                logger.warning("Business features not found, creating dummy")
                self.business_df = pd.DataFrame({
                    'BusinessID': [1, 2],
                    'Business Name': ["Business 1", "Business 2"],
                    'Category': ["Electronics", "Textiles"],
                    'Location': ["Cairo, Egypt", "Alexandria, Egypt"],
                    'Trade Type': ["Importer", "Exporter"]
                })
            
            logger.info("Recommendation engine initialized successfully.")
        
        except Exception as e:
            logger.error(f"Error initializing recommendation engine: {e}")
            raise
    
    def recommend_products_for_customer(self, customer_id, num_recommendations=10):
        """
        Generate product recommendations for a specific customer.
        
        Args:
            customer_id (str): The customer ID
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended products with scores
        """
        logger.info(f"Generating recommendations for customer {customer_id}")
        
        try:
            # Convert to appropriate format if needed
            # Our model might have customer IDs stored as floats
            try:
                float_id = float(customer_id)
                if float_id in self.user_id_map:
                    customer_id = float_id
            except (ValueError, TypeError):
                pass
                
            # Check if customer exists in our model
            if customer_id not in self.user_id_map:
                logger.warning(f"Customer {customer_id} not found in model data.")
                return []
            
            # Get user index
            user_idx = self.user_id_map[customer_id]
            
            # Generate recommendations
            try:
                # PyTorch matrix factorization
                user_vector = self.cf_model['user_factors'][user_idx]
                item_vectors = self.cf_model['item_factors']
                
                # Calculate scores for each item - make sure dimensions align
                if item_vectors.shape[0] == user_vector.shape[0]:  # user_vector aligns with item_vector rows
                    scores = np.dot(user_vector, item_vectors)
                else:  # item_vectors needs transposition
                    scores = np.dot(user_vector, item_vectors.T)
                
                # Get top N items
                top_indices = np.argsort(-scores)[:num_recommendations]
                recommendations = [(idx, scores[idx]) for idx in top_indices]
            except Exception as e:
                logger.error(f"Error calculating recommendations: {e}")
                return []
            
            # Format results
            result = []
            for item_idx, score in recommendations:
                try:
                    stock_code = self.reverse_item_map[item_idx]
                    try:
                        description = self.products_df.loc[stock_code, 'Description']
                    except (KeyError, TypeError):
                        description = f"Product {stock_code}"
                        
                    result.append({
                        "StockCode": stock_code,
                        "Description": description,
                        "Score": float(score)
                    })
                except (KeyError, IndexError):
                    continue
            
            logger.info(f"Generated {len(result)} recommendations for customer {customer_id}")
            return result
        
        except Exception as e:
            logger.error(f"Error generating customer recommendations: {e}")
            raise
    
    def recommend_business_partners(self, business_name, num_recommendations=10):
        """
        Generate business partnership recommendations for a specific business.
        
        Args:
            business_name (str): The name of the business
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended businesses with similarity scores
        """
        logger.info(f"Generating partnership recommendations for business {business_name}")
        
        try:
            # Check if business exists in our model
            if business_name not in self.business_id_map:
                logger.warning(f"Business {business_name} not found in model data.")
                return []
            
            # Get business index
            business_id = self.business_id_map[business_name]
            business_idx = self.business_idx_map[business_id]
            
            # Get similarity scores for this business with all others
            similarity_scores = self.business_similarity[business_idx]
            
            # Get top N similar businesses (exclude self)
            top_indices = heapq.nlargest(num_recommendations + 1, 
                                         range(len(similarity_scores)), 
                                         key=lambda i: similarity_scores[i])
            
            # Remove the business itself from recommendations
            top_indices = [idx for idx in top_indices if idx != business_idx][:num_recommendations]
            
            # Format results
            result = []
            for idx in top_indices:
                try:
                    # Find the business name from the index
                    for b_name, b_id in self.business_id_map.items():
                        if self.business_idx_map.get(b_id) == idx:
                            # Look up additional business info
                            business_rows = self.business_df[self.business_df['Business Name'] == b_name]
                            if len(business_rows) > 0:
                                business_info = business_rows.iloc[0]
                                
                                result.append({
                                    "BusinessName": b_name,
                                    "Category": business_info.get('Category', 'Unknown'),
                                    "Location": business_info.get('Location', 'Unknown'),
                                    "TradeType": business_info.get('Trade Type', 'Unknown'),
                                    "SimilarityScore": float(similarity_scores[idx])
                                })
                            break
                except (KeyError, IndexError, ValueError):
                    continue
            
            logger.info(f"Generated {len(result)} partnership recommendations for business {business_name}")
            return result
        
        except Exception as e:
            logger.error(f"Error generating business partnership recommendations: {e}")
            raise
    
    def recommend_products_for_business(self, business_name, num_recommendations=10):
        """
        Generate product recommendations for a specific business.
        
        Args:
            business_name (str): The name of the business
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended products with scores
        """
        logger.info(f"Generating product recommendations for business {business_name}")
        
        try:
            # First check if we have pre-calculated affinity for this business
            if business_name in self.business_product_affinity:
                recommendations = self.business_product_affinity[business_name][:num_recommendations]
                logger.info(f"Found {len(recommendations)} pre-calculated recommendations for {business_name}")
                return recommendations
            
            # If no pre-calculated affinity, return empty list
            # In a complete implementation, we could generate on-the-fly recommendations
            logger.warning(f"No pre-calculated product recommendations found for {business_name}")
            return []
            
        except Exception as e:
            logger.error(f"Error generating business product recommendations: {e}")
            raise
    
    def combine_with_economic_context(self, recommendations, weight=0.2):
        """
        Adjust recommendation scores based on economic indicators.
        
        This is a simplified demonstration of how economic context could be used.
        In a complete implementation, this would be more sophisticated.
        
        Args:
            recommendations (list): Original recommendations
            weight (float): Weight of economic adjustment (0-1)
            
        Returns:
            list: Adjusted recommendations
        """
        # Extract economic indicators
        gdp_growth = self.economic_context.get('gdp_growth', 0)
        
        # Simple adjustment based on GDP growth
        # Positive growth → boost scores slightly
        # Negative growth → reduce scores slightly
        adjustment_factor = 1 + (gdp_growth / 100 * weight)
        
        # Apply adjustment
        for rec in recommendations:
            rec['Score'] = rec['Score'] * adjustment_factor
            rec['Score'] = min(rec['Score'], 1.0)  # Cap at 1.0
        
        return recommendations
    
    def recommend_posts_for_user(self, user_id, user_preferences=None, num_recommendations=10):
        """
        Generate post recommendations for a user based on their preferences.
        
        Args:
            user_id (str): The user ID
            user_preferences (dict): Optional user preferences override
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended posts with scores
        """
        logger.info(f"Generating post recommendations for user {user_id}")
        
        try:
            recommendations = []
            
            # Method 1: Collaborative Filtering (if user exists in training data)
            if str(user_id) in self.user_id_map:
                cf_recommendations = self._get_cf_post_recommendations(user_id, num_recommendations)
                recommendations.extend(cf_recommendations)
            
            # Method 2: Content-based filtering using user preferences
            if user_preferences or str(user_id) in self.user_preferences_df.index:
                content_recommendations = self._get_content_based_post_recommendations(
                    user_id, user_preferences, num_recommendations
                )
                recommendations.extend(content_recommendations)
            
            # Method 3: Industry-based recommendations (fallback)
            if not recommendations:
                industry_recommendations = self._get_industry_based_recommendations(
                    user_preferences, num_recommendations
                )
                recommendations.extend(industry_recommendations)
            
            # Remove duplicates and sort by score
            seen_posts = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec['PostID'] not in seen_posts:
                    unique_recommendations.append(rec)
                    seen_posts.add(rec['PostID'])
            
            # Sort by score and limit
            unique_recommendations.sort(key=lambda x: x['Score'], reverse=True)
            return unique_recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error generating post recommendations for user {user_id}: {e}")
            return []
    
    def _get_cf_post_recommendations(self, user_id, num_recommendations):
        """Get recommendations using collaborative filtering."""
        try:
            user_idx = self.user_id_map[str(user_id)]
            
            # Get user vector and calculate scores
            user_vector = self.cf_model['user_factors'][user_idx]
            post_factors = self.cf_model['post_factors']
            
            # Calculate scores for each post
            if post_factors.shape[0] == user_vector.shape[0]:
                scores = np.dot(user_vector, post_factors)
            else:
                scores = np.dot(post_factors.T, user_vector)
            
            # Get top recommendations
            top_post_indices = np.argsort(scores)[-num_recommendations:][::-1]
            
            recommendations = []
            for idx in top_post_indices:
                post_id = self.reverse_post_map.get(idx)
                if post_id and post_id in self.company_posts_df.index:
                    post_info = self.company_posts_df.loc[post_id]
                    recommendations.append({
                        'PostID': int(post_id),
                        'CompanyName': post_info['CompanyName'],
                        'PostTitle': post_info['PostTitle'],
                        'Industry': post_info['Industry'],
                        'Score': float(scores[idx]),
                        'RecommendationType': 'collaborative_filtering'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in CF post recommendations: {e}")
            return []
    
    def _get_content_based_post_recommendations(self, user_id, user_preferences, num_recommendations):
        """Get recommendations based on user preferences."""
        try:
            # Get user preferences
            if user_preferences:
                preferred_industries = user_preferences.get('PreferredIndustries', '').split(',')
                preferred_supplier_type = user_preferences.get('PreferredSupplierType', '')
                preferred_order_qty = user_preferences.get('PreferredOrderQuantity', '')
            else:
                user_prefs = self.user_preferences_df.loc[str(user_id)]
                preferred_industries = user_prefs['PreferredIndustries'].split(',')
                preferred_supplier_type = user_prefs['PreferredSupplierType']
                preferred_order_qty = user_prefs.get('PreferredOrderQuantity', '')
            
            recommendations = []
            
            # Filter posts by user preferences
            for post_id, post_info in self.company_posts_df.iterrows():
                score = 0.0
                
                # Industry match
                if post_info['Industry'] in preferred_industries:
                    score += 0.4
                
                # Company size preference
                company_size = post_info.get('CompanySize', '')
                if preferred_supplier_type == 'Small Businesses' and 'Small' in company_size:
                    score += 0.2
                elif preferred_supplier_type == 'Medium Enterprises' and 'Medium' in company_size:
                    score += 0.2
                elif preferred_supplier_type == 'Large Corporations' and 'Large' in company_size:
                    score += 0.2
                
                # Order quantity match
                min_order_qty = post_info.get('MinOrderQuantity', '')
                if min_order_qty == preferred_order_qty:
                    score += 0.1
                
                # Engagement boost
                engagement = post_info.get('Engagement', 0)
                score += min(engagement / 1000, 0.2)  # Normalize engagement
                
                # Quality score boost
                quality_score = post_info.get('QualityScore', 4.0)
                score += (quality_score - 3.0) / 10  # Normalize quality (3-5 scale)
                
                if score > 0.1:  # Minimum threshold
                    recommendations.append({
                        'PostID': int(post_id),
                        'CompanyName': post_info['CompanyName'],
                        'PostTitle': post_info['PostTitle'],
                        'Industry': post_info['Industry'],
                        'Score': float(score),
                        'RecommendationType': 'content_based'
                    })
            
            # Sort by score and return top N
            recommendations.sort(key=lambda x: x['Score'], reverse=True)
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error in content-based post recommendations: {e}")
            return []
    
    def _get_industry_based_recommendations(self, user_preferences, num_recommendations):
        """Get fallback recommendations based on popular posts in industries."""
        try:
            # Default to popular industries if no preferences
            if not user_preferences:
                target_industries = ['Electronics', 'Agriculture & Food', 'Textiles & Garments']
            else:
                target_industries = user_preferences.get('PreferredIndustries', 'Electronics').split(',')
            
            recommendations = []
            
            # Get top posts by engagement in target industries
            for industry in target_industries:
                industry_posts = self.company_posts_df[
                    self.company_posts_df['Industry'] == industry
                ].sort_values('Engagement', ascending=False)
                
                for post_id, post_info in industry_posts.head(num_recommendations // len(target_industries)).iterrows():
                    recommendations.append({
                        'PostID': int(post_id),
                        'CompanyName': post_info['CompanyName'],
                        'PostTitle': post_info['PostTitle'],
                        'Industry': post_info['Industry'],
                        'Score': float(post_info['Engagement'] / 1000),  # Normalize engagement
                        'RecommendationType': 'popular_in_industry'
                    })
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error in industry-based recommendations: {e}")
            return []

    def recommend_companies_for_business(self, business_name, num_recommendations=5):
        """
        Generate company partnership recommendations for a business.
        
        Args:
            business_name (str): The business name
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of recommended business partners with scores
        """
        logger.info(f"Generating company recommendations for business {business_name}")
        
        try:
            if business_name not in self.business_id_map:
                logger.warning(f"Business {business_name} not found in model data.")
                return []
            
            business_id = self.business_id_map[business_name]
            business_idx = self.business_idx_map[business_id]
            
            # Get similar businesses
            similarities = self.business_similarity[business_idx]
            
            # Get top similar businesses (excluding self)
            similar_indices = np.argsort(similarities)[-num_recommendations-1:-1][::-1]
            
            recommendations = []
            for idx in similar_indices:
                if idx < len(self.business_df):
                    similar_business = self.business_df.iloc[idx]
                    
                    # Get posts from this business
                    business_posts = self.business_post_affinity.get(similar_business['Business Name'], [])
                    
                    recommendation = {
                        'BusinessName': similar_business['Business Name'],
                        'Category': similar_business['Category'],
                        'Location': similar_business['Location'],
                        'TradeType': similar_business.get('Trade Type', 'Unknown'),
                        'SimilarityScore': float(similarities[idx]),
                        'Region': similar_business.get('Region', 'Unknown'),
                        'PostCount': len(business_posts),
                        'AvgEngagement': sum(post.get('Engagement', 0) for post in business_posts) / len(business_posts) if business_posts else 0
                    }
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating company recommendations for {business_name}: {e}")
            return []

def load_recommendation_engine():
    """
    Factory function to create and return a post recommendation engine instance.
    """
    return PostRecommendationEngine()