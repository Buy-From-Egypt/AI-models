#!/usr/bin/env python3
import argparse
import logging
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to Python path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.recommendation_engine import PostRecommendationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
INTERACTION_LOG_PATH = Path("data/processed/user_interactions_log.csv")
DWELL_TIME_LOG_PATH = Path("data/processed/dwell_time_log.csv")
USER_SIMILARITY_PATH = Path("data/processed/user_similarity_matrix.csv")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced prediction and recommendation system with dwell time tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--user-id', 
        type=str,
        help='User ID to generate recommendations for'
    )
    
    parser.add_argument(
        '--business-name', 
        type=str,
        help='Business name to generate partner recommendations for'
    )
    
    parser.add_argument(
        '--post-id',
        type=str,
        help='Post ID to record interaction with'
    )
    
    parser.add_argument(
        '--interaction-type',
        type=str,
        choices=['view', 'like', 'comment', 'share', 'rate', 'save'],
        help='Type of interaction to record'
    )
    
    parser.add_argument(
        '--rating-value',
        type=float,
        help='Rating value (1-5) if interaction type is "rate"'
    )
    
    parser.add_argument(
        '--dwell-time',
        type=int,
        help='Dwell time in seconds for the post view'
    )
    
    parser.add_argument(
        '--num-recommendations', 
        type=int, 
        default=10,
        help='Number of recommendations to generate (default: 10)'
    )
    
    parser.add_argument(
        '--include-similar-rated',
        action='store_true',
        help='Include posts similar to those with high ratings from the user'
    )
    
    return parser.parse_args()

def load_interaction_logs():
    """Load existing interaction logs or create if not exist."""
    try:
        if INTERACTION_LOG_PATH.exists():
            return pd.read_csv(INTERACTION_LOG_PATH)
        else:
            # Create new interaction log with headers
            log_df = pd.DataFrame(columns=[
                'UserID', 'PostID', 'InteractionType', 
                'Value', 'Timestamp', 'DwellTimeSeconds'
            ])
            return log_df
    except Exception as e:
        logger.error(f"Error loading interaction logs: {e}")
        return pd.DataFrame(columns=[
            'UserID', 'PostID', 'InteractionType', 
            'Value', 'Timestamp', 'DwellTimeSeconds'
        ])

def save_interaction_logs(log_df):
    """Save interaction logs to CSV."""
    try:
        # Ensure directory exists
        INTERACTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(INTERACTION_LOG_PATH, index=False)
        logger.info(f"✅ Saved interaction logs to {INTERACTION_LOG_PATH}")
    except Exception as e:
        logger.error(f"Error saving interaction logs: {e}")

def record_interaction(user_id, post_id, interaction_type, value=None, dwell_time=None):
    """
    Record a user interaction with a post.
    
    Args:
        user_id (str): User ID
        post_id (str): Post ID
        interaction_type (str): Type of interaction (view, like, comment, rate, etc.)
        value (float, optional): Rating value if interaction type is "rate"
        dwell_time (int, optional): Dwell time in seconds for view interactions
    """
    try:
        # Load existing logs
        log_df = load_interaction_logs()
        
        # Create new interaction record
        timestamp = datetime.now().isoformat()
        
        new_record = {
            'UserID': user_id,
            'PostID': post_id,
            'InteractionType': interaction_type,
            'Value': value if value is not None else 1.0,
            'Timestamp': timestamp,
            'DwellTimeSeconds': dwell_time if dwell_time is not None else 0
        }
        
        # Add to log using concat instead of deprecated append
        log_df = pd.concat([log_df, pd.DataFrame([new_record])], ignore_index=True)
        
        # Save updated logs
        save_interaction_logs(log_df)
        
        logger.info(f"✅ Recorded {interaction_type} interaction for user {user_id} with post {post_id}")
        
        # If dwell time is provided, update the specific dwell time log - only once
        if dwell_time is not None and interaction_type == 'view':
            update_dwell_time_metrics(user_id, post_id, dwell_time)
        
        return True
    except Exception as e:
        logger.error(f"Error recording interaction: {e}")
        return False

def update_dwell_time_metrics(user_id, post_id, dwell_time):
    """
    Update dwell time metrics for a specific user-post pair.
    
    Args:
        user_id (str): User ID
        post_id (str): Post ID
        dwell_time (int): Dwell time in seconds
    """
    try:
        # Load existing dwell time logs or create new
        if DWELL_TIME_LOG_PATH.exists():
            dwell_df = pd.read_csv(DWELL_TIME_LOG_PATH)
        else:
            dwell_df = pd.DataFrame(columns=['UserID', 'PostID', 'AvgDwellTime', 'TotalViews'])
        
        # Check if entry already exists
        existing = dwell_df[(dwell_df['UserID'] == user_id) & (dwell_df['PostID'] == post_id)]
        
        if len(existing) > 0:
            # Update existing entry
            idx = existing.index[0]
            current_avg = dwell_df.loc[idx, 'AvgDwellTime']
            current_views = dwell_df.loc[idx, 'TotalViews']
            
            # Calculate new average dwell time
            new_avg = ((current_avg * current_views) + dwell_time) / (current_views + 1)
            
            # Update row
            dwell_df.loc[idx, 'AvgDwellTime'] = new_avg
            dwell_df.loc[idx, 'TotalViews'] = current_views + 1
        else:
            # Add new entry using concat instead of deprecated append
            new_record = {
                'UserID': user_id,
                'PostID': post_id,
                'AvgDwellTime': dwell_time,
                'TotalViews': 1
            }
            dwell_df = pd.concat([dwell_df, pd.DataFrame([new_record])], ignore_index=True)
        
        # Ensure directory exists
        DWELL_TIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Save updated dwell time logs
        dwell_df.to_csv(DWELL_TIME_LOG_PATH, index=False)
        
        logger.info(f"✅ Updated dwell time metrics for user {user_id} with post {post_id}")
        
    except Exception as e:
        logger.error(f"Error updating dwell time metrics: {e}")

def find_similar_rated_posts(user_id, engine, top_k=5):
    """
    Find posts similar to those highly rated by the user.
    
    Args:
        user_id (str): User ID
        engine (PostRecommendationEngine): Recommendation engine instance
        top_k (int): Number of similar posts to find per rated post
    
    Returns:
        list: List of similar posts with scores
    """
    try:
        # Load interaction logs
        log_df = load_interaction_logs()
        
        # Filter for ratings by this user
        user_ratings = log_df[(log_df['UserID'] == user_id) & 
                             (log_df['InteractionType'] == 'rate') & 
                             (log_df['Value'] >= 4.0)]  # Only high ratings (4-5)
        
        if len(user_ratings) == 0:
            logger.info(f"No high ratings found for user {user_id}")
            return []
        
        similar_posts = []
        
        # For each highly rated post, find similar posts
        for _, row in user_ratings.iterrows():
            rated_post_id = row['PostID']
            
            # Find similar posts based on content
            # This uses the recommendation engine's content-based similarity
            similar = engine.find_similar_posts(rated_post_id, top_k)
            
            # Add to results if not already rated by the user
            for post in similar:
                if post['PostID'] not in user_ratings['PostID'].values and post not in similar_posts:
                    post['RecommendationReason'] = f"Similar to post {rated_post_id} that you rated highly"
                    similar_posts.append(post)
        
        return similar_posts
    
    except Exception as e:
        logger.error(f"Error finding similar rated posts: {e}")
        return []

def main():
    """Main inference pipeline with enhanced recommendation features."""
    args = parse_arguments()
    
    logger.info("🔮 Enhanced Egyptian Hybrid Recommendation System")
    
    try:
        # Initialize recommendation engine
        engine = PostRecommendationEngine()
        
        # Record interaction if specified
        if args.user_id and args.post_id and args.interaction_type:
            record_interaction(
                args.user_id, 
                args.post_id, 
                args.interaction_type,
                args.rating_value,
                args.dwell_time
            )
            
            print(f"✅ Recorded {args.interaction_type} interaction for user {args.user_id} with post {args.post_id}")
        
        # Generate recommendations if user ID is provided
        if args.user_id:
            logger.info(f"🎯 Generating enhanced recommendations for user: {args.user_id}")
            
            # Get standard recommendations
            recommendations = engine.recommend(
                args.user_id, 
                num_recommendations=args.num_recommendations
            )
            
            # If specified, also include posts similar to highly rated posts
            if args.include_similar_rated:
                similar_posts = find_similar_rated_posts(args.user_id, engine)
                
                # Add similar posts to recommendations if they're not already there
                for post in similar_posts:
                    if post not in recommendations:
                        recommendations.append(post)
                
                # Limit to requested number of recommendations
                recommendations = recommendations[:args.num_recommendations]
            
            print(f"\n📋 Top {len(recommendations)} Recommendations for User {args.user_id}:")
            print("=" * 70)
            for i, rec in enumerate(recommendations, 1):
                title = rec.get('PostTitle', rec.get('Description', 'Unknown Post'))
                score = rec.get('Score', 0)
                reason = rec.get('RecommendationReason', 'Based on your preferences')
                print(f"{i:2d}. {title[:40]:<40} | Score: {score:.3f} | {reason}")
        
        # Generate business partner recommendations if business name is provided
        elif args.business_name:
            logger.info(f"🤝 Generating partner recommendations for: {args.business_name}")
            recommendations = engine.recommend_business_partners(
                args.business_name, 
                args.num_recommendations
            )
            
            print(f"\n🏢 Top {len(recommendations)} Business Partner Recommendations:")
            print("=" * 70)
            for i, rec in enumerate(recommendations, 1):
                print(f"{i:2d}. {rec['BusinessName']:<30} | {rec['Category']:<20} | Score: {rec['SimilarityScore']:.3f}")
        
        else:
            print("❌ Please provide either --user-id or --business-name to generate recommendations")
            print("   or include --post-id and --interaction-type to record an interaction")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
