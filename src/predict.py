#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from models.recommendation_engine import PostRecommendationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate recommendations using trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--user-id', 
        type=str,
        help='User ID to generate product recommendations for'
    )
    
    parser.add_argument(
        '--business-name', 
        type=str,
        help='Business name to generate partner recommendations for'
    )
    
    parser.add_argument(
        '--num-recommendations', 
        type=int, 
        default=10,
        help='Number of recommendations to generate (default: 10)'
    )
    
    return parser.parse_args()

def main():
    """Main inference pipeline."""
    args = parse_arguments()
    
    logger.info("🔮 Loading Egyptian Hybrid Recommendation System")
    
    try:
        # Initialize recommendation engine
        engine = PostRecommendationEngine()
        
        if args.user_id:
            # Generate product recommendations for user
            logger.info(f"🎯 Generating recommendations for user: {args.user_id}")
            recommendations = engine.recommend(
                args.user_id, 
                num_recommendations=args.num_recommendations
            )
            
            print(f"\n📋 Top {len(recommendations)} Product Recommendations:")
            print("=" * 50)
            for i, rec in enumerate(recommendations, 1):
                title = rec.get('PostTitle', rec.get('Description', 'Unknown Product'))
                score = rec.get('Score', 0)
                print(f"{i:2d}. {title[:50]:<50} (Score: {score:.3f})")
        
        elif args.business_name:
            # Generate business partner recommendations
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
            print("❌ Please provide either --user-id or --business-name")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
