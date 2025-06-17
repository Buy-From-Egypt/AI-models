#!/usr/bin/env python3


import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from models.hybrid_trainer import main as train_main
from data_processing.data_processor import main as process_main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Egyptian Hybrid Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=15,
        help='Number of training epochs (default: 15)'
    )
    
    parser.add_argument(
        '--gpu', 
        action='store_true',
        help='Use GPU acceleration if available'
    )
    
    parser.add_argument(
        '--process-data', 
        action='store_true',
        help='Process data before training'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to configuration file'
    )
    
    return parser.parse_args()

def main():
    """Main training pipeline."""
    args = parse_arguments()
    
    logger.info("🚀 Starting Egyptian Hybrid Recommendation System Training")
    logger.info("=" * 60)
    
    try:
        # Data processing if requested
        if args.process_data:
            logger.info("📊 Processing data...")
            process_main()
        
        # Train the model
        logger.info("🎯 Starting model training...")
        train_main()
        
        logger.info("✅ Training completed successfully!")
        logger.info("🏆 Model saved in models/hybrid_recommendation_model.pth")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
