#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

def main():
    """
    Main entry point for the Egyptian recommendation system.
    Provides a quick overview and system check.
    """
    print("🇪🇬 Egyptian Business Hybrid Recommendation System")
    print("=" * 60)
    print("🎯 Advanced AI-powered B2B recommendations for Egyptian market")
    print()
    
    # Check system requirements
    print("🔍 System Requirements Check:")
    
    # Check if we have the required data
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        return
    else:
        print("✅ Data directory found")
    
    # Check for processed data
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        files = list(processed_dir.glob("*.csv"))
        print(f"✅ Found {len(files)} processed data files")
    else:
        print("⚠️  No processed data found - run data processing first")
    
    # Check for trained models
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth"))
        if model_files:
            print(f"✅ Found {len(model_files)} trained model(s)")
        else:
            print("⚠️  No trained models found - run training first")
    else:
        print("⚠️  Models directory not found")
    
    print()
    print("🚀 Available Commands:")
    print("  📊 Data Processing:    python -m src.data_processing.data_processor")
    print("  🎯 Model Training:     python src/train.py")
    print("  🔮 Generate Predictions: python src/predict.py --user-id 1000")
    print("  🤝 Business Partners:   python src/predict.py --business-name 'Company Name'")
    print()
    
    # Quick system test if models are available
    if models_dir.exists() and list(models_dir.glob("*.pth")):
        try:
            print("🧪 Quick System Test:")
            from src.models.recommendation_engine import PostRecommendationEngine
            
            engine = PostRecommendationEngine()
            print("✅ Recommendation engine loaded successfully")
            
            # Test with sample data
            print("📋 Sample recommendation test:")
            try:
                recommendations = engine.recommend_products_for_customer("1000", 3)
                if recommendations:
                    for i, rec in enumerate(recommendations[:2], 1):
                        print(f"  {i}. {rec['Description'][:35]}... (Score: {rec['Score']:.3f})")
                    print("✅ Product recommendations working")
                else:
                    print("⚠️  No recommendations generated (check data)")
            except Exception as e:
                print(f"⚠️  Recommendation test failed: {e}")
                
        except Exception as e:
            print(f"⚠️  System test failed: {e}")
    
    print()
    print("🎉 Egyptian Business Recommendation System Ready!")
    print("📖 See README.md for detailed usage instructions")

if __name__ == "__main__":
    main()