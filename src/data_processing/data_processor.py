import pandas as pd
import numpy as np
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

logger = logging.getLogger(__name__)

class EgyptianDataProcessor:
    """
    Enhanced data processor for Egyptian business recommendation system.
    Handles data loading, cleaning, feature engineering, and preparation.
    """
    
    def __init__(self, data_dir: str = None):
        """Initialize the data processor"""
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent / "data"
        self.models_dir = Path(__file__).parent.parent.parent / "models"
        
        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Data storage
        self.business_data = None
        self.user_data = None
        self.interaction_data = None
        self.economic_data = None
        
        logger.info(f"Initialized EgyptianDataProcessor with data_dir: {self.data_dir}")
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available datasets"""
        datasets = {}
        
        try:
            # Load main business data
            business_file = self.data_dir / "enhanced_egypt_import_export_v2.csv"
            if business_file.exists():
                datasets['business'] = pd.read_csv(business_file)
                logger.info(f"Loaded business data: {len(datasets['business'])} records")
            
            # Load customer data
            customer_file = self.data_dir / "egyptian_customers.csv"
            if customer_file.exists():
                datasets['customers'] = pd.read_csv(customer_file)
                logger.info(f"Loaded customer data: {len(datasets['customers'])} records")
            
            # Load processed data if available
            processed_dir = self.data_dir / "processed"
            if processed_dir.exists():
                for file_path in processed_dir.glob("*.csv"):
                    key = file_path.stem
                    datasets[key] = pd.read_csv(file_path)
                    logger.info(f"Loaded processed data '{key}': {len(datasets[key])} records")
            
            # Load economic context
            economic_file = processed_dir / "latest_economic.json" if processed_dir.exists() else None
            if economic_file and economic_file.exists():
                with open(economic_file, 'r') as f:
                    datasets['economic_context'] = json.load(f)
                logger.info("Loaded economic context data")
            
            self.business_data = datasets.get('business')
            self.user_data = datasets.get('customers')
            self.economic_data = datasets.get('economic_context', {})
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {}
    
    def clean_business_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare business data"""
        if df is None or df.empty:
            logger.warning("No business data to clean")
            return pd.DataFrame()
        
        try:
            # Create a copy
            cleaned_df = df.copy()
            
            # Handle missing values
            numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(0)
            
            text_columns = cleaned_df.select_dtypes(include=['object']).columns
            cleaned_df[text_columns] = cleaned_df[text_columns].fillna('Unknown')
            
            # Standardize text fields
            if 'Description' in cleaned_df.columns:
                cleaned_df['Description'] = cleaned_df['Description'].astype(str).str.strip()
            
            if 'Country' in cleaned_df.columns:
                cleaned_df['Country'] = cleaned_df['Country'].astype(str).str.strip()
            
            # Add derived features
            if 'UnitPrice' in cleaned_df.columns and 'Quantity' in cleaned_df.columns:
                cleaned_df['TotalValue'] = cleaned_df['UnitPrice'] * cleaned_df['Quantity']
            
            # Create business categories
            if 'Description' in cleaned_df.columns:
                cleaned_df['Category'] = self._categorize_products(cleaned_df['Description'])
            
            logger.info(f"Cleaned business data: {len(cleaned_df)} records")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning business data: {e}")
            return df
    
    def _categorize_products(self, descriptions: pd.Series) -> pd.Series:
        """Categorize products based on descriptions"""
        categories = []
        
        for desc in descriptions:
            desc_lower = str(desc).lower()
            
            if any(word in desc_lower for word in ['food', 'fruit', 'vegetable', 'rice', 'oil', 'sugar']):
                categories.append('Food & Agriculture')
            elif any(word in desc_lower for word in ['textile', 'cotton', 'fabric', 'clothing', 'garment']):
                categories.append('Textiles')
            elif any(word in desc_lower for word in ['chemical', 'fertilizer', 'plastic', 'pharmaceutical']):
                categories.append('Chemicals')
            elif any(word in desc_lower for word in ['metal', 'steel', 'iron', 'aluminum', 'copper']):
                categories.append('Metals')
            elif any(word in desc_lower for word in ['electronic', 'computer', 'mobile', 'technology']):
                categories.append('Electronics')
            elif any(word in desc_lower for word in ['marble', 'stone', 'construction', 'cement']):
                categories.append('Construction')
            else:
                categories.append('Other')
        
        return pd.Series(categories)
    
    def create_user_item_matrix(self, interaction_data: pd.DataFrame = None) -> pd.DataFrame:
        """Create user-item interaction matrix"""
        if interaction_data is None:
            # Create synthetic interaction data if none provided
            interaction_data = self._generate_synthetic_interactions()
        
        try:
            # Create pivot table for user-item interactions
            user_item_matrix = interaction_data.pivot_table(
                index='CustomerID' if 'CustomerID' in interaction_data.columns else 'user_id',
                columns='StockCode' if 'StockCode' in interaction_data.columns else 'item_id',
                values='Quantity' if 'Quantity' in interaction_data.columns else 'rating',
                fill_value=0
            )
            
            logger.info(f"Created user-item matrix: {user_item_matrix.shape}")
            return user_item_matrix
            
        except Exception as e:
            logger.error(f"Error creating user-item matrix: {e}")
            return pd.DataFrame()
    
    def _generate_synthetic_interactions(self) -> pd.DataFrame:
        """Generate synthetic user interactions for demonstration"""
        if self.business_data is None or self.business_data.empty:
            return pd.DataFrame()
        
        try:
            # Get unique items and create synthetic users
            items = self.business_data['StockCode'].unique()[:1000] if 'StockCode' in self.business_data.columns else []
            users = [f"user_{i}" for i in range(100)]
            
            interactions = []
            np.random.seed(42)  # For reproducibility
            
            for user in users:
                # Each user interacts with 5-20 random items
                n_interactions = np.random.randint(5, 21)
                user_items = np.random.choice(items, size=n_interactions, replace=False)
                
                for item in user_items:
                    interactions.append({
                        'CustomerID': user,
                        'StockCode': item,
                        'Quantity': np.random.randint(1, 10),
                        'rating': np.random.randint(1, 6)
                    })
            
            interaction_df = pd.DataFrame(interactions)
            logger.info(f"Generated {len(interaction_df)} synthetic interactions")
            return interaction_df
            
        except Exception as e:
            logger.error(f"Error generating synthetic interactions: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Prepare features for machine learning"""
        if df is None or df.empty:
            return np.array([]), {}
        
        try:
            feature_data = df.copy()
            features = []
            feature_info = {'columns': [], 'encoders': {}, 'scalers': {}}
            
            # Numeric features
            numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                numeric_features = feature_data[numeric_cols].values
                numeric_features = self.scaler.fit_transform(numeric_features)
                features.append(numeric_features)
                feature_info['columns'].extend(numeric_cols.tolist())
                feature_info['scalers']['numeric'] = self.scaler
            
            # Categorical features
            categorical_cols = feature_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                encoded_values = self.label_encoders[col].fit_transform(feature_data[col].astype(str))
                features.append(encoded_values.reshape(-1, 1))
                feature_info['columns'].append(col)
                feature_info['encoders'][col] = self.label_encoders[col]
            
            # Combine all features
            if features:
                combined_features = np.hstack(features)
                logger.info(f"Prepared features: {combined_features.shape}")
                return combined_features, feature_info
            else:
                return np.array([]), feature_info
                
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.array([]), {}
    
    def save_processed_data(self, data: Dict[str, Any], filename: str = "processed_data.pkl"):
        """Save processed data to disk"""
        try:
            save_path = self.models_dir / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(data, save_path)
            logger.info(f"Saved processed data to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def load_processed_data(self, filename: str = "processed_data.pkl") -> Dict[str, Any]:
        """Load processed data from disk"""
        try:
            load_path = self.models_dir / filename
            if load_path.exists():
                data = joblib.load(load_path)
                logger.info(f"Loaded processed data from {load_path}")
                return data
            else:
                logger.warning(f"Processed data file not found: {load_path}")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return {}
    
    def get_economic_context(self) -> Dict[str, Any]:
        """Get current economic context for Egypt"""
        default_context = {
            "gdp_growth": 3.5,
            "inflation_rate": 7.2,
            "exchange_rate_usd": 30.5,
            "export_volume_growth": 12.3,
            "import_volume_growth": 8.7,
            "trade_balance": -15.2,
            "industrial_production_index": 115.3,
            "agricultural_output_index": 108.7,
            "services_growth": 4.2,
            "unemployment_rate": 7.8,
            "last_updated": datetime.now().isoformat()
        }
        
        return self.economic_data if self.economic_data else default_context
    
    def create_business_similarity_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Create business similarity matrix based on features"""
        if df is None or df.empty:
            return np.array([])
        
        try:
            # Prepare features for similarity calculation
            features, _ = self.prepare_features(df)
            
            if features.size == 0:
                return np.array([])
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(features)
            
            logger.info(f"Created business similarity matrix: {similarity_matrix.shape}")
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Error creating similarity matrix: {e}")
            return np.array([])
    
    def process_all(self) -> Dict[str, Any]:
        """Process all data and return results"""
        logger.info("Starting comprehensive data processing...")
        
        # Load all data
        datasets = self.load_all_data()
        
        if not datasets:
            logger.warning("No data loaded, returning empty results")
            return {}
        
        results = {
            'datasets': datasets,
            'processed_data': {},
            'feature_info': {},
            'economic_context': self.get_economic_context()
        }
        
        # Process business data
        if 'business' in datasets:
            cleaned_business = self.clean_business_data(datasets['business'])
            if not cleaned_business.empty:
                results['processed_data']['business'] = cleaned_business
                
                # Create features
                features, feature_info = self.prepare_features(cleaned_business)
                results['processed_data']['business_features'] = features
                results['feature_info']['business'] = feature_info
                
                # Create similarity matrix
                similarity_matrix = self.create_business_similarity_matrix(cleaned_business)
                if similarity_matrix.size > 0:
                    results['processed_data']['business_similarity'] = similarity_matrix
        
        # Create user-item matrix
        user_item_matrix = self.create_user_item_matrix()
        if not user_item_matrix.empty:
            results['processed_data']['user_item_matrix'] = user_item_matrix
        
        logger.info("Data processing completed successfully")
        return results


def main():
    """Main function for testing the data processor"""
    processor = EgyptianDataProcessor()
    results = processor.process_all()
    
    print("✅ Data Processing Results:")
    print(f"📊 Datasets loaded: {len(results.get('datasets', {}))}")
    print(f"🔧 Processed data components: {len(results.get('processed_data', {}))}")
    print(f"📈 Economic indicators available: {len(results.get('economic_context', {}))}")
    
    if 'business' in results.get('processed_data', {}):
        business_data = results['processed_data']['business']
        print(f"🏢 Business records: {len(business_data)}")
        print(f"🏷️  Business categories: {business_data['Category'].nunique() if 'Category' in business_data.columns else 0}")


if __name__ == "__main__":
    main()
