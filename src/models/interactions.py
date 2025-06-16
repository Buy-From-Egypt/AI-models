import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class UserInteractionTracker:
    """Track user interactions with posts for advanced collaborative filtering."""
    
    def __init__(self):
        self.interaction_types = {
            'view': 1.0,
            'like': 3.0, 
            'share': 4.0,
            'comment': 5.0,
            'save': 3.5,
            'click_profile': 2.0,
            'click_website': 4.5,
            'inquiry': 6.0,
            'time_spent': 0.1  # per second
        }
        
    def create_interaction_data(self, users_df, posts_df, num_interactions=200000):
        """Create realistic user interaction data."""
        logger.info(f"🎯 Creating {num_interactions:,} user interactions...")
        
        interactions = []
        
        for _ in range(num_interactions):
            user_id = np.random.choice(users_df['UserID'])
            post_id = np.random.choice(posts_df['PostID'])
            
            # Get user preferences
            user_prefs = users_df[users_df['UserID'] == user_id].iloc[0]
            post_info = posts_df[posts_df['PostID'] == post_id].iloc[0]
            
            # Calculate base interaction probability
            interaction_prob = self._calculate_interaction_probability(user_prefs, post_info)
            
            if np.random.random() < interaction_prob:
                # Determine interaction type based on engagement level
                interaction_types = self._get_likely_interactions(interaction_prob)
                
                for interaction_type in interaction_types:
                    interactions.append({
                        'UserID': user_id,
                        'PostID': post_id,
                        'InteractionType': interaction_type,
                        'InteractionValue': self.interaction_types[interaction_type],
                        'Timestamp': self._generate_timestamp(),
                        'SessionID': f"session_{np.random.randint(1000, 9999)}",
                        'DeviceType': np.random.choice(['mobile', 'tablet', 'desktop'], p=[0.7, 0.2, 0.1])
                    })
        
        interactions_df = pd.DataFrame(interactions)
        logger.info(f"✅ Created {len(interactions_df):,} interactions")
        
        return interactions_df
    
    def _calculate_interaction_probability(self, user_prefs, post_info):
        """Calculate probability of user interacting with post."""
        prob = 0.1  # base probability
        
        # Industry match
        user_industries = user_prefs['PreferredIndustries'].split(',')
        if post_info['Industry'] in user_industries:
            prob += 0.4
            
        # Company size preference
        if user_prefs['PreferredSupplierType'] in ['Small Businesses'] and post_info['CompanySize'] == 'Small':
            prob += 0.2
        elif user_prefs['PreferredSupplierType'] in ['Medium Enterprises'] and post_info['CompanySize'] == 'Medium':
            prob += 0.2
        elif user_prefs['PreferredSupplierType'] in ['Large Corporations'] and post_info['CompanySize'] == 'Large':
            prob += 0.2
            
        # Quality boost
        prob += post_info.get('QualityScore', 4.0) / 20
        
        return min(prob, 0.8)
    
    def _get_likely_interactions(self, base_prob):
        """Get likely interaction types based on engagement probability."""
        interactions = ['view']  # Everyone who interacts at least views
        
        if np.random.random() < base_prob * 0.3:
            interactions.append('like')
            
        if np.random.random() < base_prob * 0.1:
            interactions.append('share')
            
        if np.random.random() < base_prob * 0.15:
            interactions.append('comment')
            
        if np.random.random() < base_prob * 0.2:
            interactions.append('save')
            
        if np.random.random() < base_prob * 0.25:
            interactions.append('click_profile')
            
        if np.random.random() < base_prob * 0.35:
            interactions.append('click_website')
            
        if np.random.random() < base_prob * 0.05:
            interactions.append('inquiry')
            
        # Add time spent (random between 5-120 seconds)
        time_spent = np.random.uniform(5, 120)
        for _ in range(int(time_spent)):
            interactions.append('time_spent')
            
        return interactions
    
    def _generate_timestamp(self):
        """Generate realistic timestamp."""
        days_back = np.random.randint(1, 90)
        hours = np.random.randint(6, 23)  # Business hours bias
        return datetime.now() - timedelta(days=days_back, hours=hours)

class AdvancedCollaborativeFilter(nn.Module):
    """Advanced neural collaborative filtering with user interactions."""
    
    def __init__(self, num_users, num_posts, num_interactions, embedding_dim=64, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        self.num_users = num_users
        self.num_posts = num_posts
        self.num_interactions = num_interactions
        self.embedding_dim = embedding_dim
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.post_embedding = nn.Embedding(num_posts, embedding_dim)
        self.interaction_embedding = nn.Embedding(num_interactions, embedding_dim // 4)
        
        # Neural layers
        layers = []
        input_dim = embedding_dim * 2 + embedding_dim // 4
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, user_ids, post_ids, interaction_types):
        """Forward pass."""
        user_embed = self.user_embedding(user_ids)
        post_embed = self.post_embedding(post_ids)
        interaction_embed = self.interaction_embedding(interaction_types)
        
        # Concatenate embeddings
        combined = torch.cat([user_embed, post_embed, interaction_embed], dim=1)
        
        # Pass through MLP
        output = self.mlp(combined)
        
        return output.squeeze()

def create_interaction_features(interactions_df):
    """Create features from user interactions."""
    logger.info("🔧 Creating interaction features...")
    
    # Aggregate interactions per user-post pair
    features = interactions_df.groupby(['UserID', 'PostID']).agg({
        'InteractionValue': ['sum', 'mean', 'count'],
        'InteractionType': lambda x: len(x.unique()),
        'Timestamp': ['min', 'max']
    }).reset_index()
    
    features.columns = ['UserID', 'PostID', 'TotalValue', 'AvgValue', 'InteractionCount', 
                       'UniqueTypes', 'FirstInteraction', 'LastInteraction']
    
    # Calculate engagement duration
    features['EngagementDuration'] = (
        features['LastInteraction'] - features['FirstInteraction']
    ).dt.total_seconds()
    
    # Create engagement score
    features['EngagementScore'] = (
        features['TotalValue'] * 0.4 +
        features['InteractionCount'] * 0.3 + 
        features['UniqueTypes'] * 0.2 +
        np.log1p(features['EngagementDuration']) * 0.1
    )
    
    logger.info(f"✅ Created features for {len(features):,} user-post pairs")
    
    return features
