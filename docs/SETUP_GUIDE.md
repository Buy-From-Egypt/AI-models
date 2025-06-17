# 🚀 Setup Guide

This guide will  set up and run the Buy from Egypt Recommendation System.

## Prerequisites

- Python 3.8 or later
- pip package manager
- Git

## Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/your-organization/Buy-From-Egypt-AI-models.git
cd Buy-From-Egypt-AI-models
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Verify the data**

Ensure that the processed data files are available in the `data/processed/` directory:

- `user_profiles.csv`
- `user_preferences.csv`
- `business_features.csv`
- `company_posts.csv`
- `user_post_matrix.csv`
- `category_mapping.csv`

4. **Verify the model**

Check that the trained model exists at `models/hybrid_recommendation_model.pth` and model info at `models/model_info.json`.

## Running the Application

### Option 1: Run everything with a single command

```bash
./start_all.sh
```

This script will:
- Start the API server
- Warm up the recommendation model (first request is slower)
- Launch the Streamlit testing interface

### Option 2: Run components separately

**In terminal 1: Run the API server**

```bash
./start_api.sh
```

**In terminal 2: Run the testing interface**

```bash
./start_tester.sh
```

## Troubleshooting

### API timeout issues

If you're experiencing API timeout issues, try these steps:

1. Check that the API server is running (Terminal 1)
2. Run the warm-up script directly to pre-load the model:
   ```bash
   python warm_up_api.py
   ```
3. Try requesting fewer recommendations (reduce from 10 to 5)

### Data or model not found errors

Ensure all required files are in the correct locations:

```bash
ls models/hybrid_recommendation_model.pth
ls data/processed/
```

If files are missing, check the project documentation or contact the project administrator.
