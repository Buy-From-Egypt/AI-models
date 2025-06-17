#!/bin/bash
# Start script for API server

echo "🚀 Starting Buy From Egypt Recommendation API"

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "❌ uvicorn is not installed. Please install it with: pip install uvicorn"
    exit 1
fi

# Start API server
echo "🔄 Starting API server..."
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

echo "✅ API server stopped"
