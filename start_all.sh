#!/bin/bash
# Enhanced start script for Buy From Egypt AI models

echo "🚀 Starting Buy From Egypt AI Model Backend & Tester"

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "❌ uvicorn is not installed. Please install it with: pip install uvicorn"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ streamlit is not installed. Please install it with: pip install streamlit"
    exit 1
fi

# Start API server in the background
echo "🔄 Starting API server..."
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 > api_logs.txt 2>&1 &
API_PID=$!

# Wait for API to start
echo "⏳ Waiting for API server to start (5 seconds)..."
sleep 5

# Pre-warm the model
echo "🔥 Pre-warming the recommendation model..."
python warm_up_api.py

# Start Streamlit app
echo "🌐 Starting Streamlit app..."
streamlit run test_recommendations.py

# When Streamlit exits, also stop the API server
echo "🛑 Stopping API server..."
kill $API_PID

echo "✅ All services stopped"
