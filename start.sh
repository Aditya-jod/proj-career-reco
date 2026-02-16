#!/bin/bash

# Quick Start Script for Mac/Linux
# Starts both backend and frontend in separate terminals/tmux sessions

echo "================================"
echo "Career Path Recommender Startup"
echo "================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to setup backend
setup_backend() {
    cd "$PROJECT_ROOT/backend"
    
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python -m venv venv
    fi
    
    source venv/bin/activate
    
    if [ ! -f "requirements.txt" ]; then
        echo "Installing dependencies..."
        pip install -r requirements.txt
    fi
    
    echo "🎯 Backend starting on http://localhost:8000"
    echo "📚 API docs: http://localhost:8000/docs"
    python -m uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload
}

# Function to setup frontend
setup_frontend() {
    cd "$PROJECT_ROOT/frontend"
    
    if [ ! -d "node_modules" ]; then
        echo "Installing dependencies..."
        npm install
    fi
    
    echo "🎯 Frontend starting on http://localhost:8080"
    npm run dev
}

# Check if tmux is available
if command -v tmux &> /dev/null; then
    echo "Using tmux for multi-window setup..."
    
    # Create tmux session
    tmux new-session -d -s career-reco -x 120 -y 30
    
    # Backend window
    tmux new-window -t career-reco -n backend
    tmux send-keys -t career-reco:backend "cd $PROJECT_ROOT/backend && source venv/bin/activate 2>/dev/null || python -m venv venv && source venv/bin/activate && pip install -r requirements.txt -q && python -m uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload" Enter
    
    # Frontend window
    tmux new-window -t career-reco -n frontend
    tmux send-keys -t career-reco:frontend "cd $PROJECT_ROOT/frontend && npm install -q 2>/dev/null || true && npm run dev" Enter
    
    # Status window
    tmux new-window -t career-reco -n status
    tmux send-keys -t career-reco:status "echo '✅ Both services starting...' && sleep 2 && echo '📍 Frontend: http://localhost:8080' && echo '📍 Backend: http://localhost:8000' && echo '📚 API Docs: http://localhost:8000/docs' && sleep 60" Enter
    
    # Attach to tmux
    tmux select-window -t career-reco:status
    tmux attach-session -t career-reco
else
    echo "tmux not found. Starting services in background..."
    
    # Start backend in background
    echo "🚀 Starting Backend..."
    setup_backend &
    BACKEND_PID=$!
    
    sleep 2
    
    # Start frontend in background
    echo "🚀 Starting Frontend..."
    setup_frontend &
    FRONTEND_PID=$!
    
    echo ""
    echo "✅ Services started"
    echo ""
    echo "📍 Frontend: http://localhost:8080"
    echo "📍 Backend: http://localhost:8000"
    echo "📚 API Docs: http://localhost:8000/docs"
    echo ""
    echo "PIDs: Backend=$BACKEND_PID, Frontend=$FRONTEND_PID"
    echo "To stop: kill $BACKEND_PID $FRONTEND_PID"
    echo ""
    
    # Wait for services
    wait
fi
