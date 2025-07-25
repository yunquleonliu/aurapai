#!/bin/bash

# Auro-PAI Platform Backend Setup and Startup Script
# ==================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is not installed. Please install Python 3.8+ to continue."
        exit 1
    fi
}

# Setup virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    fi

    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install requirements
install_requirements() {
    print_status "Installing Python requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Requirements installed"
}

# Setup environment file
setup_env() {
    if [ ! -f ".env" ]; then
        print_status "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please edit .env file with your configuration before running"
        print_warning "Key settings to configure:"
        print_warning "  - LLAMACPP_SERVER_URL (your llama.cpp server)"
        print_warning "  - CHROMADB_HOST/PORT (ChromaDB configuration)"
        print_warning "  - Optional: OPENAI_API_KEY, GEMINI_API_KEY"
    else
        print_success ".env file already exists"
    fi
}

# Check if llama.cpp server is running
check_llamacpp() {
    LLAMACPP_URL=$(grep "^=" .env | cut -d'=' -f2 | tr -d '"')
    if [ -z "$LLAMACPP_URL" ]; then
        LLAMACPP_URL="http://10.0.0.206:8000"
    fi
    
    print_status "Checking llama.cpp server at $LLAMACPP_URL..."
    if curl -s "$LLAMACPP_URL/health" > /dev/null 2>&1; then
        print_success "llama.cpp server is running"
    else
        print_warning "llama.cpp server not accessible at $LLAMACPP_URL"
        print_warning "Please ensure your llama.cpp server is running"
        print_warning "Example command: ./server -m model.gguf --host 0.0.0.0 --port 8000"
    fi
}

# Check if ChromaDB is accessible
check_chromadb() {
    CHROMADB_HOST=$(grep CHROMADB_HOST .env | cut -d'=' -f2 | tr -d '"')
    CHROMADB_PORT=$(grep CHROMADB_PORT .env | cut -d'=' -f2 | tr -d '"')
    
    if [ -z "$CHROMADB_HOST" ]; then
        CHROMADB_HOST="localhost"
    fi
    if [ -z "$CHROMADB_PORT" ]; then
        CHROMADB_PORT="8002"
    fi
    
    print_status "Checking ChromaDB at $CHROMADB_HOST:$CHROMADB_PORT..."
    if curl -s "http://$CHROMADB_HOST:$CHROMADB_PORT/api/v1/heartbeat" > /dev/null 2>&1; then
        print_success "ChromaDB server is running"
    else
        print_warning "ChromaDB server not accessible."
        if command -v chroma &> /dev/null; then
            print_status "Attempting to start ChromaDB server in the background..."
            chroma run --host "$CHROMADB_HOST" --port "$CHROMADB_PORT" &
            sleep 5 # Give the server a moment to start
            if curl -s "http://$CHROMADB_HOST:$CHROMADB_PORT/api/v1/heartbeat" > /dev/null 2>&1; then
                print_success "ChromaDB server started successfully."
            else
                print_error "Failed to start ChromaDB server. Please start it manually."
            fi
        else
            print_error "ChromaDB is not installed or not in PATH. Please install it with 'pip install chromadb'"
        fi
    fi
}

# Run tests
run_tests() {
    if [ "$1" = "--test" ]; then
        print_status "Running tests..."
        python -m pytest tests/ -v
        return 0
    fi
    return 1
}

# Start the server
start_server() {
    # Get port from .env file, default to 8001
    SERVER_PORT=$(grep "^PORT=" .env | cut -d'=' -f2 | tr -d '"')
    if [ -z "$SERVER_PORT" ]; then
        SERVER_PORT="8001"
    fi
    
    print_status "Starting Auro-PAI Platform Backend..."
    print_status "Server will be available at: http://localhost:${SERVER_PORT}"
    print_status "API documentation: http://localhost:${SERVER_PORT}/docs"
    print_status "Health check: http://localhost:${SERVER_PORT}/api/v1/health"
    print_status ""
    print_status "Press Ctrl+C to stop the server"
    
    # Set the agent mode directly to ensure it runs as ReAct
    export AGENT_MODE="ReAct"

    # Start with uvicorn for development, or python main.py
    if command -v uvicorn &> /dev/null; then
        uvicorn main:app --host 0.0.0.0 --port ${SERVER_PORT} --reload
    else
        python main.py
    fi
}

# Main execution
main() {
    print_status "Auro-PAI Platform Backend Setup and Startup"
    print_status "============================================"
    
    # Check if tests should be run
    if run_tests "$1"; then
        exit 0
    fi
    
    check_python
    setup_venv
    install_requirements
    setup_env
    
    # Check dependencies if .env exists and is configured
    if [ -f ".env" ]; then
        check_llamacpp
        check_chromadb
    fi
    
    print_success "Setup completed!"
    print_status ""
    
    if [ "$1" = "--setup-only" ]; then
        print_success "Setup completed. Run ./start.sh to start the server."
        exit 0
    fi
    
    start_server
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Auro-PAI Platform Backend Startup Script"
        echo ""
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  --help, -h      Show this help message"
        echo "  --setup-only    Only run setup, don't start server"
        echo "  --test          Run tests and exit"
        echo ""
        echo "No arguments:     Run full setup and start server"
        exit 0
        ;;
    *)
        main "$1"
        ;;
esac
