#!/bin/bash

# IndexTTS Deployment Script
# This script helps with building, testing, and deploying the IndexTTS service

set -e  # Exit on any error

echo "🚀 IndexTTS Deployment Script"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Build the Docker image
print_status "Building Docker image..."
if docker build -t indextts-server .; then
    print_success "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Test the Docker image locally (optional)
read -p "Do you want to test the image locally? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Testing Docker image locally..."
    print_warning "This requires GPU support for full testing"

    # Run container in background
    container_id=$(docker run -d --gpus all -p 8000:8000 indextts-server)

    print_status "Container started with ID: $container_id"
    print_status "API should be available at http://localhost:8000"
    print_status "Health check: http://localhost:8000/health"

    # Wait for API to be ready
    print_status "Waiting for API to be ready..."
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &>/dev/null; then
            print_success "API is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_warning "API may not be fully ready yet. Check logs with: docker logs $container_id"
        fi
        sleep 2
    done

    print_status "To stop the test container: docker stop $container_id"
    print_status "To view logs: docker logs $container_id"
fi

print_status "Build and test completed!"
print_status ""
print_status "Next steps for deployment:"
print_status "1. Push image to registry (if needed): docker tag indextts-server your-registry/indextts-server"
print_status "2. Deploy to RunPod or similar GPU serverless platform"
print_status "3. Update Flutter app with your deployment endpoint"
print_status "4. Test voice cloning functionality"

print_success "IndexTTS deployment preparation completed! 🎉"
