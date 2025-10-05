#!/bin/bash

# Set your GitHub username
GITHUB_USERNAME="Aswath1709"

# Base directory containing all projects
BASE_DIR="$HOME/Downloads/MyProjects"

# List of project folders (based on your screenshot)
PROJECTS=(
"Distributed_ML_collision_severity_classification"
)

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create a temporary directory for processing
TEMP_DIR="$HOME/Desktop/repo_migration_temp"
mkdir -p "$TEMP_DIR"

echo "Starting repository migration..."

for PROJECT in "${PROJECTS[@]}"; do
    echo -e "\n${GREEN}Processing: $PROJECT${NC}"
    
    # Check if project folder exists
    if [ ! -d "$BASE_DIR/$PROJECT" ]; then
        echo -e "${RED}Warning: $BASE_DIR/$PROJECT not found, skipping...${NC}"
        continue
    fi
    
    # Convert to lowercase and replace underscores with hyphens for repo name
    REPO_NAME=$(echo "$PROJECT" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    
    # Copy project to temp directory
    echo "Copying project files..."
    cp -r "$BASE_DIR/$PROJECT" "$TEMP_DIR/$PROJECT"
    cd "$TEMP_DIR/$PROJECT"
    
    # Initialize git repository
    echo "Initializing Git repository..."
    git init
    
    # Create a README if it doesn't exist
    if [ ! -f "README.md" ]; then
        echo "# $PROJECT" > README.md
        echo "" >> README.md
        echo "Project migrated from MyProjects repository." >> README.md
    fi
    
    # Add all files
    git add .
    git commit -m "Initial commit - migrated from MyProjects"
    
    # Create GitHub repository
    echo "Creating GitHub repository: $REPO_NAME"
    gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully created and pushed $REPO_NAME${NC}"
        echo "  URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    else
        echo -e "${RED}✗ Failed to create $REPO_NAME${NC}"
    fi
    
    # Clean up temp folder for this project
    cd "$TEMP_DIR"
    rm -rf "$PROJECT"
done

# Clean up temp directory
cd "$HOME"
rm -rf "$TEMP_DIR"

echo -e "\n${GREEN}Migration complete!${NC}"
echo "Check your GitHub profile: https://github.com/$GITHUB_USERNAME"
