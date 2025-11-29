#!/bin/bash

# GitHub Repository Setup Script
# Please run this script after creating a new repository on GitHub

echo "=== Random Feature Diffusion - GitHub Setup ==="
echo ""
echo "Before running this script:"
echo "1. Create a new repository on GitHub"
echo "2. Copy the repository URL (HTTPS or SSH)"
echo ""
echo "Example repository URL: https://github.com/YOUR_USERNAME/random-feature-diffusion.git"
echo ""

read -p "Enter your GitHub repository URL: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "Error: Repository URL is required"
    exit 1
fi

echo ""
echo "Setting up remote repository..."
git remote add origin "$REPO_URL"

echo "Setting main branch..."
git branch -M main

echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Successfully pushed to GitHub!"
echo ""
echo "Your repository is now available at: $REPO_URL"
echo ""
echo "Repository structure:"
echo "📁 Root directory - Core implementation files"
echo "📁 results/ - All experimental results and visualizations"
echo "📁 archive/ - Historical and deprecated files"
echo "📄 README.md - Comprehensive project documentation"
echo "📄 ORGANIZATION_NOTES.md - Detailed organization notes"
