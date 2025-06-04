#!/bin/bash

# Step 1: Pull latest changes
echo "🔄 Pulling latest changes from remote..."
git pull

# Step 2: Add all changes
echo "➕ Adding all changes..."
git add .

# Step 3: Check if there are any staged changes
if git diff --cached --quiet; then
    echo "⚠️ No changes to commit. Exiting."
    exit 0
fi

# Step 4: Create commit message with timestamp and hostname
timestamp=$(date "+%Y-%m-%d %H:%M:%S")
hostname=$(hostname)
commit_message="partial results [$hostname] $timestamp"

echo "📝 Committing with message: \"$commit_message\""
git commit -m "$commit_message"

# Step 5: Push to remote
echo "🚀 Pushing to remote..."
git push

echo "✅ All done!"
