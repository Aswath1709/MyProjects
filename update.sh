# See how many files changed
git status | grep -c "modified:"

# Add all changes at once
git add -A

# Commit with a comprehensive message
git commit -m "Major update: restructured entire project"

# Push everything
git push origin main
