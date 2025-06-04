#!/bin/bash
CONFIG_FILE=host_repo.conf
USER="yen-chi-chen"
LINENO=0

while IFS= read -r LINE || [ -n "$LINE" ]; do
    LINENO=$((LINENO+1))
    [[ -z "$LINE" || "$LINE" =~ ^# ]] && continue

    HOST=$(echo "$LINE" | awk '{print $1}')
    # REPO_PATH=$(echo "$LINE" | cut -d' ' -f2-)
    REPO_PATH=$(echo "$LINE" | cut -d' ' -f2- | sed 's/^ *//;s/ *$//')

    echo "=== 🐺 [$HOST] Updating repo at $REPO_PATH ==="
    
    ssh "$USER@$HOST" "bash -s" <<EOF
if [ -d "$REPO_PATH" ]; then
    echo "✅ Found directory: $REPO_PATH"
    cd "$REPO_PATH"
    if [ -f ./commit_results.sh ]; then
        bash ./commit_results.sh
    else
        echo "⚠️ Missing commit_results.sh"
    fi
else
    echo "❌ Directory not found: $REPO_PATH"
fi
EOF

    echo "-----"
done < "$CONFIG_FILE"