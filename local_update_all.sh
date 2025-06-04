#!/bin/bash

echo "🔄 Update all remote servers..."
bash update_all_from_conf_3.sh

echo "🔄 Get all results to local workstation..."
bash commit_results.sh