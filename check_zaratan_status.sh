#!/bin/bash
echo "Attempting to connect and check Zaratan status..."
timeout 5 ssh zaratan "echo '=== CONNECTED TO ZARATAN ===' && hostname && echo '=== PROJECT STATUS ===' && ls -la ~/text2video/ 2>&1 | head -20 && echo '=== MODULE STATUS ===' && module list 2>&1 && echo '=== PYTHON STATUS ===' && python --version 2>&1 && pip list 2>&1 | grep -E '(torch|transformers|diffusers)' | head -10" 2>&1 || echo "Not connected (SSH requires interactive authentication)"
