#!/bin/bash

# UMD Zaratan Project Transfer Script
# This script helps transfer your text2video project to Zaratan GPU cluster

echo "🚀 UMD Zaratan Project Transfer Script"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from your text2video project directory"
    exit 1
fi

echo "📁 Current project directory: $(pwd)"
echo "🔍 Project files:"
ls -la

echo ""
echo "📋 Next steps to connect to Zaratan:"
echo "1. Ensure you have Zaratan access (contact HPC support if needed)"
echo "2. Test SSH connection: ssh zaratan"
echo "3. Create project directory on Zaratan: mkdir -p ~/text2video"
echo "4. Transfer project files:"
echo "   scp -r . zaratan:~/text2video/"
echo ""
echo "🔧 For GPU job submission, you'll need to:"
echo "1. Load required modules (Python, CUDA, etc.)"
echo "2. Install project dependencies"
echo "3. Create Slurm job scripts"
echo ""
echo "📚 Zaratan Documentation: https://hpcc.umd.edu/hpcc/help/basics.html"
echo "🆘 HPC Help Desk: 301-405-1500"
