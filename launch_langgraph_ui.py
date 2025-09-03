#!/usr/bin/env python3
"""
Launcher for LangGraph Agent Streamlit UI
Demonstrates the complete AI workflow with visual feedback
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the LangGraph Agent Streamlit UI."""
    
    print("🚀 LangGraph Agent Workflow UI Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    ui_file = current_dir / "src" / "frontend" / "langgraph_ui.py"
    
    if not ui_file.exists():
        print(f"❌ UI file not found at: {ui_file}")
        print("Please run this script from the project root directory.")
        return
    
    print(f"✅ Found UI file at: {ui_file}")
    
    # Check if Streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} is installed")
    except ImportError:
        print("❌ Streamlit is not installed")
        print("Installing Streamlit...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Check if required packages are installed
    required_packages = ["plotly", "pandas"]
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package])
    
    print("\n🚀 Starting LangGraph Agent Workflow UI...")
    print("📱 The UI will open in your default web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("\n" + "=" * 50)
    print("🎯 Features:")
    print("• Complete workflow visualization")
    print("• Real-time step-by-step execution")
    print("• Embedding generation with Voyage AI")
    print("• MongoDB Atlas vector search")
    print("• Local LLM reasoning")
    print("• Performance metrics and analytics")
    print("• Workflow history and comparison")
    print("=" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(ui_file),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 UI stopped by user")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")

if __name__ == "__main__":
    main()

