"""
Setup Script for LangGraph MCP Integration Demo
===============================================

This script handles the complete setup and installation process.
"""

import subprocess
import sys
import os
import asyncio
from pathlib import Path

def run_command(command, description, shell=False):
    """Run a command and handle errors gracefully."""
    print(f"\n🔧 {description}...")
    try:
        if shell:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"   Error: {e.stderr.strip() if e.stderr else str(e)}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def setup_virtual_environment():
    """Create and activate virtual environment."""
    print("\n📦 Setting up virtual environment...")
    # .\lc-academy-env\Scripts\Activate.ps1
    # Create virtual environment
    if not run_command([sys.executable, "-m", "venv", "lc-academy-en"], 
                      "Creating virtual environment"):
        return False
    
    # Determine activation script based on OS
    if sys.platform == "win32":
        activate_script = ".\lc-academy-env\\Scripts\\activate.bat"
        pip_path = ".\lc-academy-env\\Scripts\\pip.exe"
    else:
        activate_script = "source ./lc-academy-env/bin/activate"
        pip_path = "./lc-academy-env/bin/pip"
    
    print(f"📋 To activate virtual environment manually, run: {activate_script}")
    return pip_path

def install_requirements(pip_path):
    """Install all required packages."""
    print("\n📥 Installing requirements...")
    
    # Upgrade pip first
    if not run_command([pip_path, "install", "--upgrade", "pip"], 
                      "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command([pip_path, "install", "-r", "requirements.txt"], 
                      "Installing requirements from requirements.txt"):
        return False
    
    return True

def verify_installation(pip_path):
    """Verify that all packages are installed correctly."""
    print("\n🔍 Verifying installation...")
    
    key_packages = [
        "langgraph",
        "langchain-mcp-adapters", 
        "mcp",
        "langchain-anthropic"
    ]
    
    for package in key_packages:
        if not run_command([pip_path, "show", package], 
                          f"Checking {package}"):
            return False
    
    return True

def create_env_file():
    """Create environment file for API keys."""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    print("\n🔑 Creating .env file for API keys...")
    
    env_content = """# Environment Variables for LangGraph MCP Demo
# ==============================================

# Choose your preferred LLM provider (uncomment one):

# Option 1: OpenAI GPT models (GPT-4.1, GPT-4o, GPT-4o mini)
OPENAI_API_KEY=your_openai_api_key_here

# Option 2: Anthropic Claude (Default)
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Option 3: Google Gemini 2.5 Pro (Fully MCP Compatible)
# GOOGLE_API_KEY=your_google_api_key_here

# Optional: For enterprise users
# OPENAI_ORG_ID=your_openai_org_id_here

# Optional: LangChain tracing
# LANGCHAIN_API_KEY=your_langchain_api_key_here

# Logging level
LOG_LEVEL=INFO

# Development settings
DEBUG=True
"""
    
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    print("⚠️  Please edit .env and add your actual API keys!")

def create_startup_scripts():
    """Create convenience startup scripts."""
    print("\n📜 Creating startup scripts...")
    
    # Windows batch file
    windows_script = """@echo off
echo Starting LangGraph MCP Demo...
echo.

REM Start weather server in background
echo Starting Weather MCP Server (HTTP)...
start /B python weather_mcp_server.py

REM Wait a moment for server to start
timeout /t 3 /nobreak > nul

echo Starting main demo...
python main.py

pause
"""
    
    with open("start_demo.bat", "w") as f:
        f.write(windows_script)
    
    # Unix shell script
    unix_script = """#!/bin/bash
echo "Starting LangGraph MCP Demo..."
echo

# Start weather server in background
echo "Starting Weather MCP Server (HTTP)..."
python weather_mcp_server.py &
WEATHER_PID=$!

# Wait for server to start
sleep 3

echo "Starting main demo..."
python main.py

# Cleanup
echo "Cleaning up background processes..."
kill $WEATHER_PID 2>/dev/null || true
"""
    
    with open("start_demo.sh", "w") as f:
        f.write(unix_script)
    
    # Make shell script executable
    if sys.platform != "win32":
        run_command(["chmod", "+x", "start_demo.sh"], 
                   "Making shell script executable")
    
    print("✅ Created startup scripts: start_demo.bat and start_demo.sh")

def main():
    """Main setup function."""
    print("="*60)
    print("🚀 LANGGRAPH MCP INTEGRATION DEMO - SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup virtual environment
    pip_path = setup_virtual_environment()
    if not pip_path:
        sys.exit(1)
    
    # Install requirements
    # if not install_requirements(pip_path):
    #     sys.exit(1)
    
    # Verify installation
    if not verify_installation(pip_path):
        sys.exit(1)
    
    # Create environment file
    # create_env_file()
    
    # Create startup scripts
    create_startup_scripts()
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📋 NEXT STEPS:")
    print("1. Edit .env file and add your Anthropic API key")
    print("2. Run the demo using:")
    if sys.platform == "win32":
        print("   • Windows: start_demo.bat")
        print("   • Or manually: python main.py")
    else:
        print("   • Unix/Mac: ./start_demo.sh")
        print("   • Or manually: python main.py")
    
    print("\n⚠️  IMPORTANT:")
    print("   • Make sure to set your ANTHROPIC_API_KEY in .env")
    print("   • The weather server will start on port 8001")
    print("   • Check README.md for detailed instructions")

if __name__ == "__main__":
    main()