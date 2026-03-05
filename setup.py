#!/usr/bin/env python3
"""
Installation and Setup Script
Run this to automatically set up the entire system
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*60}")
    print(f"► {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("Multi-Asset ML Prediction System - Setup")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("✗ Python 3.10+ required")
        return False
    
    print("✓ Python 3.10+ detected")
    
    # Create virtual environment
    if not os.path.exists("venv"):
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
    else:
        print("\n✓ Virtual environment already exists")
    
    # Determine pip command based on OS
    pip_cmd = "venv\\Scripts\\pip" if os.name == 'nt' else "venv/bin/pip"
    python_cmd = "venv\\Scripts\\python" if os.name == 'nt' else "venv/bin/python"
    
    # Install requirements
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Create necessary directories
    print("\n✓ Creating project directories")
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs("data_cache", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Copy .env file if not exists
    if not os.path.exists(".env"):
        print("\n✓ Creating .env from .env.example")
        subprocess.run("cp .env.example .env", shell=True)
    else:
        print("\n✓ .env file already exists")
    
    # Print completion message
    print("\n" + "="*60)
    print("✓ Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("\n2. Start the application:")
    print("   python app.py")
    print("\n3. Open browser to:")
    print("   http://localhost:5000")
    print("\n" + "="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
