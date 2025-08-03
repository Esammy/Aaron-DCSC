#!/usr/bin/env python3
"""
One-command demo setup and execution for DCSC
Combines dependency checking, data preparation, and model training
"""

import subprocess
import sys
import os
import time

def run_command(command, description):
    """Run a command and handle errors gracefully"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 6):
        print(f"❌ Python 3.6+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    
    # Basic packages
    basic_packages = ["torch", "numpy", "scikit-learn"]
    for package in basic_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    
    # torch_scatter (requires special handling)
    print("\n🔧 Installing torch_scatter...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            print(f"   CUDA {cuda_version} detected")
            cmd = "pip install torch_scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html"
        else:
            print("   No CUDA detected, installing CPU version")
            cmd = "pip install torch_scatter -f https://data.pyg.org/whl/torch-1.12.0+cpu.html"
        
        if not run_command(cmd, "Installing torch_scatter"):
            print("⚠️  torch_scatter installation failed, trying alternative...")
            return run_command("pip install torch_scatter", "Installing torch_scatter (fallback)")
    except ImportError:
        print("⚠️  PyTorch not found, installing torch_scatter without CUDA optimization")
        return run_command("pip install torch_scatter", "Installing torch_scatter")
    
    return True

def main():
    """Main demo execution function"""
    print("🎬 DCSC Complete Demo Setup")
    print("=" * 50)
    print("This will:")
    print("  1. ✅ Check Python version")
    print("  2. 📦 Install dependencies") 
    print("  3. 📊 Generate demo data")
    print("  4. 🏋️ Train DCSC model")
    print("  5. 📈 Show results")
    print("=" * 50)
    
    # Get user confirmation
    response = input("\n🚀 Continue with demo setup? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Demo cancelled.")
        return
    
    start_time = time.time()
    
    # Step 1: Check Python version
    if not check_python_version():
        print("\n❌ Please upgrade Python to 3.6+ and try again")
        return
    
    # Step 2: Install dependencies
    print(f"\n📦 Installing dependencies (this may take a few minutes)...")
    if not install_dependencies():
        print(f"\n❌ Dependency installation failed.")
        print(f"💡 Try manual installation:")
        print(f"   pip install torch numpy scikit-learn")
        print(f"   pip install torch_scatter")
        return
    
    # Step 3: Prepare demo data
    print(f"\n📊 Preparing demo data...")
    if not run_command("python prepare_demo_data.py", "Generating demo datasets"):
        print(f"❌ Data preparation failed")
        return
    
    # Step 4: Run training demo
    print(f"\n🏋️ Starting DCSC training demo...")
    print(f"   (This will train for 5 epochs - should take 2-5 minutes)")
    
    if not run_command("python demo_main.py", "Running DCSC training"):
        print(f"❌ Training demo failed")
        print(f"💡 Try running manually: python demo_main.py")
        return
    
    # Success!
    elapsed_time = time.time() - start_time
    print(f"\n🎉 Demo completed successfully!")
    print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
    print(f"\n📊 Results saved in: ./demo_results/")
    print(f"📁 Demo data in: ./demo_data/")
    
    print(f"\n🔬 What's next?")
    print(f"  • Check demo_results/ for trained models")
    print(f"  • Review DEMO_README.md for customization")
    print(f"  • Try with real datasets (see main README.md)")
    print(f"  • Experiment with model improvements")
    
    print(f"\n📝 Demo files created:")
    files_created = [
        "demo_main.py - Modified main script",
        "prepare_demo_data.py - Data preparation",
        "requirements.txt - Dependencies list", 
        "DEMO_README.md - Detailed demo guide",
        "demo_data/ - Sample datasets",
        "demo_results/ - Training outputs"
    ]
    for file in files_created:
        print(f"  ✓ {file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"💡 Please try running components manually:")
        print(f"   python prepare_demo_data.py")
        print(f"   python demo_main.py")