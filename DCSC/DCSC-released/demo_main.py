#!/usr/bin/env python3
"""
Demo script for DCSC fake news detection
Fixed hardcoded paths and simplified for easy testing
"""

import torch
from DCSC import Net
import argparse
import os
import random
import numpy as np
import sys
from string import digits

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def run_demo(dataset='4charliehebdo', seed=2023, use_cuda=True):
    """
    Run DCSC demo with flexible configuration
    """
    parser = argparse.ArgumentParser()
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default=dataset,
                        choices=['weibo', '4charliehebdo', '4ferguson', '4germanwings-crash', 
                                '4ottawashooting', '4sydneysiege'],
                        help='The name of dataset')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (reduced for demo)')
    parser.add_argument('--start_epoch', type=int, default=0, help='Continue to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (reduced for demo)')
    parser.add_argument('--num_worker', type=int, default=2, help='Number of workers (reduced for demo)')
    parser.add_argument('--num_posts', type=int, default=100)
    parser.add_argument('--margin', type=float, default=0.5, help='Margin in recovered loss')
    parser.add_argument('--domain_num', type=int, default=4, help='Number of domains')
    parser.add_argument('--num_style', type=int, default=8, help='Number of styles')
    
    # Model architecture
    parser.add_argument('--TextCNN_pars', type=tuple, default=(1, 200), help='num_filters, h_hid')
    parser.add_argument('--text_embedding', type=tuple, default=(768, 200), help='Reduce dimension of text vector')
    parser.add_argument('--encoder_pars', type=tuple, default=(1, 1, 200, 100, 0.2), 
                        help='num_layers, n_head, f_in, f_hid, dropout')
    
    # Training settings
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Attention dropout in GAT')
    parser.add_argument('--decay_start_epoch', type=float, default=10, help='Decay start epoch')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (reduced for demo)')
    parser.add_argument("--b1", type=float, default=0.5, help="Adam: decay of first order momentum")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam: decay of second order momentum")
    
    # Fixed paths for demo (relative to current directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--save_dir', type=str, default=os.path.join(current_dir, 'demo_results'),
                        help='Directory to save the model')
    parser.add_argument('--model_name', type=str, default=f'dcsc_demo_{dataset}_{seed}',
                        help='Model name for saving')
    parser.add_argument('--data_dir', type=str, 
                        default=os.path.join(current_dir, 'demo_data', 'data_withdomain', dataset),
                        help='Data directory')
    parser.add_argument('--target_domain', type=str,
                        default=os.path.join(current_dir, 'demo_data', 'raw_data_withdomain'),
                        help='Target domain data directory')
    parser.add_argument('--data_eval', default='', type=str)

    args = parser.parse_args([])  # Empty list for demo
    
    # Set device
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using GPU for acceleration")
    else:
        device = torch.device("cpu") 
        print("💻 Using CPU (consider using GPU for faster training)")
    
    print(f"📊 Demo Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Device: {device}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Save Directory: {args.save_dir}")
    print(f"   Data Directory: {args.data_dir}")
    
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.model_name), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.model_name, args.dataset), exist_ok=True)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        print(f"📁 Please create the data directory structure or download the datasets")
        print(f"   Expected structure:")
        print(f"   {args.data_dir}/")
        print(f"   ├── train/")
        print(f"   │   ├── rumor/")
        print(f"   │   └── nonrumor/")
        print(f"   ├── val/")
        print(f"   │   ├── rumor/")
        print(f"   │   └── nonrumor/")
        print(f"   └── test/ (optional)")
        return False
    
    # Initialize model
    try:
        model = Net(args, device)
        print("✅ Model initialized successfully")
        
        # Save configuration
        args_dict = args.__dict__
        with open(os.path.join(args.save_dir, args.model_name, args.dataset, 'demo_config.txt'), 'w') as f:
            f.writelines('=============== DCSC Demo Configuration ===============\n')
            for arg, value in args_dict.items():
                f.writelines(f'{arg}: {value}\n')
            f.writelines('=====================================================\n')
        
        # Domain configuration for demo
        if dataset == 'weibo':
            target_data = 'weibo2021'
        elif dataset == 'weibo2021':
            target_data = 'weibo'
        else:
            target_data = dataset.strip(digits)
        
        # Create domain dictionary (simplified for demo)
        all_datasets = ['4charliehebdo', '4ferguson', '4germanwings-crash', '4ottawashooting', '4sydneysiege']
        domains = set(all_datasets) - set([args.dataset])
        domain_dict = {domain.strip(digits): idx for idx, domain in enumerate(domains)}
        
        print(f"🎯 Domain mapping: {domain_dict}")
        
        # Start training if data exists
        if os.path.exists(os.path.join(args.data_dir, 'train')):
            print("🏋️ Starting training...")
            model.train_epoch(args.data_dir, 0, domain_dict)
            
            # Test if test data exists
            if os.path.exists(args.target_domain):
                domain_dict[target_data.strip(digits)] = len(all_datasets)
                print("🧪 Running evaluation...")
                model.test(args.target_domain, target_data, domain_dict)
                
            print("✅ Demo completed successfully!")
            return True
        else:
            print("⚠️ Training data not found. Model initialized but not trained.")
            return True
            
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = ['torch', 'numpy', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    # Check torch_scatter separately
    try:
        import torch_scatter
    except ImportError:
        missing_packages.append('torch_scatter')
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print(f"💡 Install with: pip install {' '.join(missing_packages)}")
        if 'torch_scatter' in missing_packages:
            print("   For torch_scatter: pip install torch_scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html")
        return False
    else:
        print("✅ All dependencies are installed")
        return True

if __name__ == '__main__':
    print("🎬 Starting DCSC Demo...")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        sys.exit(1)
    
    # Run demo
    success = run_demo(dataset='4charliehebdo', seed=2023, use_cuda=True)
    
    if success:
        print("\n🎉 Demo setup completed!")
        print("📋 Next steps:")
        print("   1. Download dataset from the URLs in README.md")
        print("   2. Prepare data in the expected format (pickle files)")
        print("   3. Run: python demo_main.py")
    else:
        print("\n❌ Demo setup failed. Please check the error messages above.")