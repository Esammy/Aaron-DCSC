# DCSC Demo Setup Guide 🚀

This guide will help you quickly set up and run a demo of the Domain- and Category-Style Clustering (DCSC) fake news detection system.

## Quick Start (2 minutes setup!)

### Option 1: One-Command Setup
```bash
# Run the complete demo setup and training
python run_demo.py
```

### Option 2: Step-by-Step Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Demo Data**
   ```bash
   python prepare_demo_data.py
   ```

3. **Run Demo Training**
   ```bash
   python demo_main.py
   ```

## What the Demo Does

The demo will:
- ✅ Check all required dependencies
- 📊 Generate sample fake news detection data
- 🏋️ Train the DCSC model for 5 epochs (quick demo)
- 📈 Show training progress and metrics
- 💾 Save model checkpoints
- 🧪 Run evaluation on test data

## Expected Output

```
🎬 Starting DCSC Demo...
==================================================
✅ All dependencies are installed
📊 Demo Configuration:
   Dataset: 4charliehebdo
   Device: cuda
   Epochs: 5
   Batch Size: 32
   Save Directory: ./demo_results
🏋️ Starting training...
Epoch [1/5] - Loss: 1.234, Accuracy: 65.4%
Epoch [2/5] - Loss: 0.987, Accuracy: 72.1%
...
✅ Demo completed successfully!
```

## Directory Structure After Demo

```
DCSC-released/
├── demo_data/
│   └── data_withdomain/
│       ├── 4charliehebdo/
│       ├── 4ferguson/
│       └── 4germanwings-crash/
├── demo_results/
│   └── dcsc_demo_*/
├── demo_main.py          # Modified main script
├── prepare_demo_data.py  # Data preparation
├── run_demo.py          # One-command setup
└── requirements.txt     # Dependencies
```

## Using Real Data

To use real datasets instead of demo data:

1. **Download Datasets**
   - [PHEME Dataset](https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619)
   - [Weibo Dataset](https://github.com/ICTMCG/Characterizing-Weibo-Multi-Domain-False-News)

2. **Convert to Required Format**
   ```python
   # Each data file should be a pickle with format:
   # (X, y, domain_label) where:
   # X: numpy array of shape (N, 768) - text embeddings
   # y: int (0 for real news, 1 for fake news)
   # domain_label: string (e.g., 'charliehebdo', 'ferguson')
   ```

3. **Update Data Paths**
   ```python
   # In demo_main.py, modify:
   parser.add_argument('--data_dir', type=str, 
                       default='path/to/your/real/data')
   ```

## Customization

### Model Parameters
Edit `demo_main.py` to modify:
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--num_style`: Number of style clusters
- `--domain_num`: Number of domains

### Architecture Changes
Edit `DCSC.py` and `DCSC_layers.py` for:
- Model architecture modifications
- New loss functions
- Enhanced attention mechanisms

## Troubleshooting

### Common Issues

**1. torch_scatter Installation Error**
```bash
# For CUDA 11.3
pip install torch_scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

# For CPU only
pip install torch_scatter -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
```

**2. CUDA Out of Memory**
- Reduce `--batch_size` to 16 or 8
- Reduce `--num_posts` to 50

**3. Data Format Errors**
- Ensure pickle files contain (X, y, domain_label) tuples
- X should be numpy array of shape (N, 768)
- y should be int (0 or 1)

**4. Path Issues on Windows**
- Use forward slashes `/` instead of backslashes `\`
- Ensure no spaces in directory names

### Performance Tips

**For Better Accuracy:**
1. Use real pre-trained BERT embeddings instead of TF-IDF
2. Increase training epochs to 50-200
3. Use larger batch sizes (64-128) if GPU memory allows
4. Add more training data

**For Faster Training:**
1. Use GPU if available
2. Increase `--num_worker` for data loading
3. Use mixed precision training (add to model)

## Next Steps

After running the demo successfully:

1. **Experiment with Improvements** (see main README.md)
2. **Add Real Data** for better performance
3. **Modify Architecture** for your specific use case
4. **Compare with Baselines** in the `baselines/` directory

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify data format matches expected structure
4. Check GPU memory availability for large models

Happy experimenting! 🎉