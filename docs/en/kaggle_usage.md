# Kaggle Integration Guide

This guide explains how to use the point cloud classification system on Kaggle.

## Prerequisites

1. **Kaggle Account**: You need a Kaggle account
2. **GitHub Repository**: Your project should be on GitHub (https://github.com/ybyyb1/pointcloud-classification)
3. **Kaggle Notebook**: Create a new notebook on Kaggle (https://www.kaggle.com/code)

## Option 1: Automatic Setup (Recommended)

### Step 1: Clone and Setup in Kaggle Notebook

Add the following code to the first cell of your Kaggle notebook:

```python
# Clone GitHub repository
!git clone https://github.com/ybyyb1/pointcloud-classification.git

# Navigate to project directory
import os
os.chdir('/kaggle/working/pointcloud-classification')

# Install dependencies
!pip install -r requirements.txt

# Setup Kaggle environment
!python scripts/setup_kaggle_github.py
```

### Step 2: Download Dataset

```python
# Download ScanObjectNN dataset
!python main.py download-scanobjectnn --data_dir /kaggle/working/data/scanobjectnn
```

### Step 3: Train Model with Kaggle Optimization

```python
# Train with Kaggle optimization
!python scripts/train.py \
  --experiment kaggle_training \
  --epochs 50 \
  --batch_size 16 \
  --kaggle \
  --data_dir /kaggle/working/data/scanobjectnn \
  --model point_transformer \
  --dataset scanobjectnn
```

### Step 4: Evaluate Model

```python
# Evaluate trained model
!python main.py evaluate \
  --checkpoint /kaggle/working/checkpoints/kaggle_training/best_model.pth \
  --dataset scanobjectnn \
  --data_dir /kaggle/working/data/scanobjectnn
```

## Option 2: Manual Setup

### Step 1: Clone Repository

```python
import os
import sys

# Clone repository
!git clone https://github.com/ybyyb1/pointcloud-classification.git

# Add to Python path
project_path = '/kaggle/working/pointcloud-classification'
sys.path.insert(0, project_path)
os.chdir(project_path)
```

### Step 2: Install Dependencies

```python
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install -r requirements.txt
```

### Step 3: Import Project Modules

```python
# Import project modules
from main import download_scanobjectnn, train_model, evaluate_model
from config import SystemConfig
import argparse
```

### Step 4: Prepare Dataset

```python
# Create namespace for arguments
class Args:
    def __init__(self):
        self.data_dir = '/kaggle/working/data/scanobjectnn'
        self.version = 'main_split'
        self.num_points = 1024
        self.batch_size = 16

# Download dataset
args = Args()
download_scanobjectnn(args)
```

### Step 5: Train Model

```python
# Create training configuration
config = SystemConfig()

# Update config for Kaggle
config.training.epochs = 50
config.training.batch_size = 16
config.training.use_amp = True
config.training.checkpoint_dir = '/kaggle/working/checkpoints'
config.dataset.data_dir = '/kaggle/working/data/scanobjectnn'

# Run training
from scripts.train import train_model as train_func
results = train_func(
    experiment_name='kaggle_manual',
    use_kaggle=True,
    num_epochs=50
)

print(f"Best accuracy: {results['best_accuracy']:.4f}")
```

## Kaggle-Specific Optimizations

### GPU Memory Management

```python
import torch

# Check GPU memory
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    
    # Set memory fraction for Kaggle P100 (16GB)
    if gpu_memory >= 16:
        torch.cuda.set_per_process_memory_fraction(14.0 / gpu_memory)
```

### Performance Optimization

```python
# Set environment variables for performance
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
```

### Kaggle Dataset Integration

If you want to use datasets from Kaggle Datasets:

```python
# Mount Kaggle dataset
import opendatasets as od
od.download("https://www.kaggle.com/datasets/hkustvgd/scanobjectnn")

# Use the dataset
!python main.py train \
  --data_dir /kaggle/input/scanobjectnn \
  --kaggle
```

## Complete Kaggle Notebook Example

Here's a complete example for a Kaggle notebook:

```python
# Cell 1: Setup
!git clone https://github.com/ybyyb1/pointcloud-classification.git
import os
os.chdir('/kaggle/working/pointcloud-classification')
!pip install -r requirements.txt

# Cell 2: Dataset
!python main.py download-scanobjectnn --data_dir /kaggle/working/data/scanobjectnn

# Cell 3: Training
!python scripts/train.py \
  --experiment my_kaggle_experiment \
  --epochs 100 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --kaggle \
  --model point_transformer \
  --dataset scanobjectnn

# Cell 4: Evaluation
!python main.py evaluate \
  --checkpoint /kaggle/working/checkpoints/my_kaggle_experiment/best_model.pth \
  --dataset scanobjectnn

# Cell 5: Visualization
!python main.py visualize \
  --dataset scanobjectnn \
  --num_samples 5 \
  --output_dir /kaggle/working/visualizations
```

## Kaggle Competition Submission

If you're participating in a Kaggle competition:

```python
# Create submission file
import pandas as pd
import numpy as np

# Load predictions from your model
predictions = np.load('/kaggle/working/predictions.npy')

# Create submission DataFrame
submission = pd.DataFrame({
    'id': range(len(predictions)),
    'prediction': predictions
})

# Save submission
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file created")
```

## Tips for Kaggle Success

### 1. **Use Kaggle's GPU Efficiently**
- Batch size: 16-32 for P100
- Enable mixed precision training (`--kaggle` flag does this automatically)
- Use gradient accumulation for larger effective batch sizes

### 2. **Manage Disk Space**
- Kaggle provides ~20GB disk space
- Clean up intermediate files:
  ```python
  !rm -rf /kaggle/working/__pycache__
  !rm -rf /kaggle/working/*.pyc
  ```

### 3. **Save Checkpoints**
- Save model checkpoints to `/kaggle/working/checkpoints/`
- Kaggle Notebook outputs are saved automatically

### 4. **Use Kaggle Secrets for API Keys**
```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()

# Get API key (if needed)
api_key = secrets.get_secret("kaggle_api_key")
```

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce batch size
```python
# Use smaller batch size
!python scripts/train.py --batch_size 8 --kaggle
```

### Issue: Slow Training
**Solution**: Enable cuDNN benchmark
```python
import torch
torch.backends.cudnn.benchmark = True
```

### Issue: Import Errors
**Solution**: Check Python path
```python
import sys
print(sys.path)
sys.path.insert(0, '/kaggle/working/pointcloud-classification')
```

### Issue: Dataset Download Failed
**Solution**: Use Kaggle dataset instead
```python
# Download from Kaggle dataset
!kaggle datasets download -d hkustvgd/scanobjectnn
!unzip scanobjectnn.zip -d /kaggle/working/data/
```

## Resources

- [Kaggle Notebook Documentation](https://www.kaggle.com/docs/notebooks)
- [Kaggle GPU Guide](https://www.kaggle.com/docs/gpu)
- [Point Cloud Classification GitHub](https://github.com/ybyyb1/pointcloud-classification)

## Support

If you encounter issues:
1. Check the [FAQ](faq.md)
2. Submit [GitHub Issues](https://github.com/ybyyb1/pointcloud-classification/issues)
3. Ask on [Kaggle Discussions](https://www.kaggle.com/discussions)