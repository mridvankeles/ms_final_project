# Building Annotation Error Detection Pipeline

An automated pipeline for detecting and clustering annotation errors in building segmentation datasets using multi-modal feature analysis.

## Features
- Usage of HiSup model for traning and inference parts for extracting features.
- Multi-modal feature extraction (metrics, losses, visual features)
- Autoencoder-based feature compression
- Clustering of annotation errors
- Visual analysis tools

## Resources
https://drive.google.com/file/d/1ySa9cfm-fw2xv5qEZpGITXlkS0nHbquy/view?usp=sharing
- data
- pretrained model
- finetuned models outputs json
- figures
needs to put them at the needed paths.

## Requirements

### System
- Ubuntu 18.04/20.04 LTS
- NVIDIA GPU (recommended)

### Python Environment (Python 3.9)
```bash
conda create -n hisup-error python=3.9
conda activate hisup-error
pip install -r requirements.txt


