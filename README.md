# Deep Learning Automatic Detection and Quantification of Adult Neural Progenitors

This project implements YOLOv8-based detection and quantification of adult neural progenitors using double thymidine analog labeling and confocal microscopy images.

## Project Structure

```
detection-and-quantification-of-adult-neural-progenitors/
├── dataset.yaml                    # YOLOv8 dataset configuration
├── train/                          # Training data (not included - see note below)
│   ├── images/
│   └── labels/
├── validation/                     # Validation data (not included - see note below)
│   ├── images/
│   └── labels/
├── image-preprocessing/             # Data augmentation and preprocessing
│   ├── image-preprocessing.py
│   └── run-image-preprocessing.sh
├── yolov8/                         # Main YOLOv8 training
│   ├── v8.py
│   └── run_yolov8.sh
├── k-fold/                         # K-fold cross validation
│   ├── yolov8_kfold.py
│   ├── config_kfold.py
│   ├── analyze_results.py
│   └── run_kfold.sh
├── optuna/                         # Hyperparameter optimization
│   ├── optuna_v8.py
│   └── run_optuna.sh
└── runs/                           # Training outputs and results
```

## Requirements

### Prerequisites

- Python 3.12+
- PyTorch 2.0+
- CUDA 12.4+ (for GPU training)
- SLURM (for cluster execution)

### Python Dependencies

```bash
pip install ultralytics pandas scikit-learn tqdm albumentations opencv-python numpy matplotlib seaborn optuna pyyaml
```

### Model Files

**IMPORTANT**: YOLOv8 pretrained model files (.pt) must be placed in the project root directory:

```bash
# Download required model files to project root:
# - yolov8n.pt (nano)
# - yolov8s.pt (small)
# - yolov8m.pt (medium)
# - yolov8l.pt (large)
# - yolov8x.pt (extra large)
```

The scripts will automatically use available models or download them if not found.

## Dataset Classes

The project detects 3 classes of neural progenitors:

- **CLDU**: ClDU-positive cells
- **DM**: Double-marked cells
- **IDU**: IdU-positive cells

## Usage

### 1. Image Preprocessing and Data Augmentation

Performs comprehensive data augmentation specifically designed for microscopy images:

**Geometric Transformations:**

- Horizontal/vertical flips
- 90°, 180°, 270° rotations
- Scale and shift variations

**Photometric Enhancements:**

- Brightness and contrast adjustments
- Gamma correction
- **GaussNoise**: Simulates microscope acquisition noise (5.0-15.0 variance, 30% probability)
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization for enhanced local contrast (30% probability)

**Additional Features:**

- Patch generation for large images
- HueSaturationValue and RGB shifts
- Gaussian blur simulation

```bash
cd image-preprocessing/
sbatch run-image-preprocessing.sh
```

Or run directly:

```bash
python image-preprocessing/image-preprocessing.py
```

### 2. Main YOLOv8 Training

Trains YOLOv8x model with 150 epochs:

```bash
cd yolov8/
sbatch run_yolov8.sh
```

Or run directly:

```bash
python yolov8/v8.py
```

### 3. K-Fold Cross Validation

Performs 5-fold cross validation for robust model evaluation:

```bash
cd k-fold/
sbatch run_kfold.sh
```

Or run directly:

```bash
python k-fold/yolov8_kfold.py
```

#### Analyze K-Fold Results

```bash
python k-fold/analyze_results.py
```

### 4. Hyperparameter Optimization

Uses Optuna for automated hyperparameter tuning:

```bash
cd optuna/
sbatch run_optuna.sh
```

Or run directly:

```bash
python optuna/optuna_v8.py
```

## Configuration

### Dataset Configuration (`dataset.yaml`)

```yaml
train: ./train
val: ./validation
test: ./test
nc: 3
names: ["CLDU", "DM", "IDU"]
```

### K-Fold Configuration (`k-fold/config_kfold.py`)

Modify training parameters:

- K_FOLDS: Number of folds (default: 5)
- EPOCHS: Training epochs (default: 300)
- BATCH_SIZE: Batch size (default: 4)
- MODEL_SIZE: YOLOv8 variant ('n', 's', 'm', 'l', 'x')

## Output Structure

Results are organized in the `runs/` directory:

- `runs/train/`: Main training results
- `runs/yolov8_kfold_cv/`: K-fold validation results
- `runs/optuna/`: Hyperparameter optimization results

Each run includes:

- Trained model weights (`best.pt`, `last.pt`)
- Training metrics and plots
- Confusion matrices
- Validation predictions

## Important Notes

### Data Availability

**Images and labels are not included in this repository due to data rights and privacy restrictions.** The folder structure shows where data should be placed:

- Training images: `train/images/`
- Training labels: `train/labels/`
- Validation images: `validation/images/`
- Validation labels: `validation/labels/`

### Cluster Execution

This project was designed for SLURM cluster execution due to:

- **GPU time limitations**: Each component runs separately to manage limited GPU allocation
- **Resource optimization**: Modular structure allows independent execution based on available resources
- **Checkpoint management**: Each stage can be run independently without losing previous work

### Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **Memory**: 16GB+ RAM for large image processing
- **Storage**: 50GB+ for datasets and results

## Execution Workflow

The recommended methodology follows a systematic approach:

1. **Preprocessing**: Run image augmentation to expand training data
2. **Hyperparameter Optimization**: Use Optuna to find optimal training parameters
3. **K-Fold Cross Validation**: Validate model robustness and identify best data partition
4. **Final Training**: Train final model using best hyperparameters on best-performing fold
5. **Analysis**: Evaluate final results

### Methodology Rationale

This approach is optimal because:

- **Step 2**: Finds optimal hyperparameters before expensive k-fold validation
- **Step 3**: Validates hyperparameters across folds AND identifies best data split
- **Step 4**: Combines optimized parameters with best data partition for final model
- **Efficiency**: Avoids redundant hyperparameter search across multiple folds

Each step can be executed independently based on cluster availability and requirements.
