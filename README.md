# FMA Music Genre Classification with MFCC and Transfer Learning Models

This repository contains the code and experimental workflow for a master's thesis on automatic music genre classification using spectral audio analysis. The project focuses on the Free Music Archive (FMA) dataset and compares classical machine learning, deep learning and transfer learning approaches.

The main feature representation is based on Mel-Frequency Cepstral Coefficients (MFCCs). In addition, transfer learning approaches based on Audio Spectrogram Transformers (AST) and Pretrained Audio Neural Networks (PANNs) are evaluated.

## Project Description

The goal of this project is to classify music tracks into top-level genres based on audio signal characteristics.

The experiments investigate how different feature representations and model architectures perform for music genre classification. The project includes:

- preprocessing of FMA audio files and metadata
- conversion of MP3 files to WAV format
- optional audio augmentation
- MFCC extraction from random audio segments
- tabular MFCC mean and standard deviation features
- MFCC matrix inputs for deep learning models
- classical machine learning models
- convolutional and recurrent neural network models
- transfer learning using AST and PANNs
- segment-level and track-level evaluation
- misclassification and dataset statistics

The dataset itself is not included in this repository due to its size and licensing conditions.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing_functions.py
│   └── model_functions.py
│
├── notebooks/
│   ├── 01_preprocessing/
│   │   ├── 01_MP3_to_WAV.ipynb
│   │   ├── 02_Prepare_Metadata.ipynb
│   │   ├── 03_create_audio_augmentation.ipynb
│   │   ├── 04_extract_mfcc_segments.ipynb
│   │   └── 05_extract_mfcc_mean_features.ipynb
│   │
│   ├── 02_models/
│   │   ├── 01_RandomForest.ipynb
│   │   ├── 02_LightGBM.ipynb
│   │   ├── 03_SVM.ipynb
│   │   ├── 04_ResNet.ipynb
│   │   ├── 04_ResNet_aug.ipynb
│   │   ├── 05_CNN_RNN.ipynb
│   │   ├── 05_CNN_RNN_aug.ipynb
│   │   ├── 06_LSTM.ipynb
│   │   ├── 06_LSTM_aug.ipynb
│   │   ├── 07_CNN1.ipynb
│   │   ├── 07_CNN1_aug.ipynb
│   │   ├── 08_CNN2.ipynb
│   │   ├── 08_CNN2_aug.ipynb
│   │   ├── 09_CNN3.ipynb
│   │   ├── 09_CNN3_aug.ipynb
│   │   ├── 10_CNN4.ipynb
│   │   ├── 11_MIT_AST_V1.ipynb
│   │   ├── 11_MIT_AST_V1_3s_25.ipynb
│   │   ├── 12_MIT_AST_V2.ipynb
│   │   ├── 12_MIT_AST_V2_3s_25.ipynb
│   │   ├── 13_PANN_Save_Embeddings.ipynb
│   │   ├── 13_PANN_Save_Embeddings_3s_25.ipynb
│   │   ├── 14_PANN_LogisticRegression.ipynb
│   │   ├── 14_PANN_LogisticRegression_3s_25.ipynb
│   │   ├── 15_PANN_SVM_RBF.ipynb
│   │   ├── 16_PANN_MLP.ipynb
│   │   ├── 16_PANN_MLP_3s_25.ipynb
│   │   ├── 17_PANN_RandomForest.ipynb
│   │   └── 17_PANN_RandomForest_3s_25.ipynb
│   │
│   └── 03_statistics/
│       ├── 01_dataset_statistics.ipynb
│       └── 02_misclassification_statistics.ipynb
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── confusion_matrices/
│
└── models/
    └── saved_models/
```

## Dataset

This project uses the Free Music Archive dataset.

The expected local input files and folders are:

```text
fma_metadata/tracks.csv
fma_large/
```

The original MP3 files and generated WAV files are not included in this repository.

Before running the notebooks, configure the local dataset paths in:

```text
src/config.py
```

Example:

```python
BASE_DIR = Path("J:/Documents/FH/MA-Daten")
```

## Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The project uses packages such as NumPy, pandas, scikit-learn, librosa, LightGBM, TensorFlow, PyTorch, torchaudio and transformers. These dependencies are listed in `requirements.txt`.

## Configuration

Central paths and experiment parameters are defined in:

```text
src/config.py
```

Important configuration values include:

```python
BASE_DIR
RAW_TRACKS_CSV
PROCESSED_TRACKS_CSV
MP3_AUDIO_DIR
WAV_AUDIO_DIR
AUGMENTED_WAV_AUDIO_DIR
SPLIT_OUTPUT_DIR

SAMPLE_RATE
N_MFCC
SEGMENT_LENGTH_SECONDS
N_SEGMENTS_PER_TRACK

N_TRAIN_PER_GENRE
N_VAL_PER_GENRE
N_TEST_PER_GENRE

EXCLUDED_GENRES
RANDOM_STATE

MFCC_MEAN_TRAIN_PKL
MFCC_MEAN_VAL_PKL
MFCC_MEAN_TEST_PKL

MFCC_MATRIX_TRAIN_PKL
MFCC_MATRIX_VAL_PKL
MFCC_MATRIX_TEST_PKL

TRAINED_MODELS_DIR
MODEL_RESULTS_DIR

AST_AUDIO_DIR
AST_METADATA_CSV
AST_SAMPLE_RATE
AST_MODEL_CHECKPOINT

PANN_AUDIO_DIR
PANN_METADATA_CSV
PANN_CHECKPOINT_PATH
PANN_EMBEDDING_FILE
PANN_3S25_EMBEDDING_FILE
```

## Preprocessing Workflow

The preprocessing notebooks are located in:

```text
notebooks/01_preprocessing/
```

They should be executed in the following order.

### 1. Convert MP3 Files to WAV

```text
01_MP3_to_WAV.ipynb
```

This notebook converts the original MP3 audio files into WAV format.

Main functionality:

- recursive MP3 file search
- MP3-to-WAV conversion
- local storage of converted WAV files

The generated WAV files are not committed to the repository.

### 2. Prepare Metadata

```text
02_Prepare_Metadata.ipynb
```

This notebook prepares the FMA metadata for the following preprocessing and feature extraction steps.

Main functionality:

- loading the original FMA metadata
- preparing the `track_id` column
- checking the availability of top-level genre labels
- saving a cleaned metadata file

### 3. Create Audio Augmentation

```text
03_create_audio_augmentation.ipynb
```

This notebook creates augmented WAV files for selected training experiments.

Implemented augmentation methods include:

- Gaussian noise
- pitch shifting
- time stretching
- gain adjustment
- time shifting

Validation and test data are kept unchanged in the later evaluation workflow.

### 4. Extract MFCC Segment Datasets

```text
04_extract_mfcc_segments.ipynb
```

This notebook creates MFCC matrix datasets from randomly sampled audio segments.

For each selected track, a fixed number of random audio segments is sampled. Each segment is transformed into an MFCC matrix.

These datasets are mainly used for deep learning models such as CNN, ResNet, CNN-RNN and LSTM models.

### 5. Extract MFCC Mean Feature Datasets

```text
05_extract_mfcc_mean_features.ipynb
```

This notebook creates tabular MFCC feature datasets.

For each randomly sampled segment, MFCCs are computed and aggregated using mean and standard deviation values. These feature vectors are used for classical machine learning models such as:

- Random Forest
- LightGBM
- Support Vector Machine

## Model Training

The model notebooks are located in:

```text
notebooks/02_models/
```

The models are grouped into classical machine learning models, deep learning models and transfer learning models.

## Classical Machine Learning Models

### Random Forest

```text
01_RandomForest.ipynb
```

This notebook trains and evaluates a Random Forest classifier using MFCC mean and standard deviation features.

The notebook includes:

- loading tabular MFCC feature datasets
- feature preparation and optional scaling
- Random Forest training
- validation and test evaluation
- segment-level evaluation
- track-level aggregation
- misclassification export

A validation-based hyperparameter search is defined for selected Random Forest parameters.

### LightGBM

```text
02_LightGBM.ipynb
```

This notebook trains and evaluates a LightGBM classifier using MFCC mean and standard deviation features.

The model is trained on the training set and monitored on the validation set using early stopping. The final model is evaluated on the test set.

### Support Vector Machine

```text
03_SVM.ipynb
```

This notebook trains and evaluates a Support Vector Machine classifier using MFCC mean and standard deviation features.

The SVM uses an RBF kernel and is evaluated on segment level and track level. If probability estimates are enabled, track-level predictions can be obtained using mean-probability aggregation.

## Deep Learning Models

The deep learning notebooks use MFCC matrices as input. These matrices are extracted from 25 random 3-second segments per track.

### ResNet

```text
04_ResNet.ipynb
04_ResNet_aug.ipynb
```

These notebooks train and evaluate residual neural network models using MFCC matrix inputs.

The augmented variant uses augmented audio data for training, while validation and test data remain unchanged.

### CNN-RNN

```text
05_CNN_RNN.ipynb
05_CNN_RNN_aug.ipynb
```

These notebooks train and evaluate a hybrid convolutional and recurrent neural network model.

The convolutional layers extract local time-frequency patterns, while the recurrent part models sequential dependencies.

### LSTM

```text
06_LSTM.ipynb
06_LSTM_aug.ipynb
```

These notebooks train and evaluate LSTM-based models using MFCC matrix inputs.

The recurrent architecture is used to model temporal dependencies within MFCC sequences.

### CNN Variants

```text
07_CNN1.ipynb
07_CNN1_aug.ipynb
08_CNN2.ipynb
08_CNN2_aug.ipynb
09_CNN3.ipynb
09_CNN3_aug.ipynb
10_CNN4.ipynb
```

These notebooks contain different CNN architectures for MFCC-based genre classification.

The augmented notebooks use augmented audio data for training experiments.

## Transfer Learning Models

### Audio Spectrogram Transformer

```text
11_MIT_AST_V1.ipynb
11_MIT_AST_V1_3s_25.ipynb
12_MIT_AST_V2.ipynb
12_MIT_AST_V2_3s_25.ipynb
```

These notebooks fine-tune Audio Spectrogram Transformer models for music genre classification.

The AST experiments include:

- full-track input variants
- 3-second segment variants with 25 segments per track
- segment-level evaluation
- track-level aggregation for segmented inputs

### Pretrained Audio Neural Networks

```text
13_PANN_Save_Embeddings.ipynb
13_PANN_Save_Embeddings_3s_25.ipynb
```

These notebooks extract PANN embeddings from audio files.

The generated embeddings are saved and reused by the PANN classifier notebooks to avoid repeating the computationally expensive embedding extraction step.

### PANN Classifiers

```text
14_PANN_LogisticRegression.ipynb
14_PANN_LogisticRegression_3s_25.ipynb
15_PANN_SVM_RBF.ipynb
16_PANN_MLP.ipynb
16_PANN_MLP_3s_25.ipynb
17_PANN_RandomForest.ipynb
17_PANN_RandomForest_3s_25.ipynb
```

These notebooks train classifiers on top of precomputed PANN embeddings.

The evaluated classifiers include:

- Logistic Regression
- Support Vector Machine with RBF kernel
- Multilayer Perceptron
- Random Forest

Both full-track and 3s/25-segment PANN variants are included.

## Statistics and Error Analysis

The statistics notebooks are located in:

```text
notebooks/03_statistics/
```

### Dataset Statistics

```text
01_dataset_statistics.ipynb
```

This notebook analyzes the genre distribution of the available WAV files.

The resulting statistics are used to describe the available dataset sizes and class distributions.

### Misclassification Statistics

```text
02_misclassification_statistics.ipynb
```

This notebook analyzes misclassified tracks across multiple models.

The goal is to identify tracks that are consistently misclassified by several models and to inspect common confusion patterns between true and predicted genres.

## Evaluation Levels

The project distinguishes between two evaluation levels.

### Segment-Level Evaluation

In segment-level evaluation, each extracted audio segment is treated as one model input. This is useful for evaluating how well models classify individual short audio excerpts.

### Track-Level Evaluation

In track-level evaluation, all segment predictions belonging to the same track are aggregated into one final prediction per track.

The implemented aggregation strategies include:

- majority voting
- mean probability aggregation
- mean decision score aggregation where applicable

Track-level evaluation is especially important for segmented input strategies because the final task is genre classification at track level.

## Generated Files

The following generated files are not included in the repository:

```text
*.wav
*.mp3
*.pkl
*.joblib
*.keras
*.h5
*.pt
*.pth
```

This includes:

- converted WAV files
- augmented audio files
- extracted MFCC datasets
- PANN embedding files
- trained model files
- large intermediate result files

These files must be generated locally by running the notebooks.

## Reproducibility Note

The repository contains the code used for preprocessing, feature extraction, model training and evaluation.

Due to the size of the dataset and the computational cost of several experiments, generated datasets, trained models and large intermediate outputs are not included.

The notebooks document the experimental workflow and can be adapted to reproduce the experiments after configuring the local dataset paths in `src/config.py`.

Minor deviations may occur due to hardware differences, library versions or non-deterministic behavior in some audio processing and deep learning operations.

## Suggested Execution Order

A typical execution order is:

```text
1. Configure paths in src/config.py
2. Run notebooks/01_preprocessing/01_MP3_to_WAV.ipynb
3. Run notebooks/01_preprocessing/02_Prepare_Metadata.ipynb
4. Optional: run notebooks/01_preprocessing/03_create_audio_augmentation.ipynb
5. Run notebooks/01_preprocessing/04_extract_mfcc_segments.ipynb
6. Run notebooks/01_preprocessing/05_extract_mfcc_mean_features.ipynb
7. Run selected model notebooks in notebooks/02_models/
8. Run statistics notebooks in notebooks/03_statistics/
```

Transfer learning workflows may require additional setup, such as downloading pretrained model checkpoints for PANN or using GPU-enabled environments for AST training.

## Author

Jasmin Rotheneder

Master's Thesis  
Music Genre Classification using Spectral Analysis
