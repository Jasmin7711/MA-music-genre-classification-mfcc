from pathlib import Path

# Base path
BASE_DIR = Path("J:/Documents/FH/MA-Daten")

# Metadata paths
FMA_METADATA_DIR = BASE_DIR / "fma_metadata"
RAW_TRACKS_CSV = FMA_METADATA_DIR / "tracks.csv"
PROCESSED_TRACKS_CSV = BASE_DIR / "tracks.csv"

# Audio directories
MP3_AUDIO_DIR = BASE_DIR / "fma_large"
WAV_AUDIO_DIR = BASE_DIR / "wav_large"
AUGMENTED_WAV_AUDIO_DIR = BASE_DIR / "wav_large_augmented"
INVALID_AUDIO_DIR = BASE_DIR / "fma_large_invalid"

# Output directories
SPLIT_OUTPUT_DIR = BASE_DIR / "split"
SPLIT_AUG_OUTPUT_DIR = BASE_DIR / "split_aug"

# Audio and feature extraction settings
TARGET_WAV_LENGTH = 1321967
SAMPLE_RATE = 22050
N_MFCC = 20

SEGMENT_LENGTH_SECONDS = 3
N_SEGMENTS_PER_TRACK = 25

# Dataset split settings
N_TRAIN_PER_GENRE = 427
N_VAL_PER_GENRE = 53
N_TEST_PER_GENRE = 53

EXCLUDED_GENRES = [
    "Experimental",
    "International",
    "Instrumental"
]

USE_TOP_GENRES = True
TOP_N_GENRES = 11

# Audio augmentation settings
N_AUGMENTATIONS = 5

# Reproducibility
RANDOM_STATE = 42

# MFCC mean feature files for classical ML models: 3s / 25 segments
MFCC_MEAN_TRAIN_PKL = SPLIT_OUTPUT_DIR / "train_large_mfcc_mean_3s_25_427_53_53_filtered_genres.pkl"
MFCC_MEAN_VAL_PKL = SPLIT_OUTPUT_DIR / "val_large_mfcc_mean_3s_25_427_53_53_filtered_genres.pkl"
MFCC_MEAN_TEST_PKL = SPLIT_OUTPUT_DIR / "test_large_mfcc_mean_3s_25_427_53_53_filtered_genres.pkl"

# Model outputs
TRAINED_MODELS_DIR = BASE_DIR / "Trained_Models"
MODEL_RESULTS_DIR = BASE_DIR / "Results"

# MFCC matrix feature files for deep learning models: 3s / 25 segments
MFCC_MATRIX_TRAIN_PKL = SPLIT_OUTPUT_DIR / "mfcc_large_25_3s_20mfcc_train_427_53_53_filtered_genres.pkl"
MFCC_MATRIX_VAL_PKL = SPLIT_OUTPUT_DIR / "mfcc_large_25_3s_20mfcc_val_427_53_53_filtered_genres.pkl"
MFCC_MATRIX_TEST_PKL = SPLIT_OUTPUT_DIR / "mfcc_large_25_3s_20mfcc_test_427_53_53_filtered_genres.pkl"


# AST settings
AST_AUDIO_DIR = WAV_AUDIO_DIR
AST_METADATA_CSV = PROCESSED_TRACKS_CSV

AST_SAMPLE_RATE = 16000
AST_MODEL_CHECKPOINT = "MIT/ast-finetuned-audioset-16-16-0.442"

MAX_AUDIO_LENGTH_SECONDS = 30


# PANN settings
PANN_AUDIO_DIR = WAV_AUDIO_DIR
PANN_METADATA_CSV = PROCESSED_TRACKS_CSV

PANN_SAMPLE_RATE = 16000
PANN_SEGMENT_LENGTH_SECONDS = 3
PANN_N_SEGMENTS_PER_TRACK = 25

# Adjust this path to your local PANN checkpoint location.
PANN_CHECKPOINT_PATH = Path(
    "J:/Documents/FH/Models/audioset_tagging_cnn-master/pytorch/Cnn14_mAP=0.431.pth"
)
PANN_3S25_EMBEDDING_FILE = BASE_DIR / "Models_final" / "PANN_25x3s_embeddings.joblib"
PANN_EMBEDDING_FILE = BASE_DIR / "Models_final" / "PANN_full_embeddings.joblib"