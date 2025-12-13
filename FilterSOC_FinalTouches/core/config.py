# core/config.py
import torch
import os


class Config:
    # Cesty
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'training_data.csv')
    MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


    # Model settings
    BERT_MODEL_NAME = "gerulata/slovakbert"
    MAX_LENGTH = 128
    BATCH_SIZE = 4  # Menšie batchy pre GPU s 2GB VRAM

    # Feature extraction
    USE_BERT = True
    USE_TFIDF = True
    BERT_EMBEDDING_DIM = 768
    REDUCED_BERT_DIM = 100
    TFIDF_MAX_FEATURES = 2000

    # Training
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_ESTIMATORS = 100

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Categories
    CATEGORIES = [
        'clickbait', 'conspiracy', 'false_news',
        'propaganda', 'satire', 'misleading',
        'biased', 'legitimate'
    ]