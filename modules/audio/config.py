# VATSA Audio Module Configuration

AUDIO_CONFIG = {
    # Audio processing
    "sample_rate"   : 16000,
    "duration"      : 5,           # seconds — ESC-50 standard
    "n_mels"        : 64,
    "fmax"          : 8000,

    # Model architecture
    "backbone"      : "wav2vec2-base",
    "embedding_dim" : 512,
    "num_classes"   : 10,
    "dropout"       : 0.3,
    "hidden_size"   : 768,         # Wav2Vec2-base hidden size

    # Training (reference only)
    "learning_rate" : 1e-4,
    "weight_decay"  : 1e-4,
    "epochs"        : 30,
    "batch_size"    : 16,
    "n_folds"       : 5,

    # Checkpoint
    "checkpoint"    : "modules/audio/vatsa_audio_encoder_transfer.pth",

    # CIFAR-10 aligned classes
    "classes"       : [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
}