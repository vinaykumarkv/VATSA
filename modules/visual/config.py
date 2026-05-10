# VATSA Visual Module Configuration

VISUAL_CONFIG = {
    # Image processing
    "image_size"    : 224,
    "mean"          : [0.485, 0.456, 0.406],
    "std"           : [0.229, 0.224, 0.225],

    # Model architecture
    "backbone"      : "efficientnet_b0",
    "embedding_dim" : 512,
    "num_classes"   : 10,
    "dropout"       : 0.3,

    # Checkpoint
    "checkpoint"    : "vatsa_visual_encoder_cifar10_deeper_unfreeze.pth",

    # CIFAR-10 aligned classes
    "classes"       : [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
}