from hyperopt import hp

hyperparameter_config = {
    "lr": hp.uniform("lr", 1e-6, 1e-3),
    "batch_size": hp.choice("batch_size", [16, 32, 64, 128, 256, 512]),
    "epochs": hp.choice("epochs", [i for i in range(50, 1000, 10)]),
    "optimizer": hp.choice("optimizer", ["Adam", "RAdam"]),
    "l2_reg": hp.quniform("l2_reg", 0, 0.2, 0.05),
    "hidden_dim": hp.choice("hidden_dim", [256, 512]),
    "dropout": hp.quniform("dropout", 0, 0.2, 0.05),
    "activation": hp.choice("activation", ["relu", "gelu"]),
    "data_normalization": hp.choice(
        "data_normalization", ["standardization", "minmax", "none"]
    ),
    "data_chunk_len": hp.choice(
        "data_chunk_len", [i for i in range(50, 160, 10)]
    ),  #  hp.choice("data_chunk_len", [i for i in range(50,  200, 10)]),
    "harden_step": hp.choice("harden_step", [10, 15, 20]),
    "mask_mode": hp.choice("mask_mode", ["separate", "concurrent"]),
    "mask_distribution": hp.choice("mask_distribution", ["geometric", "bernoulli"]),
}
