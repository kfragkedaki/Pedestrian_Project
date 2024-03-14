import argparse


class Options(object):
    def __init__(self):
        # Initialize Config
        self.parser = argparse.ArgumentParser(
            description="Attention based model for Unsupservised Pretraining of Time Series Data."
        )

        ## Run from config file
        self.parser.add_argument(
            "--config",
            dest="config_filepath",
            help="Configuration .json file (optional). Overwrites existing command-line args!",
        )

        ## Run from command-line arguments
        # I/O
        self.parser.add_argument(
            "--output_dir",
            default="./experiments",
            help="Root output directory. Time-stamped directories will be created inside. Required.",
        )
        self.parser.add_argument(
            "--data_dir", default="./data", help="Data directory. Required."
        )
        self.parser.add_argument("--load_model", help="Path to pre-trained model.")
        self.parser.add_argument(
            "--resume",
            action="store_true",
            help="If set, will load `starting_epoch` and state of optimizer, besides model weights.",
        )
        self.parser.add_argument(
            "--save_all",
            action="store_true",
            help="If set, will save model weights (and optimizer state) for every epoch; otherwise just latest",
        )
        self.parser.add_argument(
            "--name",
            dest="experiment_name",
            default="",
            help="A string identifier/name for the experiment to be run - \
                                    it will be appended to the output directory name, before the timestamp",
        )
        self.parser.add_argument(
            "--comment",
            type=str,
            default="",
            help="A comment/description of the experiment",
        )
        self.parser.add_argument(
            "--records_file",
            default="./records.xls",
            help="Excel file keeping all records of experiments",
        )

        # System
        self.parser.add_argument(
            "--console",
            action="store_true",
            help="Optimize printout for console output; otherwise for file",
        )
        self.parser.add_argument(
            "--print_interval",
            type=int,
            default=1,
            help="Print batch info every this many batches",
        )
        self.parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
        self.parser.add_argument(
            "--n_proc",
            type=int,
            default=-1,
            help="Number of processes for data loading/preprocessing. By default, equals num. of available cores.",
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="dataloader threads. 0 for single-thread.",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            default=1337,
            help="Random seed to use. None by default, set to an integer for reproducibility",
        )

        # Dataset
        self.parser.add_argument(
            "--data_class",
            type=str,
            default="sind",
            help="Which type of data should be processed.",
        )
        self.parser.add_argument(
            "--data_normalization",
            choices={
                "standardization",
                "minmax",
                "per_sample_std",
                "per_sample_minmax",
                "none",
            },
            default="none",
            help="If specified, will apply normalization on the input features of a dataset.",
        )
        self.parser.add_argument(
            "--pattern",
            type=str,
            help="Regex pattern used to select files contained in `data_dir`. If None, all data will be used.",
        )
        self.parser.add_argument(
            "--val_ratio",
            type=float,
            default=0.2,
            help="Proportion of the dataset to be used as a validation set",
        )
        self.parser.add_argument(
            "--data_chunk_len",
            type=int,
            default=90,
            help="""Used to segment the data samples into chunks. Determines maximum input sequence length 
                                 (size of transformer layers). To use the maximum possible length in the data 
                                 (cannot exceed 350), set to 0.""",
        )

        # Training process
        self.parser.add_argument(
            "--task",
            choices={"imputation"},
            default="imputation",
            help=("Training objective/task: imputation of masked values"),
        )
        self.parser.add_argument(
            "--masking_ratio",
            type=float,
            default=0.1,
            help="Imputation: mask this proportion of each variable",
        )
        self.parser.add_argument(
            "--mean_mask_length",
            type=float,
            default=3,
            help="Imputation: the desired mean length of masked segments. Used only when `mask_distribution` is 'geometric'.",
        )
        self.parser.add_argument(
            "--mask_mode",
            choices={"separate", "concurrent"},
            default="separate",
            help=(
                "Imputation: whether each variable should be masked separately "
                "or all variables at a certain positions should be masked concurrently"
            ),
        )
        self.parser.add_argument(
            "--mask_distribution",
            choices={"geometric", "bernoulli"},
            default="geometric",
            help=(
                "Imputation: whether each mask sequence element is sampled independently at random"
                "or whether sampling follows a markov chain (stateful), resulting in "
                "geometric distributions of masked squences of a desired mean_mask_length"
            ),
        )
        self.parser.add_argument(
            "--exclude_feats",
            type=str,
            default=None,
            help="Imputation: Comma separated string of indices corresponding to features to be excluded from masking",
        )
        self.parser.add_argument(
            "--harden",
            action="store_true",
            help="Makes training objective progressively harder, by masking more of the input",
        )
        self.parser.add_argument(
            "--harden_step",
            type=int,
            default=10,
            help="Makes training objective progressively harder every this many epochs",
        )
        self.parser.add_argument(
            "--early_stopping_patience",
            type=str,
            default=10,
            help="Use of Ray for hyperparameter tuning. When the model does not improve for that many consecutive epochs, terminate training. Use None for no early stopping",
        )
        self.parser.add_argument(
            "--early_stopping_delta",
            type=float,
            default=0.0,
            help="Use of Ray for hyperparameter tuning. A change of less than delta in the monitored quantity will count as no improvement",
        )
        self.parser.add_argument(
            "--epochs", type=int, default=500, help="Number of training epochs"
        )
        self.parser.add_argument(
            "--val_interval",
            type=int,
            default=1,
            help="Evaluate on validation set every this many epochs. Must be >= 1.",
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=256,
            help="Number of instances per batch during training",
        )
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument(
            "--lr_step",
            type=int,
            default=1,
            help="Reduce learning rate by a factor of lr_decay every lr_step steps."
            " The default is 1, meaning that the learning rate will decay every epoch.",
        )
        self.parser.add_argument(
            "--lr_decay",
            type=float,
            default=1.0,
            help=("Learning rate decay per lr_step epochs"),
        )
        self.parser.add_argument(
            "--optimizer",
            choices={"Adam", "RAdam"},  # "NAdam", "Adamax"
            default="RAdam",
            help="Optimizer",
        )
        self.parser.add_argument(
            "--l2_reg",
            type=float,
            default=0.0,
            help="L2 weight regularization parameter. Set to 0 for no regularization.",
        )
        self.parser.add_argument(
            "--max_grad_norm",
            type=float,
            default=4.0,
            help="Maximum L2 norm for gradient clipping, default 4.0 (0 to disable clipping)",
        )

        # Evaluation
        self.parser.add_argument(
            "--eval_only",
            action="store_true",
            help="Set this value to only evaluate model",
        )
        self.parser.add_argument(
            "--save_embeddings",
            action="store_true",
            help="Set this value to extract the embeddings when evaluating a model.",
        )

        # Model
        self.parser.add_argument(
            "--embedding_dim",
            type=int,
            default=128,
            help="Internal dimension of transformer embeddings",
        )
        self.parser.add_argument(
            "--hidden_dim",
            type=int,
            default=512,
            help="Dimension of feedforward part in transformer layer",
        )
        self.parser.add_argument(
            "--num_heads",
            type=int,
            default=8,
            help="Number of multi-headed attention heads",
        )
        self.parser.add_argument(
            "--num_layers",
            type=int,
            default=3,
            help="Number of transformer encoder layers (blocks)",
        )
        self.parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="Dropout applied to transformer encoder layers. Set to 0 for no dropout.",
        )
        self.parser.add_argument(
            "--pos_encoding",
            choices={"fixed", "learnable"},
            default="fixed",
            help="Positional encoding to be used in transformer encoder",
        )
        self.parser.add_argument(
            "--activation",
            choices={"relu", "gelu"},
            default="gelu",
            help="Activation to be used in transformer encoder",
        )
        self.parser.add_argument(
            "--normalization_layer",
            choices={"BatchNorm", "LayerNorm"},
            default="BatchNorm",
            help="Normalization layer to be used internally in transformer encoder",
        )

        # Hyperparameter tunning
        self.parser.add_argument(
            "--hyperparameter_tuning",
            action="store_true",
            help="Use of Ray for hyperparameter tuning",
        )

    def parse(self, args=None):

        args = self.parser.parse_args(args)

        if args.exclude_feats is not None:
            args.exclude_feats = [int(i) for i in args.exclude_feats.split(",")]

        if args.early_stopping_patience is not None:
            args.early_stopping_patience = int(args.early_stopping_patience)

        # If evaluating, no validation set is used
        if args.eval_only:
            args.val_ratio = 1.0
            args.save_embeddings = True
            args.dropout = 0.0  # No dropout during evaluation
        elif not args.eval_only and args.val_ratio == 0.0:
            raise ValueError("Validation ratio cannot be 0.0 when training the model")

        return args
