from src.datasets.data import data_factory, Normalizer
from src.datasets.datasplit import split_dataset, save_indices
from torch.utils.data import DataLoader
from functools import partial
from src.datasets.masked_datasets import ImputationDataset, collate_unsuperv


def load_task_datasets(config):
    """For the task specified in the configuration returns the corresponding combination of
    Task-specific Dataset class and collate function."""

    task = config["task"]

    if task == "imputation":
        return (
            partial(
                ImputationDataset,
                mean_mask_length=config["mean_mask_length"],
                masking_ratio=config["masking_ratio"],
                mode=config["mask_mode"],
                distribution=config["mask_distribution"],
                exclude_feats=config["exclude_feats"],
            ),
            collate_unsuperv,
        )
    else:
        raise NotImplementedError("Task '{}' not implemented".format(task))


def load_data(config, logger):
    """ Load data, and split train and test dataset. If eval_only then only val_dataset will be created."""
    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config["data_class"]]
    my_data = data_class(config, n_proc=config["n_proc"])
    # my_data.tensor_3d.shape[0], (my_data.all_df.groupby(by='track_id').size()/60).sum() # TODO CHECK TENSOR

    # Split dataset
    if config["val_ratio"] == 1:
        val_indices = my_data.all_IDs
        train_indices = []
        logger.info("{} samples may be used for evaluation".format(len(val_indices)))
    else:
        train_indices, val_indices = split_dataset(
            data_indices=my_data.all_IDs,
            validation_ratio=config["val_ratio"],
            random_seed=config["seed"],
        )

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    save_indices(
        indices={"train": train_indices, "val": val_indices},
        folder=config["output_dir"],
    )
    train_data = my_data.feature_df.loc[train_indices]
    val_data = my_data.feature_df.loc[val_indices]

    # Pre-process features
    if config["data_normalization"] is not None:
        normalizer = Normalizer(config["data_normalization"])
        train_data = normalizer.normalize(train_data)
        if len(val_indices):
            val_data = normalizer.normalize(val_data)

    # Initialize data generators
    task_dataset_class, collate_fn = load_task_datasets(config)

    # Dataloaders
    val_dataset = task_dataset_class(val_data, val_indices)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, max_len=my_data.max_seq_len),
    )

    train_loader = None
    if config["val_ratio"] < 1:
        train_dataset = task_dataset_class(train_data, train_indices)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x, max_len=my_data.max_seq_len),
        )

    return train_loader, val_loader, my_data
