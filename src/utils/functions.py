import torch
import os
import json
import torch.nn.functional as F
from torch.nn import DataParallel
import sys


class Arguments:
    def __init__(self, data):
        self.__dict__.update(data)

    def __getattr__(self, item):
        return self.__dict__[item]


def load_dataset(name: str):
    from src.datasets import PedestrianData

    dataset = {
        "sind": PedestrianData,
    }.get(name, None)
    assert dataset is not None, "Currently unsupported problem: {}!".format(name)
    return dataset


def load_attention_model(name: str):
    from src.model import AttentionPredictionModel

    model = {
        "pedestrianPrediction": AttentionPredictionModel,
    }.get(name, None)
    assert model is not None, "Currently unsupported model: {}!".format(name)
    return model


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def torch_load_cpu(load_path):
    return torch.load(
        load_path, map_location=lambda storage, loc: storage
    )  # Load on CPU


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print("  [*] Loading model from {}".format(load_path))

    load_data = torch.load(
        os.path.join(os.getcwd(), load_path), map_location=lambda storage, loc: storage
    )

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get("optimizer", None)
        load_model_state_dict = load_data.get("model", load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_model(path, epoch=None):
    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == ".pt"
            )
        model_filename = os.path.join(path, "epoch-{}.pt".format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = os.path.join(path, "args.json")
    problem = load_dataset(args["dataset"])
    model_class = load_attention_model(args["problem"])

    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
        embedding_dim=args["embedding_dim"],
        problem=problem,
        n_encode_layers=args["n_encode_layers"],
        mask_inner=True,
        mask_logits=True,
        normalization=args["normalization"],
        tanh_clipping=args["tanh_clipping"],
        checkpoint_encoder=args.get("checkpoint_encoder", False),
        opts=Arguments(args),
    )
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get("model", {})})

    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def get_baseline_model(model, env, opts, load_data):
    from src.nets.reinforce_baselines import NoBaseline, WarmupBaseline, RolloutBaseline

    # Initialize baseline
    if opts.baseline == "rollout":
        baseline_model = RolloutBaseline(model, env, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline_model = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline_model = WarmupBaseline(
            baseline_model, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta
        )

    # Load baseline from data, make sure script is called with same type of baseline
    if "baseline" in load_data:
        baseline_model.load_state_dict(load_data["baseline"])

    return baseline_model


def load_optimizers(name: str):
    optimizers = {
        "Adam": torch.optim.Adam,
        "NAdam": torch.optim.NAdam,
        "Adamax": torch.optim.Adamax,
    }.get(name, None)
    assert optimizers is not None, "Currently unsupported optimizer: {}!".format(name)
    return optimizers


class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop