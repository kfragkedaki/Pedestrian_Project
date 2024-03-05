import logging
import os
import time
from collections import OrderedDict

import torch
from src.utils.print_helpers import readable_time, Printer, count_parameters
from src.utils.model_helpers import l2_reg_loss, save_model, load_model, get_loss_module, get_optimizer
from src.model.encoder import model_factory

logger = logging.getLogger("__main__")

val_times = {"total_time": 0, "count": 0}

def load_task_model(config):
    """For the task specified in the configuration returns the corresponding combination of
    Model class."""

    task = config["task"]

    if task == "imputation":
        return (
            UnsupervisedAttentionModel
        )
    else:
        raise NotImplementedError("Task '{}' not implemented".format(task))


def create_model(config, train_loader, val_loader, data, logger, device):
    """Create model from configuration"""

    model_class = load_task_model(config)
    model = model_factory(config, data)

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(count_parameters(model)))
    logger.info(
        "Trainable parameters: {}".format(count_parameters(model, trainable=True))
    )

    # Initialize optimizer
    optim_class = get_optimizer(config["optimizer"])
    optimizer = optim_class(model.parameters(), lr=config["lr"])

    start_epoch = 0

    # Load model and optimizer state
    if config["load_model"]:
        model, optimizer, start_epoch = load_model(
            model,
            config["load_model"],
            optimizer,
            config["resume"],
            config["lr"],
            config["lr_step"],
            config["lr_decay"],
        )
    model.to(device)

    loss_module = get_loss_module(config)

    trainer = model_class(
        model,
        train_loader,
        device,
        loss_module,
        optimizer,
        l2_reg=config["l2_reg"],
        print_interval=config["print_interval"],
        console=config["console"],
    )

    val_evaluator = model_class(
        model,
        val_loader,
        device,
        loss_module,
        print_interval=config["print_interval"],
        console=config["console"],
    )

    return trainer, val_evaluator, start_epoch


def evaluate(evaluator, config=None, save_embeddings=True):
    """Perform a single, one-off evaluation on an evaluator object (initialized with a dataset)"""

    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = evaluator.evaluate(
            keep_all=True, save_embeddings=save_embeddings
        )
    eval_runtime = time.time() - eval_start_time

    if save_embeddings:
        embeddings_filepath = os.path.join(
            os.path.join(config["output_dir"], "embeddings.pt")
        )
        torch.save(per_batch["embeddings"], embeddings_filepath)

    print_str = "Evaluation Summary: "
    for k, v in aggr_metrics.items():
        if v is not None:
            print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)
    logger.info(
        "Evaluation runtime: {} hours, {} minutes, {} seconds\n".format(
            *readable_time(eval_runtime)
        )
    )

    return aggr_metrics, per_batch


def validate(
    val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch
):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Evaluating on validation set ...")
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = val_evaluator.evaluate(epoch, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    logger.info(
        "Validation runtime: {} hours, {} minutes, {} seconds\n".format(
            *readable_time(eval_runtime)
        )
    )

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
    logger.info(
        "Avg val. time: {} hours, {} minutes, {} seconds".format(
            *readable_time(avg_val_time)
        )
    )
    logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print_str = "Epoch {} Validation Summary: ".format(epoch)
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar("{}/val".format(k), v, epoch)
        print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)

    key_metric = 'loss'
    # Update Best Model
    if aggr_metrics[key_metric] < best_value:
        best_value = aggr_metrics[config[key_metric]]
        save_model(
            os.path.join(config["save_dir"], "model_best.pth"),
            epoch,
            val_evaluator.encoder,
        )
        best_metrics = aggr_metrics.copy()

    return aggr_metrics, best_metrics, best_value


class BaseModel(object):

    def __init__(
        self,
        encoder,
        dataloader,
        device,
        loss_module,
        optimizer=None,
        l2_reg=None,
        print_interval=10,
        console=True,
    ):

        self.encoder = encoder
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = Printer(console=console)

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError("Please override in child class")

    def evaluate(self, epoch_num=None, keep_all=True, save_embeddings=False):
        raise NotImplementedError("Please override in child class")

    def print_callback(self, i_batch, metrics, prefix=""):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class UnsupervisedAttentionModel(BaseModel):

    def train_epoch(self, epoch_num=None, max_norm=1.0):

        self.encoder = self.encoder.train()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(
                self.device
            )  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            predictions, _ = self.encoder(
                X.to(self.device), padding_masks
            )  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(
                predictions, targets, target_masks
            )  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(
                loss
            )  # mean loss (over active elements) used for optimization

            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.encoder)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.encoder.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=max_norm)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Training " + ending)

            with torch.no_grad():
                total_active_elements += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = (
            epoch_loss / total_active_elements
        )  # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True, save_embeddings=False):

        self.encoder = self.encoder.eval()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch

        if keep_all:
            per_batch = {
                "target_masks": [],
                "targets": [],
                "predictions": [],
                "metrics": [],
                "IDs": [],
            }
            if save_embeddings:
                per_batch["embeddings"] = []

        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(
                self.device
            )  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(
                self.device
            )  # 0s: ignore (because they are padded)

            predictions, embedding = self.encoder(
                X.to(self.device), padding_masks
            )  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(
                predictions, targets, target_masks
            )  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(
                loss
            )  # mean loss (over active elements) used for optimization the batch

            if keep_all:
                per_batch["target_masks"].append(target_masks.cpu().numpy())
                per_batch["targets"].append(targets.cpu().numpy())
                per_batch["predictions"].append(predictions.cpu().numpy())
                per_batch["metrics"].append([loss.cpu().numpy()])
                per_batch["IDs"].append(IDs)
                if save_embeddings:
                    per_batch["embeddings"].append(embedding.cpu().numpy())

            metrics = {"loss": mean_loss}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Evaluating " + ending)

            total_active_elements += len(loss)
            epoch_loss += batch_loss  # add total loss of batch

        epoch_loss = (
            epoch_loss / total_active_elements
        )  # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics
