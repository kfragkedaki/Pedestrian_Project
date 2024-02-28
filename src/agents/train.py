import os
import time
import tqdm
import torch
import math

from torch.utils.data import DataLoader
from src.utils.functions import move_to, get_inner_model, set_decode_type


def validate(model, dataset, opts):
    # Validate
    print("Validating...")
    cost, length, reward, penalty = rollout(model, dataset, opts, type="validate")
    avg_cost = cost.mean()
    avg_length = length.mean()
    avg_reward = reward.mean()
    avg_penalty = penalty.mean()

    print(
        "Validation overall avg_cost: {} +- {}".format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))
        ),
        " Validation overall avg_length: {} +- {}".format(
            avg_length, torch.std(length) / math.sqrt(len(length))
        ),
        " Validation overall avg_reward: {} +- {}".format(
            avg_reward, torch.std(reward) / math.sqrt(len(reward))
        ),
        " Validation overall avg_penalty: {} +- {}".format(
            avg_penalty, torch.std(penalty) / math.sqrt(len(penalty))
        ),
    )

    return avg_cost, avg_length, avg_reward, avg_penalty


def rollout(model, dataset, opts, type="baseline"):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(batch_data, graph_batch, batch_id):
        with torch.no_grad():
            cost, length, reward, penalty, _ = model(
                move_to(batch_data, opts.device),
                graphs=graph_batch,
                epoch=batch_id,
                type=type,
            )
        return cost.data.cpu(), length.data.cpu(), reward.data.cpu(), penalty.data.cpu()

    results = [
        eval_model_bat(data_batch, graph_batch, batch_id)
        for batch_id, (data_batch, graph_batch) in enumerate(
            tqdm(
                DataLoader(
                    dataset, batch_size=opts.eval_batch_size
                ),
                disable=opts.no_progress_bar,
            )
        )
    ]

    # Transpose results from batch-major to tensor-major
    transposed_results = list(map(list, zip(*results)))

    # Concatenate each tensor
    concatenated_tensors = [torch.cat(tensors, 0) for tensors in transposed_results]

    return concatenated_tensors


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            max_norm
            if max_norm > 0
            else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    )
    return grad_norms, grad_norms_clipped


def train_epoch(
    model,
    optimizer,
    baseline,
    lr_scheduler,
    epoch,
    val_dataset,
    problem,
    opts,
):
    print(
        "Start train epoch {}, lr={} for run {}".format(
            epoch, optimizer.param_groups[0]["lr"], opts.run_name
        )
    )
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(
        problem.make_dataset(
            size=opts.graph_size,
            num_samples=opts.epoch_size,
            distribution=opts.data_distribution,
            num_trucks=opts.num_trucks,
            num_trailers=opts.num_trailers,
            truck_names=opts.truck_names,
            display_graphs=opts.display_graphs,
            r_threshold=opts.battery_limit,
        )
    )
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=opts.batch_size,
        num_workers=0,
    )

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for _, data_batch in enumerate(
        tqdm(training_dataloader, disable=opts.no_progress_bar)
    ):
        train_batch(
            model,
            optimizer,
            baseline,
            step,
            data_batch,
            opts,
        )

        step += 1

    epoch_duration = time.time() - start_time
    print(
        "Finished epoch {}, took {} s".format(
            epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        )
    )

    if (
        opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0
    ) or epoch == opts.n_epochs - 1:
        print("Saving model and state...")
        torch.save(
            {
                "model": get_inner_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "baseline": baseline.state_dict(),
            },
            os.path.join(opts.save_dir, "epoch-{}.pt".format(epoch)),
        )

    avg_val_cost, avg_val_length, avg_val_reward, avg_val_penalty = validate(
        model, val_dataset, opts
    )

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()

    return avg_val_cost, get_inner_model(model).state_dict()


def train_batch(
    model, optimizer, baseline, step, batch, opts
):
    x, graph_batch, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, length, reward, penalty, log_likelihood = model(
        input=x, graphs=graph_batch, epoch=step, type="train"
    )

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = (
        baseline.eval(x, cost, graph_batch) if bl_val is None else (bl_val, 0)
    )

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    optimizer.step()
