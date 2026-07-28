import os
import random
import hydra
import imageio
from omegaconf import OmegaConf
from tqdm import tqdm, trange

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import wandb

from failure_prob.data import load_rollouts, split_rollouts
from failure_prob.data.utils import Rollout, RolloutDataset, normalize_rollouts_hidden_states
from failure_prob.model import get_model
from failure_prob.model.base import BaseModel
from failure_prob.utils.constants import MANUAL_METRICS, EVAL_TIME_QUANTILES
from failure_prob.utils.timer import Timer
from failure_prob.utils.routines import (
    eval_model_and_log,
    eval_metrics_and_log,
    eval_save_timing_plots,
)
from failure_prob.utils.video import eval_save_videos, eval_save_videos_functional_cp
from failure_prob.utils.random import seed_everything

from failure_prob.conf import Config, process_cfg
import re

# Map natural language task names to directory names
NATURAL_TO_DIR = {
    'stack the green block on the yellow block': 'widowx_stack_cube',
    'put eggplant into yellow basket': 'widowx_put_eggplant_in_basket',
    'put carrot on plate': 'widowx_carrot_on_plate',
    'put the spoon on the towel': 'widowx_spoon_on_towel',
}

def _fmt_val(v):
    """Make v into a compact string for filenames/run-names."""
    # handle lists/tuples (join with '-')
    if isinstance(v, (list, tuple)):
        return "-".join(_fmt_val(x) for x in v)
    # floats: use compact representation (scientific for very small/large)
    if isinstance(v, float):
        # small/large -> scientific, otherwise a compact decimal
        if abs(v) < 1e-3 or abs(v) >= 1e3:
            s = "{:.0e}".format(v)   # e.g. '1e-05' or '1e+03'
        else:
            s = "{:.6g}".format(v)   # e.g. '0.001' -> '0.001'
        # tidy exponents: '1e-05' -> '1e-5', '1e+03' -> '1e3'
        s = re.sub(r'e-0+','e-', s)
        s = re.sub(r'e\+0*','e', s)
        return s
    # ints / strings / others
    return str(v)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    cfg = process_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    if (
        cfg.train.eval_save_logs 
        or cfg.train.eval_save_video 
        or cfg.train.eval_save_ckpt 
        or cfg.train.eval_save_video_functional
        or cfg.train.eval_save_timing_plots
    ):
        os.makedirs(cfg.train.logs_save_path, exist_ok=True)
        # Save the config
        with open(os.path.join(cfg.train.logs_save_path, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
            
    # Set seed for randomness in loading rollouts
    seed_everything(0)
            
    # Load and preprocess the rollout data
    with Timer("Loading rollouts"):
        all_rollouts = load_rollouts(cfg)
        print(f"Loaded {len(all_rollouts)} rollouts")
        if cfg.dataset.load_to_cuda:
            all_rollouts = [r.to("cuda") for r in all_rollouts]

    if len(all_rollouts) == 0:
        raise ValueError(f"No rollouts loaded from {cfg.dataset.data_path}")
    
    if cfg.dataset.normalize_hidden_states:
        # Normalize the hidden states to zero mean and unit variance
        all_rollouts = normalize_rollouts_hidden_states(all_rollouts)
    
    # Seed again for splitting rollouts
    if isinstance(cfg.train.seed, int):
        seeds = [cfg.train.seed]
    elif cfg.train.seed.isnumeric() and "-" not in cfg.train.seed:
        seeds = [int(cfg.train.seed)]
    else:
        # assert there are only integer seeds separated by "-" in cfg.train.seed
        assert all(s.isdigit() for s in cfg.train.seed.split("-")), "All seeds must be integers separated by '_'"

        # Run different seeds in the same call, to speed up 
        seeds = [int(s) for s in cfg.train.seed.split("-")]
        
    for seed in seeds:
        print(f"Running seed {seed}")
        cfg.train.seed = seed
        seed_everything(seed)
        
        # pick values (guard against missing fields)
        model_name = getattr(cfg.model, "name", "model")
        lr_val = getattr(cfg.model, "lr", None)
        reg_val = getattr(cfg.model, "lambda_reg", None)
        seed_val = getattr(cfg.train, "seed", None)
        dataset_name = getattr(cfg.dataset, "name", None)
        token_idx_rel = getattr(cfg.dataset, "token_idx_rel", None)
        feat_name = getattr(cfg.dataset, "feat_name", None)

        # format them
        lr_str = _fmt_val(lr_val) if lr_val is not None else "lr?"
        reg_str = _fmt_val(reg_val) if reg_val is not None else "reg?"
        seed_str = _fmt_val(seed_val) if seed_val is not None else "seed?"
        dataset_str = dataset_name if dataset_name is not None else "data"
        token_str = str(token_idx_rel) if token_idx_rel is not None else "tok?"
        feat_str = str(feat_name) if feat_name is not None else "feat?"

        # build the run name
        run_name = (
            f"{model_name}-{dataset_str}-"
            f"{feat_str}-"
            f"tok{token_str}-"
            f"lr{lr_str}-"
            f"reg{reg_str}-"
            f"seed{seed_str}"
        )

        # init wandb
        wandb.init(
            project=cfg.train.wandb_project,
            dir=cfg.train.wandb_dir,
            name=run_name,
            group=cfg.train.wandb_group_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )

        
        rollouts_by_split_name = split_rollouts(cfg, all_rollouts)
        
        train_rollouts = rollouts_by_split_name["train"]
        input_dim = train_rollouts[0].hidden_states.shape[-1]
        print(f"hidden_feature: {train_rollouts[0].hidden_states.shape} {train_rollouts[0].hidden_states.dtype}")

        # Construct datasets and dataloaders from the rollouts
        dataset_by_split_name = {
            k: RolloutDataset(cfg, v) 
            for k, v in rollouts_by_split_name.items()
        }
        dataloader_by_split_name = {
            k: DataLoader(
                v, 
                batch_size=cfg.model.batch_size, 
                shuffle="train" in k, 
                num_workers=0)
            for k, v in dataset_by_split_name.items()
        }

        # Plot and log the precomputed metrics
        if cfg.train.log_precomputed or cfg.train.log_precomputed_only:
            to_be_logged = eval_metrics_and_log(
                cfg,
                rollouts_by_split_name, 
                MANUAL_METRICS[cfg.dataset.name],
                EVAL_TIME_QUANTILES[cfg.dataset.name],
            )
            to_be_logged['epoch'] = 0
            wandb.log(to_be_logged)
        
        if cfg.train.log_precomputed_only:
            wandb.finish(quiet=True)
            continue
        
        model: BaseModel = get_model(cfg, input_dim)
        print(model)
        model.to("cuda")
        
        optimizer, lr_scheduler = model.get_optimizer()

        n_epochs = cfg.model.n_epochs
        epoch_losses = []
        pbar = trange(n_epochs)
        for epoch in pbar:
            to_be_logged = {"epoch": epoch + 1}
            
            # Training
            model.train()
            avg_loss = model.train_epoch(optimizer, dataloader_by_split_name["train"])
            epoch_losses.append(avg_loss)
            pbar.set_description(f"Avg Loss: {avg_loss:.4f}")
            to_be_logged["train_loss"] = avg_loss

            if lr_scheduler is not None:
                lr_scheduler.step()
                to_be_logged["learning_rate"] = optimizer.param_groups[0]['lr']
            
            # Evaluation
            model.eval()
            if epoch % cfg.train.roc_every == 0 or epoch == n_epochs - 1:
                eval_logs = eval_model_and_log(
                    cfg,
                    model, 
                    rollouts_by_split_name, 
                    dataloader_by_split_name,
                    EVAL_TIME_QUANTILES[cfg.dataset.name], 
                    plot_score_curves = epoch == n_epochs - 1,
                    plot_auc_curves = epoch == n_epochs - 1,
                    log_classification_metrics = epoch == n_epochs - 1,
                )
                to_be_logged.update(eval_logs)
                
            wandb.log(to_be_logged)
        
        # Final evaluation
        if cfg.train.eval_save_video:
            for split, dataloader in dataloader_by_split_name.items():
                if split == "train":
                    continue
                video_save_folder = os.path.join(cfg.train.logs_save_path, f"videos_{split}")
                print("Saving videos to", os.path.abspath(video_save_folder))
                os.makedirs(video_save_folder, exist_ok=True)
                eval_save_videos(dataloader, model, cfg, video_save_folder)
                
        if cfg.train.eval_save_video_functional:
            video_save_folder = os.path.join(cfg.train.logs_save_path, f"videos_functional")
            os.makedirs(video_save_folder, exist_ok=True)

            # Determine calibration and test splits
            calib_split_names = ["val_seen"] if "val_seen" in rollouts_by_split_name else ["train"]
            if "val_unseen" in rollouts_by_split_name:
                test_split_names = ["val_unseen"]
            elif "val_seen" in rollouts_by_split_name:
                test_split_names = ["val_seen"]
            else:
                print("No val_unseen split found. Using train split for testing (for CP band visualization only).")
                test_split_names = ["train"]

            # Helper: extract collection seed string from rollout file path.
            # files at .../task_folder/task_seedX_subfolder/episode.*
            _seed_re = re.compile(r'_seed(\d+)$')
            def get_rollout_seed(r):
                subfolder = os.path.basename(os.path.dirname(r.mp4_path.replace(".mp4", ".pkl")))
                m = _seed_re.search(subfolder)
                return m.group(1) if m else '?'

            # ========== CONDITIONAL: CP band grouping ==========
            # combine_cp_band=True,  combine_seed_cp_band=True  -> global pooled (all tasks, all seeds)
            # combine_cp_band=True,  combine_seed_cp_band=False -> per collection-seed (tasks pooled)
            # combine_cp_band=False, combine_seed_cp_band=True  -> per task (seeds pooled)
            # combine_cp_band=False, combine_seed_cp_band=False -> per (task, seed)

            if cfg.train.combine_cp_band and cfg.train.combine_seed_cp_band:
                # === Pooled CP band across all tasks and seeds ===
                print("\n[Config] combine_cp_band=True, combine_seed_cp_band=True: Generating POOLED CP band across all tasks and seeds")

                cp_band_by_alpha, df = eval_save_videos_functional_cp(
                    cfg, model,
                    rollouts_by_split_name,
                    dataloader_by_split_name,
                    video_save_folder,
                    alpha = cfg.train.eval_cp_alpha,
                    calib_split_names=calib_split_names,
                    test_split_names=test_split_names,
                )

                # Save the CP band separately
                print("\ncp_band_by_alpha", cp_band_by_alpha)
                if cp_band_by_alpha is not None:
                    cp_band_by_alpha_path = os.path.join(cfg.train.logs_save_path, "cp_band_by_alpha.npy")
                    np.save(cp_band_by_alpha_path, cp_band_by_alpha)
                    print(f"Saved CP band to {os.path.abspath(cp_band_by_alpha_path)}")

            elif cfg.train.combine_cp_band and not cfg.train.combine_seed_cp_band:
                # === Per-collection-seed CP bands (tasks pooled within each seed) ===
                print("\n[Config] combine_cp_band=True, combine_seed_cp_band=False: Generating CP bands PER COLLECTION SEED (tasks pooled)")

                unique_seeds = sorted(set(get_rollout_seed(r) for r in all_rollouts))
                print(f"Collection seeds found: {unique_seeds}")

                cp_bands_by_seed = {}
                dfs_by_seed = {}

                for seed_str in unique_seeds:
                    group_name = f"seed{seed_str}"
                    print(f"\n{'='*80}")
                    print(f"SEED: {group_name}".center(80))
                    print(f"{'='*80}\n")

                    # Filter rollouts to this seed across all tasks
                    rollouts_by_split_name_seed = {
                        split: [r for r in rollouts if get_rollout_seed(r) == seed_str]
                        for split, rollouts in rollouts_by_split_name.items()
                    }

                    # Print statistics
                    for split_name, rollouts in rollouts_by_split_name_seed.items():
                        n_success = sum([r.episode_success for r in rollouts])
                        n_fail = len(rollouts) - n_success
                        print(f"{split_name}: {len(rollouts)} rollouts ({n_success} success, {n_fail} fail)")

                    # Check sufficient calibration data
                    calib_rollouts = []
                    for split_name in calib_split_names:
                        if split_name in rollouts_by_split_name_seed:
                            calib_rollouts.extend(rollouts_by_split_name_seed[split_name])
                    n_calib_success = sum([r.episode_success for r in calib_rollouts])
                    if n_calib_success < 5:
                        print(f"WARNING: Only {n_calib_success} successful calibration rollouts for {group_name}. Skipping (need at least 5).\n")
                        continue

                    video_save_folder_seed = os.path.join(video_save_folder, f"seed_{seed_str}")
                    os.makedirs(video_save_folder_seed, exist_ok=True)

                    dataset_by_split_name_seed = {k: RolloutDataset(cfg, v) for k, v in rollouts_by_split_name_seed.items()}
                    dataloader_by_split_name_seed = {
                        k: DataLoader(v, batch_size=cfg.model.batch_size, shuffle=False, num_workers=0)
                        for k, v in dataset_by_split_name_seed.items()
                    }

                    cp_band_by_alpha, df = eval_save_videos_functional_cp(
                        cfg, model,
                        rollouts_by_split_name_seed,
                        dataloader_by_split_name_seed,
                        video_save_folder_seed,
                        alpha=cfg.train.eval_cp_alpha,
                        calib_split_names=calib_split_names,
                        test_split_names=test_split_names,
                    )

                    df['task'] = group_name
                    cp_bands_by_seed[group_name] = cp_band_by_alpha
                    dfs_by_seed[group_name] = df

                # Save and log
                cp_bands_path = os.path.join(cfg.train.logs_save_path, "cp_bands_by_seed.npy")
                np.save(cp_bands_path, cp_bands_by_seed, allow_pickle=True)
                print(f"\nSaved per-seed CP bands to {os.path.abspath(cp_bands_path)}")

                for group_name, df_seed in dfs_by_seed.items():
                    wandb.log({f"classify_cp_functional_per_task/{group_name}": wandb.Table(dataframe=df_seed)})

            else:
                # === Per-task CP bands ===
                print("\n[Config] combine_cp_band=False: Generating SEPARATE CP bands per task")

                # Extract unique task IDs and task names from all rollouts
                all_task_ids = sorted(set([r.task_id for r in all_rollouts]))
                # Create mapping from task_id to task_name (task_description)
                task_id_to_name = {r.task_id: r.task_description for r in all_rollouts}
                task_id_to_name = {tid: NATURAL_TO_DIR.get(name, name) for tid, name in task_id_to_name.items()}

                # Helper: extract seed string from rollout file path.
                # files at .../task_folder/task_seed_subfolder/episode.*
                _seed_re = re.compile(r'_seed(\d+)$')
                def get_rollout_seed(r):
                    subfolder = os.path.basename(os.path.dirname(r.mp4_path.replace(".mp4", ".pkl")))
                    m = _seed_re.search(subfolder)
                    return m.group(1) if m else '?'

                # Build iteration list of (task_id, seed_str) pairs:
                #   combine_seed_cp_band=True  -> one entry per task_id, seed_str=None (all seeds merged)
                #   combine_seed_cp_band=False -> one entry per unique (task_id, seed_str) pair
                if cfg.train.combine_seed_cp_band:
                    task_loop_items = [(task_id, None) for task_id in all_task_ids]
                else:
                    seen_combos = set()
                    task_loop_items = []
                    for r in all_rollouts:
                        key = (r.task_id, get_rollout_seed(r))
                        if key not in seen_combos:
                            seen_combos.add(key)
                            task_loop_items.append(key)
                    task_loop_items.sort()

                print("\n" + "="*80)
                print(f"Generating CP bands for {len(task_loop_items)} groups (combine_seed_cp_band={cfg.train.combine_seed_cp_band}):")
                for task_id, seed_str in task_loop_items:
                    base = task_id_to_name[task_id]
                    label = f"{base}_seed{seed_str}" if seed_str is not None else base
                    print(f"  {label} (task_id={task_id})")
                print("="*80 + "\n")

                # Store CP bands and dataframes per group label
                cp_bands_by_task = {}
                dfs_by_task = {}

                for task_id, seed_str in task_loop_items:
                    base_name = task_id_to_name[task_id]
                    task_name = f"{base_name}_seed{seed_str}" if seed_str is not None else base_name
                    print(f"\n{'='*80}")
                    print(f"TASK {task_id}: {task_name}".center(80))
                    print(f"{'='*80}\n")

                    # === VERIFICATION: Print all rollout task_ids before filtering ===
                    all_task_ids_in_data = sorted(set([r.task_id for split_name, rollouts in rollouts_by_split_name.items() for r in rollouts]))
                    print(f"[VERIFY] All task IDs in rollouts_by_split_name: {all_task_ids_in_data}")

                    # Filter rollouts_by_split_name to only include this task
                    rollouts_by_split_name_task = {
                        split_name: [r for r in rollouts if r.task_id == task_id and (seed_str is None or get_rollout_seed(r) == seed_str)]
                        for split_name, rollouts in rollouts_by_split_name.items()
                    }

                    # === VERIFICATION: Confirm filtering worked ===
                    task_ids_after_filter = sorted(set([r.task_id for split_name, rollouts in rollouts_by_split_name_task.items() for r in rollouts]))
                    print(f"[VERIFY] Task IDs after filtering for {task_name}: {task_ids_after_filter}")
                    assert all(tid == task_id for tid in task_ids_after_filter), f"Unexpected task_ids for {task_name}: {task_ids_after_filter}"

                    # Print statistics for this task
                    for split_name, rollouts in rollouts_by_split_name_task.items():
                        n_success = sum([r.episode_success for r in rollouts])
                        n_fail = len(rollouts) - n_success
                        print(f"{split_name}: {len(rollouts)} rollouts ({n_success} success, {n_fail} fail)")

                    # Check if we have sufficient calibration data
                    calib_rollouts = []
                    for split_name in calib_split_names:
                        if split_name in rollouts_by_split_name_task:
                            calib_rollouts.extend(rollouts_by_split_name_task[split_name])

                    n_calib_success = sum([r.episode_success for r in calib_rollouts])

                    if n_calib_success < 5:
                        print(f"WARNING: Only {n_calib_success} successful calibration rollouts for task {task_id} ({task_name}).")
                        print(f"Skipping CP band generation (need at least 5).\n")
                        continue

                    # Create task-specific video folder
                    video_save_folder_task = os.path.join(video_save_folder, f"task_{task_name}")
                    os.makedirs(video_save_folder_task, exist_ok=True)

                    # === Create task-specific dataloaders ===
                    # NOTE: Dataloaders are used in eval_save_videos_functional_cp to run model forward pass
                    # They must match the filtered rollouts, otherwise scores won't align with rollouts
                    dataset_by_split_name_task = {
                        k: RolloutDataset(cfg, v)
                        for k, v in rollouts_by_split_name_task.items()
                    }
                    dataloader_by_split_name_task = {
                        k: DataLoader(
                            v,
                            batch_size=cfg.model.batch_size,
                            shuffle=False,  # Don't shuffle for eval
                            num_workers=0)
                        for k, v in dataset_by_split_name_task.items()
                    }

                    # === VERIFICATION: Print dataloader sizes ===
                    print(f"[VERIFY] Dataloader sizes for task {task_id}:")
                    for split_name, dataloader in dataloader_by_split_name_task.items():
                        print(f"  {split_name}: {len(dataloader.dataset)} samples")

                    # Call existing CP band generation function with task-filtered data
                    cp_band_by_alpha, df = eval_save_videos_functional_cp(
                        cfg, model,
                        rollouts_by_split_name_task,    # Task-filtered rollouts
                        dataloader_by_split_name_task,  # Task-filtered dataloaders 
                        video_save_folder_task,
                        alpha = cfg.train.eval_cp_alpha,
                        calib_split_names=calib_split_names,
                        test_split_names=test_split_names,
                    )

                    # === VERIFICATION: Print returned CP band structure ===
                    print(f"[VERIFY] CP band structure for {task_name}:")
                    print(f"  cp_band_by_alpha keys: {list(cp_band_by_alpha.keys())}")
                    for alpha_val, band_data in cp_band_by_alpha.items():
                        if isinstance(band_data, np.ndarray):
                            print(f"  Alpha {alpha_val}: direct array, shape {band_data.shape}")
                        elif isinstance(band_data, (list, tuple)):
                            print(f"  Alpha {alpha_val}: list/tuple of length {len(band_data)}, first element shape {band_data[0].shape if len(band_data) > 0 else 'N/A'}")
                        else:
                            print(f"  Alpha {alpha_val}: type {type(band_data)}")

                    # Store results using task_name as key
                    df['task'] = task_name  # overwrite 'all' with actual task name (combine_cp_band=False only)
                    cp_bands_by_task[task_name] = cp_band_by_alpha
                    dfs_by_task[task_name] = df

                # ========== Save all task-specific CP bands ==========

                print("\n" + "="*80)
                print("CP BANDS SUMMARY".center(80))
                print("="*80)

                # === VERIFICATION: Print final structure ===
                print(f"[VERIFY] Total tasks with CP bands: {len(cp_bands_by_task)}")
                print(f"[VERIFY] Task names: {sorted(cp_bands_by_task.keys())}")

                for task_name in sorted(cp_bands_by_task.keys()):
                    cp_band_by_alpha = cp_bands_by_task[task_name]
                    print(f"\nTask '{task_name}':")
                    for alpha, band_data in cp_band_by_alpha.items():
                        if isinstance(band_data, np.ndarray):
                            band = band_data
                        else:
                            band = band_data[0] if len(band_data) > 0 else band_data
                        print(f"  Alpha {alpha}: shape {band.shape}, range [{band.min():.4f}, {band.max():.4f}]")

                # Save to file
                cp_bands_path = os.path.join(cfg.train.logs_save_path, "cp_bands_by_task.npy")
                np.save(cp_bands_path, cp_bands_by_task, allow_pickle=True)
                print(f"\nSaved task-specific CP bands to {os.path.abspath(cp_bands_path)}")

                # === VERIFICATION: Test loading immediately ===
                print(f"[VERIFY] Testing load...")
                loaded_bands = np.load(cp_bands_path, allow_pickle=True).item()
                print(f"[VERIFY] Loaded successfully. Keys: {sorted(loaded_bands.keys())}")

                print("="*80 + "\n")

                # Log per-task CP metrics to W&B
                for task_name, df_task in dfs_by_task.items():
                    task_slug = task_name.replace('/', '_')
                    wandb.log({f"classify_cp_functional_per_task/{task_slug}": wandb.Table(dataframe=df_task)})

        if cfg.train.eval_save_timing_plots:
            plot_save_folder = os.path.join(cfg.train.logs_save_path, f"timing_plots")
            os.makedirs(plot_save_folder, exist_ok=True)
            eval_save_timing_plots(
                cfg, model, 
                rollouts_by_split_name,
                dataloader_by_split_name,
                plot_save_folder,
                alpha= cfg.train.eval_cp_alpha,
            )
            
                
        if cfg.train.eval_save_ckpt:
            os.makedirs(cfg.train.logs_save_path, exist_ok=True)
            ckpt_save_path = os.path.join(cfg.train.logs_save_path, "model_final.ckpt")
            print("Saving model checkpoint to", os.path.abspath(ckpt_save_path))
            torch.save(model.state_dict(), ckpt_save_path)
        
        wandb.finish(quiet=True)
        

if __name__ == "__main__":
    main()
