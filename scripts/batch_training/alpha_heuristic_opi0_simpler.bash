#!/bin/bash

# Run all experiments on SimplerEnv rollouts
# Note: For alpha heuristic, we set combine_seed_cp_band=False for per-seed top-k alphas
# Note: speed up by setting train.skip_video_gen_during_eval=True

GROUP_NAME=opi0_simpler_v1

# LSTM - Per-Task (for in-domain tasks)
for DATASET in open_pizero_simpler_bridge; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${SAFE_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.unseen_task_ratio=0.0 \
        dataset.horizon_idx_rel=mean \
        dataset.diff_idx_rel=0.0 \
        dataset.load_to_cuda=True \
        model=lstm \
        model.lr=0.001 \
        model.lambda_reg=0.1 \
        train.combine_cp_band=False \
        train.combine_seed_cp_band=False \
        train.seed=0 \
        train.eval_save_video_functional=True \
        train.eval_save_video_multiproc=True \
        train.eval_save_ckpt=False \
        model.n_epochs=2000 \
        train.skip_video_gen_during_eval=True \
        train.wandb_project=rl2-vla-safe-per-task-heuristic \
        train.exp_suffix=lstm   
done

# LSTM - Combined Task (for OOD tasks)
for DATASET in open_pizero_simpler_bridge; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${SAFE_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.unseen_task_ratio=0.0 \
        dataset.horizon_idx_rel=mean \
        dataset.diff_idx_rel=0.0 \
        dataset.load_to_cuda=True \
        model=lstm \
        model.lr=0.001 \
        model.lambda_reg=0.1 \
        train.combine_cp_band=True \
        train.combine_seed_cp_band=False \
        train.seed=0 \
        train.eval_save_video_functional=True \
        train.eval_save_video_multiproc=True \
        train.eval_save_ckpt=False \
        model.n_epochs=2000 \
        train.skip_video_gen_during_eval=True \
        train.wandb_project=rl2-vla-safe-combined-heuristic \
        train.exp_suffix=lstm   
done