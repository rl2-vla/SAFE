# Adapted SAFE for [RL2-VLA](https://github.com/rl2-vla/RL2-VLA)

<b>Note:</b> This fork adapts the SAFE repository to Pi0 used in RL2-VLA. It trains a multi-task failure detector and calibrates per-task/combined conformal prediction (CP) bands across all 4 SimplerEnv tasks. Please refer to the official [SAFE repository](https://github.com/vla-safe/SAFE) for the original implementation.

## Training SAFE on CoVer Pi0 rollouts
Running this involves five steps:

1. **Collect VLA rollouts** for all 4 SimplerEnv tasks using pi0 following instructions from the [RL2-VLA repository](https://github.com/rl2-vla/RL2-VLA). You must organize and rename the rollout folders based on the correct [directory structure](#rollouts-folder-structure), and specify the rollout path in the [training config file](failure_prob/conf/dataset/open_pizero_simpler_bridge.yaml) (`open_pizero_simpler_bridge.yaml`).

2. **Sweep SAFE hyperparameters** - concurrent runs are highly recommended. Training logs and artifacts are saved to wandb.
   ```bash
   # Single instance
   bash scripts/batch_training/submit_opi0_simpler.bash

   # Concurrent runs
   bash scripts/batch_training/opi0_simpler_parallel_runs/submit_opi0_simpler_[0-3].bash
   ```

3. **Determine the best hyperparameters** by sorting the wandb runs (from step 2) based on the ROC-AUC metric. See [example output CSV](scripts/wandb_metrics_batch_v2/SAVED/example_opi0_simpler/opi0_simpler_v1.csv).
   ```bash
   python scripts/get_wandb_metrics.py
   ```

4. **Train with the optimal hyperparameters** to output the final SAFE checkpoint and to calibrate CP bands. See examples: [per-task and combined](scripts/batch_training/logs/SAVED/).
   ```bash
   # Train SAFE ckpt and calibrate CP bands
   bash scripts/batch_training/train_optimal_opi0_simpler.bash

   # Visualize CP bands
   python scripts/batch_training/visualize_cp_band.py

   # Rerun for optimal hyperparams training run (note: change WANBD_PROJECT_NAME)
   python scripts/get_wandb_metrics.py  
   ```

5. **Generate balanced-accuracy plots** using the CSV and wandb artifacts from steps 3-4 to identify the top-k alpha hyperparameters. See [example plots and json](scripts/notebooks/figures/example_plots_opi0_simpler/opi0_cp_plots_per_task_per_seed_safe_s0/opi0_simpler_v1-per_task-bal_acc_seed0.pdf).
   ```bash
   python scripts/eval_conformal_figure.py
   ```

## Rollouts Folder Structure

Rollouts must be organized by task, then by seed, with each episode's `.mp4`, `.pkl`, and `_meta.json` files inside the corresponding seed folder. Please follow the structure and naming convention as follows:

```bash
rl2_vla_collected_rollouts/
├── widowx_carrot_on_plate/
│   ├── widowx_carrot_on_plate_seed0/
│   ├── widowx_carrot_on_plate_seed7/
│   └── widowx_carrot_on_plate_seed42/
├── widowx_put_eggplant_in_basket/
├── widowx_spoon_on_towel/
└── widowx_stack_cube/
    ├── widowx_stack_cube_seed0/
    ├── widowx_stack_cube_seed7/
    └── widowx_stack_cube_seed42/
        ├── episode_1_success_False_meta.json
        ├── episode_1_success_False.mp4
        ├── episode_1_success_False.pkl
        ├── episode_2_success_False_meta.json
        ├── episode_2_success_False.mp4
        ├── episode_2_success_False.pkl
        └── ...
```

___

<br>



# SAFE: Multitask Failure Detection for Vision-Language-Action Models

**Neural Information Processing Systems (NeurIPS) 2025**

[**Project Page**](https://vla-safe.github.io/) |
[**Paper**](https://arxiv.org/pdf/2506.09937) |
[**ArXiv**](https://arxiv.org/abs/2506.09937)


[Qiao Gu](https://georgegu1997.github.io/),
[Yuanliang Ju](https://scholar.google.com/citations?user=rG90YVAAAAAJ&hl=zh-CN),
[Shengxiang Sun](https://owensun2004.github.io/),
[Igor Gilitschenski](https://www.gilitschenski.org/igor/),
[Haruki Nishimura](https://harukins.github.io/),
[Masha Itkina](https://mashaitkina.weebly.com/),
[Florian Shkurti](https://www.cs.toronto.edu/~florian/)

![Splash Figure](assets/safe-teaser-static.png)

We introduce the **multitask failure detection** problem for VLA models, and propose **SAFE**, a failure detector that can detect failures for unseen tasks zero-shot and achieve state-of-the-art performance. This repo contains the implementation of SAFE. 

## Generate rollouts from VLA models

Please follow the following repo for adapted code that runs VLA models on simulated environments and generates rollouts for failure detection. Detailed instructions can be found in the README files of these repos. 

* [openvla](https://github.com/vla-safe/openvla) for OpenVLA model on the LIBERO benchmark.
* [openpi](https://github.com/vla-safe/openpi) for pi0 and pi0-FAST models on the LIBERO benchmark. 
* [open-pi-zero](https://github.com/vla-safe/open-pi-zero) for pi0* models on the SimplerEnv benchmark. 

After generating the rollouts, please duplicate `setup_envs.bash.template` and edit environment variables inside according to the locations of the generated rollouts.

```bash
cp setup_envs.bash.template setup_envs.bash

# TODO: Please edit the setup_envs.bash file to set the environment variables
```


## Train and evaluate SAFE and baseline failure detectors

### Setup

```bash
git clone git@github.com:vla-safe/SAFE.git

# Create a new conda environment (or other virtual environment management tool)
conda create -n vla-safe python=3.10
conda activate vla-safe

# Install pytorch (the newest version should be fine)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other required packages
pip install pandas scipy pyyaml tqdm imageio[ffmpeg] hydra-core omegaconf scikit-learn opencv_python einops wandb plotly matplotlib natsort flask

# Log in your wandb account
wandb login

# Install this codebase as a package
# cd to the root directory of this repo
pip install -e .
```
# H100 Specifics
```bash
sudo apt install -y libgl1
sudo apt install -y libglib2.0-0
```
### Training and evaluation 

Please see the following file for training and evaluation scripts for SAFE failure detector and all baselines. 

* [scripts/batch_training/submit_openvla_libero.bash](scripts/batch_training/submit_openvla_libero.bash)
* [scripts/batch_training/submit_pi0diff_libero.bash](scripts/batch_training/submit_pi0diff_libero.bash)
* [scripts/batch_training/submit_pi0fast_libero.bash](scripts/batch_training/submit_pi0fast_libero.bash)
* [scripts/batch_training/submit_opi0_simpler.bash](scripts/batch_training/submit_opi0_simpler.bash)


### Aggregate and plot metrics 

The script `scripts/get_wandb_metrics.py` pulls the evaluation metrics from WandB, aggregates them, and saves them to CSV files, which should reproduce the results in Table 1 of the paper. You can run the script as follows:

```bash
python scripts/get_wandb_metrics.py
```

Other useful scripts are as follows:

```bash
# To generate plots as shown in Figure 1 and Figure 7
python scripts/visualize_features.py 
python scripts/visualize_features.py dataset=openvla_simpler_bridge projector=tsne

# To generate plots as shown in Figure 8
python scripts/eval_conformal_figure.py
```


## Acknowledgements

The SAFE project and this codebase are inspired by and built on the following repos:

* https://github.com/openvla/openvla
* https://github.com/Physical-Intelligence/openpi
* https://github.com/allenzren/open-pi-zero
* https://github.com/CXU-TRI/FAIL-Detect



## Reference

Please cite our work if you find it useful:

```bibtex
@article{gu2025safe,
  title={SAFE: Multitask Failure Detection for Vision-Language-Action Models},
  author={Gu, Qiao and Ju, Yuanliang and Sun, Shengxiang and Gilitschenski, Igor and Nishimura, Haruki and Itkina, Masha and Shkurti, Florian},
  journal={arXiv preprint arXiv:2506.09937},
  year={2025}
}
```
