# ⚽ PPO Training in Google Research Football (GFootball)

This repository contains two reinforcement learning training scripts using **Proximal Policy Optimization (PPO)** from Stable-Baselines3 to train agents in the [Google Research Football Environment](https://github.com/google-research/football).

Both scripts include periodic evaluation, checkpoint saving, and training loss logging.

---

## 📂 Scripts Overview

###  `train_simple_mlp_with_checkpoint_loss.py`

- **Environment:** `1_vs_1_easy`
- **State Representation:** `simple115v2`
- **Parallel Environments:** 5
- **Training Interval:** Every **10,000** timesteps
- **Logging & Output Directory:** `./output_simple115_mlp_checkpoint/`
- **Features:**
  - Saves model checkpoints
  - Logs rewards and PPO loss metrics to CSV
  - Evaluates performance every interval

---

###  `train_checkpointLoss_extracted2D.py`

- **Environment:** `1_vs_1_easy`
- **State Representation:** `extracted`
- **Parallel Environments:** 4
- **Training Interval:** Every **30,000** timesteps
- **Logging & Output Directory:** `./output/`
- **Features:**
  - Records full episodes and generates videos
  - Saves the evaluation rewards along with the PPO loss 
  - Saves checkpoints with step count suffix

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install stable-baselines3 pandas
pip install git+https://github.com/google-research/football.git

