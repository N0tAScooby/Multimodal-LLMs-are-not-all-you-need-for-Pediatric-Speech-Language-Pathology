# SLP Audio Training & Sweep Scripts

This repository contains the code to the paper "Multimodal LLMs are not all you need for Pediatric Speech Language Pathology".
It contains the scripts for our experiments on the `SAA-Lab/SLPHelmUltraSuitePlus` dataset.

---
We use Weights and Biases for logging and experiment tracking. Moreover, we use the sweep function of Weights and Biases to perform both hyperparameter search but also provide error bars.


# ASR task
Files:
* train_whisper.py: Runs a Bayesian search sweep.

* whisper_error_bar_run.py: Similar to the script above, but configured as a sweep with fixed hyperparameters and only random seeds are changing. We use this with the best hyperparameters found in the other script to create our error bars. 

How to run:

```bash
python train_whisper.py

# OR

python whisper_error_bar_run.py
```
# Classification T1-3
These files are used to train an audio classification models for T1-3. The training pipeline includes automatic class balancing and gender-specific augmentations.

Files:

* hyperparam_tuning_slp_audio.py: The main hyperparameter search and training script.

* sweep.yaml: The W&B configuration file for the sweep.

How to run:
Initialize sweep agent:
```bash
wandb sweep sweep.yaml
```

Run the agent with the command output from W&B, e.g.
```bash
wandb agent your-username/project/sweepID

```
