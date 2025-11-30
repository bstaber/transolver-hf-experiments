
# Transolver HF Experiments

This repository contains two separate experiments exploring how to integrate the Transolver architecture with the Hugging Face Hub and the `transformers` library.

The goal of these experiments is to determine practical workflows for:

- exporting custom scientific models to the Hugging Face Hub,
- making them loadable through `AutoModel` and `AutoConfig`,
- training models using the Hugging Face `Trainer`,
- and preparing checkpoints that can be pushed as fully functional HF models.

The content is experimental and focuses on process, not performance.

---

## 1. Exporting a Custom Model to the Hugging Face Hub (AutoModel workflow)

The directory `transolver-rotor37-small-train/` contains:

- a minimal implementation of `TransolverModel` and `TransolverConfig` following the `PreTrainedModel` and `PretrainedConfig` interfaces,
- the `hf_export.py` script, which packages model weights, configuration, and modeling code into an AutoModel-compatible folder,
- a training script (`train_and_export.py`) that trains a small Transolver instance on the Rotor37 dataset and exports it using the Hugging Face Hub format.

This setup demonstrates how to:

1. Train a model normally using PyTorch.
2. Package it into a folder containing:
   - `config.json`
   - `pytorch_model.bin`
   - `modeling_transolver.py`
   - optional normalization statistics or helper files
3. Upload it to the Hugging Face Hub.
4. Load it elsewhere using:

```python
from transformers import AutoModel, AutoConfig

model = AutoModel.from_pretrained("user/repo")
config = AutoConfig.from_pretrained("user/repo")
```

This workflow ensures that the model is self-contained and reproducible without requiring users to manually install the source code.

---

## 2. Using the Hugging Face Trainer with a Custom Scientific Model

The script `trainer_custom_hf.py` explores whether the Hugging Face `Trainer` can be used to train a physics-based Transolver model.

Because the Transolver architecture does not fit standard NLP conventions, a few adjustments are needed to align with the `transformers` API expectations.

We made a custom trainer class, `Rotor37Trainer` , that extends `Trainer` and overrides the loss computation.

Once these adaptations were made, the Trainer was able to run normally:

```python
trainer.train()
```

This produces checkpoints compatible with the Hugging Face export format, enabling them to be uploaded directly without additional conversion steps.

---

## Summary

This repository contains two parallel workflows for integrating a custom architecture with Hugging Face:

1. **Manual export** using a custom script (`hf_export.py`) to generate an AutoModel-compatible folder.
2. **Trainer-based training**, which produces HF-style checkpoints automatically once the model's API matches `transformers` expectations.

These experiments form a foundation for building reproducible and shareable scientific machine learning models using the Hugging Face ecosystem.
