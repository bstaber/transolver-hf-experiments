import json
import os
from typing import Any, Optional

import torch
from huggingface_hub import HfApi, upload_folder
from transformers import PretrainedConfig


def build_config_dict(
    cfg: PretrainedConfig,
    *,
    model_type: str,
    architectures: list[str],
    auto_map_module: str,
    auto_map_model_class: str,
    auto_map_config_class: str,
    extra_fields: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    base = cfg.to_dict()
    config = {
        "model_type": model_type,
        "architectures": architectures,
        "auto_map": {
            "AutoConfig": f"{auto_map_module}.{auto_map_config_class}",
            "AutoModel": f"{auto_map_module}.{auto_map_model_class}",
        },
        "torch_dtype": "float32",
        **base,
    }
    if extra_fields:
        config.update(extra_fields)
    return config


def export_to_hf_automodel(
    model: torch.nn.Module,
    cfg_dataclass: PretrainedConfig,
    *,
    save_dir: str,
    model_type: str,
    architectures: list[str],
    auto_map_module: str,
    auto_map_model_class: str,
    auto_map_config_class: str,
    extra_files: Optional[dict[str, Any]] = None,
    repo_id: Optional[str] = None,
    push_to_hub: bool = False,
) -> None:
    """Generic exporter: saves an AutoModel-ready folder for any model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model whose state_dict() will be saved.
    cfg_dataclass : dataclass
        Dataclass with all hyperparameters needed to rebuild the model.
    save_dir : str
        Target directory.
    model_type : str
        HF model_type (e.g. "transolver", "my-cfd-net").
    architectures : list[str]
        e.g. ["TransolverModel"].
    auto_map_module : str
        Module name in the HF repo, e.g. "modeling_transolver".
    auto_map_model_class : str
        e.g. "TransolverModel".
    auto_map_config_class : str
        e.g. "TransolverConfig".
    extra_files : dict[str, Any]
        Extra JSON-like files to dump, e.g. {"norm_stats.json": norms_dict}.
    repo_id : str | None
        HF repo id, e.g. "Nionio/transolver-rotor37-v2".
    push_to_hub : bool
        If True, upload folder to HF.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) config.json
    cfg_dict = build_config_dict(
        cfg_dataclass,
        model_type=model_type,
        architectures=architectures,
        auto_map_module=auto_map_module,
        auto_map_model_class=auto_map_model_class,
        auto_map_config_class=auto_map_config_class,
    )
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # 2) weights
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    # 3) extra JSON files (e.g. norm_stats.json)
    if extra_files:
        for filename, payload in extra_files.items():
            with open(os.path.join(save_dir, filename), "w") as f:
                json.dump(payload, f, indent=2)

    # 4) optional push
    if push_to_hub and repo_id is not None:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        upload_folder(folder_path=save_dir, repo_id=repo_id, repo_type="model")
