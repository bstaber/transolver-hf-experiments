from typing import Any, Optional, Union

import torch
import torch.nn as nn
from plaid.bridges import huggingface_bridge as hfb
from transformers import Trainer, TrainingArguments

from transolver_rotor37_v1.modeling_transolver import TransolverConfig, TransolverModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = (
    hfb.load_dataset_from_hub("fabiencasenave/Rotor37")
    .select_columns(
        [
            "Base_2_3/Zone/GridCoordinates/CoordinateX",
            "Base_2_3/Zone/GridCoordinates/CoordinateY",
            "Base_2_3/Zone/GridCoordinates/CoordinateZ",
            "Base_2_3/Zone/PointData/NormalsX",
            "Base_2_3/Zone/PointData/NormalsY",
            "Base_2_3/Zone/PointData/NormalsZ",
            "Base_2_3/Zone/PointData/Pressure",
            "Global/Omega",
            "Global/P",
        ]
    )
    .with_format("torch", dtype=torch.float32)
)

# Compute normalization constants from training set
train_ds = ds["train"]

all_pressures = []
all_scalars = []

for sample in train_ds:
    p = sample["Base_2_3/Zone/PointData/Pressure"]  # [N]
    all_pressures.append(p)

    s = torch.stack([sample["Global/Omega"], sample["Global/P"]])  # [2]
    all_scalars.append(s)

all_pressures = torch.cat(all_pressures)  # [num_points_total]
all_scalars = torch.stack(all_scalars)  # [num_samples, 2]

# Compute normalization constants
p_mean = all_pressures.mean().to(device)
p_std = all_pressures.std().to(device)

scalar_mean = all_scalars.mean(dim=0).squeeze().to(device)  # shape [2]
scalar_std = all_scalars.std(dim=0).squeeze().to(device)  # shape [2]


# ---- Define custom Trainer ----
class Rotor37Trainer(Trainer):
    def __init__(
        self,
        *args,
        p_mean=None,
        p_std=None,
        scalar_mean=None,
        scalar_std=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # store normalization constants as tensors on the right device later
        self._p_mean = p_mean
        self._p_std = p_std
        self._scalar_mean = scalar_mean
        self._scalar_std = scalar_std
        self.loss_fn = torch.nn.MSELoss()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        device = model.device

        # ---- Extract columns from batch ----
        coords_x = inputs["Base_2_3/Zone/GridCoordinates/CoordinateX"].to(device)
        coords_y = inputs["Base_2_3/Zone/GridCoordinates/CoordinateY"].to(device)
        coords_z = inputs["Base_2_3/Zone/GridCoordinates/CoordinateZ"].to(device)

        normals_x = inputs["Base_2_3/Zone/PointData/NormalsX"].to(device)
        normals_y = inputs["Base_2_3/Zone/PointData/NormalsY"].to(device)
        normals_z = inputs["Base_2_3/Zone/PointData/NormalsZ"].to(device)

        pressure = inputs["Base_2_3/Zone/PointData/Pressure"].to(device)
        omega = inputs["Global/Omega"].to(device)
        Pglob = inputs["Global/P"].to(device)

        # ---- Build coords [B, N, 3] ----
        coords = torch.stack([coords_x, coords_y, coords_z], dim=-1)  # [B, N, 3]

        # ---- Build normals [B, N, 3] ----
        normals = torch.stack([normals_x, normals_y, normals_z], dim=-1)  # [B, N, 3]

        # ---- Normalize pressure ----
        p_mean = self._p_mean.to(device)
        p_std = self._p_std.to(device)
        pressure_norm = (pressure - p_mean) / p_std  # [B, N]

        # ---- Normalize globals [B, 2] -> broadcast to [B, N, 2] ----
        scalars = torch.concat([omega, Pglob], dim=-1)  # [B, 2]
        scalar_mean = self._scalar_mean.to(device)
        scalar_std = self._scalar_std.to(device)
        scalars_norm = (scalars - scalar_mean) / scalar_std  # [B, 2]
        scalars_expanded = scalars_norm.unsqueeze(1).expand(
            -1, normals.shape[1], -1
        )  # [B, N, 2]

        # ---- Build fx [B, N, 5] ----
        fx = torch.cat([normals, scalars_expanded], dim=-1)  # [B, N, 5]

        # ---- Forward + loss ----
        outputs = model(fx, coords)  # [B, N, 1]
        preds = outputs.squeeze(-1)  # [B, N]

        loss = self.loss_fn(preds, pressure_norm)

        return (loss, outputs) if return_outputs else loss


cfg = TransolverConfig(
    functional_dim=5,
    out_dim=1,
    embedding_dim=3,
    n_layers=4,
    n_hidden=128,
    dropout=0.0,
    n_head=8,
    act="gelu",
    mlp_ratio=4,
    slice_num=32,
    unified_pos=False,
    ref=8,
    structured_shape=None,
    use_te=False,  # not using transformer_engine
    time_input=False,
)

model = TransolverModel(cfg).to(device)

training_args = TrainingArguments(
    output_dir="transolver-rotor37-small-train-trainer",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    push_to_hub=False,
    remove_unused_columns=False,
    fp16=True,
)

trainer = Rotor37Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    p_mean=p_mean,
    p_std=p_std,
    scalar_mean=scalar_mean,
    scalar_std=scalar_std,
)

trainer.train()
