"""Training script for Transolver model on Rotor37 dataset."""

import mlflow
import torch
from hf_export import export_to_hf_automodel
from modeling_transolver import TransolverConfig, TransolverModel
from plaid.bridges import huggingface_bridge as hfb
from torch.utils.data import DataLoader

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
    .with_format("torch")
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

norms = {
    "p_mean": float(p_mean.cpu()),
    "p_std": float(p_std.cpu()),
    "scalar_mean": scalar_mean.cpu().tolist(),
    "scalar_std": scalar_std.cpu().tolist(),
}

model = TransolverModel(cfg).to(device)

num_epochs = 100
ds_trainval = ds["train"].train_test_split(test_size=0.2, seed=42, shuffle=True)
train_dl = DataLoader(ds_trainval["train"], batch_size=4, shuffle=True)
val_dl = DataLoader(ds_trainval["test"], batch_size=4, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, total_steps=num_epochs * len(train_dl)
)
loss_fn = torch.nn.MSELoss()
scaler = torch.GradScaler(device=device.type, enabled=True)


def train_one_epoch(model, train_dl, optimizer, loss_fn):
    model.train()
    loss_train = 0.0
    for batch in train_dl:
        optimizer.zero_grad()

        coords = torch.stack(
            [
                batch["Base_2_3/Zone/GridCoordinates/CoordinateX"],
                batch["Base_2_3/Zone/GridCoordinates/CoordinateY"],
                batch["Base_2_3/Zone/GridCoordinates/CoordinateZ"],
            ],
            dim=-1,
        ).to(device)  # [B, N, 3]

        normals = torch.stack(
            [
                batch["Base_2_3/Zone/PointData/NormalsX"],
                batch["Base_2_3/Zone/PointData/NormalsY"],
                batch["Base_2_3/Zone/PointData/NormalsZ"],
            ],
            dim=-1,
        ).to(device)  # [B, N, 3]

        # === Normalize pressure field ===
        pressure = batch["Base_2_3/Zone/PointData/Pressure"].to(device)
        pressure = (pressure - p_mean) / p_std  # normalized target

        # === Normalize global input scalars ===
        input_scalars = torch.concat(
            [batch["Global/Omega"], batch["Global/P"]],
            dim=-1,
        ).to(device)  # [B, 2]
        input_scalars = (input_scalars - scalar_mean) / scalar_std  # normalize
        # Broadcast scalars to nodes
        inputs = input_scalars.unsqueeze(1).expand(
            -1, normals.shape[1], -1
        )  # [B, N, 2]

        fx = torch.cat([normals, inputs], dim=-1)  # [B, N, 5]
        with torch.autocast(device_type=device.type, enabled=False):
            pred = model(fx, coords)  # [B, N, 1]
            loss = loss_fn(pred.squeeze(-1), pressure)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_train += loss.item()

    loss_train /= len(train_dl)
    return loss_train


def val_one_epoch(model, val_dl, loss_fn):
    model.eval()
    loss_val = 0.0
    with torch.no_grad():
        for batch in val_dl:
            coords = torch.stack(
                [
                    batch["Base_2_3/Zone/GridCoordinates/CoordinateX"],
                    batch["Base_2_3/Zone/GridCoordinates/CoordinateY"],
                    batch["Base_2_3/Zone/GridCoordinates/CoordinateZ"],
                ],
                dim=-1,
            ).to(device)  # [B, N, 3]

            normals = torch.stack(
                [
                    batch["Base_2_3/Zone/PointData/NormalsX"],
                    batch["Base_2_3/Zone/PointData/NormalsY"],
                    batch["Base_2_3/Zone/PointData/NormalsZ"],
                ],
                dim=-1,
            ).to(device)  # [B, N, 3]

            # === Normalize pressure field ===
            pressure = batch["Base_2_3/Zone/PointData/Pressure"].to(device)
            pressure = (pressure - p_mean) / p_std  # normalized target

            # === Normalize global input scalars ===
            input_scalars = torch.concat(
                [batch["Global/Omega"], batch["Global/P"]],
                dim=-1,
            ).to(device)  # [B, 2]
            input_scalars = (input_scalars - scalar_mean) / scalar_std  # normalize
            # Broadcast scalars to nodes
            inputs = input_scalars.unsqueeze(1).expand(
                -1, normals.shape[1], -1
            )  # [B, N, 2]

            fx = torch.cat([normals, inputs], dim=-1)  # [B, N, 5]
            with torch.autocast(device_type=device.type, enabled=False):
                pred = model(fx, coords)  # [B, N, 1]
                loss = loss_fn(pred.squeeze(-1), pressure)
            loss_val += loss.item()

        loss_val /= len(val_dl)
    return loss_val


with mlflow.start_run():
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_dl, optimizer, loss_fn)
        val_loss = val_one_epoch(model, val_dl, loss_fn)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

export_to_hf_automodel(
    model=model,
    cfg_dataclass=cfg,
    save_dir=".",
    model_type="transolver",
    architectures=["TransolverModel"],
    auto_map_module="modeling_transolver",
    auto_map_model_class="TransolverModel",
    auto_map_config_class="TransolverConfig",
    extra_files={"norm_stats.json": norms},
    repo_id="Nionio/transolver-rotor37-small-train",
    push_to_hub=False,
)
