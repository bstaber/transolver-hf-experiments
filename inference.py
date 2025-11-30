import json

import torch
from huggingface_hub import hf_hub_download
from plaid.bridges import huggingface_bridge as hfb
from torch.utils.data import DataLoader
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load model ----
model = AutoModel.from_pretrained(
    "Nionio/transolver-rotor37-small-train", trust_remote_code=True
).to(device)
model.eval()

# ---- Load normalization stats ----
norm_path = hf_hub_download("Nionio/transolver-rotor37-small-train", "norm_stats.json")
with open(norm_path) as f:
    norms = json.load(f)

# ---- Load a test set ----
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
            "Global/Omega",
            "Global/P",
        ]
    )
    .with_format("torch")
)["test"]

p_mean = torch.tensor(norms["p_mean"], device=device)
p_std = torch.tensor(norms["p_std"], device=device)
scalar_mean = torch.tensor(norms["scalar_mean"], device=device)
scalar_std = torch.tensor(norms["scalar_std"], device=device)

# ---- Example batch ----
dataloader = DataLoader(ds, batch_size=4, shuffle=False)
for batch in dataloader:
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

    # === Normalize global input scalars ===
    input_scalars = torch.concat(
        [batch["Global/Omega"], batch["Global/P"]],
        dim=-1,
    ).to(device)  # [B, 2]
    input_scalars = (input_scalars - scalar_mean) / scalar_std  # normalize
    # Broadcast scalars to nodes
    inputs = input_scalars.unsqueeze(1).expand(-1, normals.shape[1], -1)  # [B, N, 2]

    fx = torch.cat([normals, inputs], dim=-1)  # [B, N, 5]
    pred = model(fx, coords)  # [B, N, 1]

    # Unormalize prediction
    pred = pred * p_std + p_mean  # [B, N, 1]

    break
