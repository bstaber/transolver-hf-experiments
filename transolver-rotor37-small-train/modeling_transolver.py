# modeling_transolver.py

from typing import Optional, Tuple

import torch
from physicsnemo.models.transolver import Transolver as TransolverBase
from transformers import PretrainedConfig, PreTrainedModel


class TransolverConfig(PretrainedConfig):
    model_type = "transolver"

    def __init__(
        self,
        functional_dim: int = 5,
        out_dim: int = 1,
        embedding_dim: Optional[int] = 3,
        n_layers: int = 4,
        n_hidden: int = 128,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 4,
        slice_num: int = 32,
        unified_pos: bool = False,
        ref: int = 8,
        structured_shape: Optional[Tuple[int, ...]] = None,
        use_te: bool = False,
        time_input: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.functional_dim = functional_dim
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.n_head = n_head
        self.act = act
        self.mlp_ratio = mlp_ratio
        self.slice_num = slice_num
        self.unified_pos = unified_pos
        self.ref = ref
        self.structured_shape = structured_shape
        self.use_te = use_te
        self.time_input = time_input


class TransolverModel(PreTrainedModel):
    config_class = TransolverConfig

    def __init__(self, config: TransolverConfig):
        super().__init__(config)

        self.transolver = TransolverBase(
            functional_dim=config.functional_dim,
            out_dim=config.out_dim,
            embedding_dim=config.embedding_dim,
            n_layers=config.n_layers,
            n_hidden=config.n_hidden,
            dropout=config.dropout,
            n_head=config.n_head,
            act=config.act,
            mlp_ratio=config.mlp_ratio,
            slice_num=config.slice_num,
            unified_pos=config.unified_pos,
            ref=config.ref,
            structured_shape=config.structured_shape,
            use_te=config.use_te,
            time_input=config.time_input,
        )

        # Transformers expects the model to register its weights for saving/loading
        self.post_init()

    def forward(
        self,
        fx: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Thin wrapper around TransolverBase.forward.

        Args:
            fx: [B, N, functional_dim] or [B, *structure, functional_dim]
            embedding: position / embeddings
            time: optional time tensor
        """
        return self.transolver(fx, embedding=embedding, time=time)
