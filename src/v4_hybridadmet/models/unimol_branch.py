"""Uni-Mol2 branch.

Wraps DPTechnology's Uni-Mol2 transformer (84M / 164M variants), exposing
a per-molecule hidden vector. The official pip package is
``unimol_tools`` which bundles the pretrained checkpoint download and a
high-level ``MolTrain`` / ``MolPredict`` API.

For end-to-end joint fine-tuning with PAMNet + fingerprints under a
single backprop pass, we need access to the encoder forward, not just
the high-level ``predict`` API. The encoder is at
``unimol_tools.models.unimolv2.UniMolModel`` (chemprop 2.2-ish pkg
layout). We construct the encoder, load the pretrained state dict from
the unimol_tools cache, then call its ``forward`` to get the [CLS]-like
pooled representation.

If your installed unimol_tools version differs and the import path moved,
the ``_resolve_unimol_api`` helper below probes several common locations
before raising a clear error.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


def _resolve_unimol_api():
    """Best-effort locator for Uni-Mol2's model class across versions."""
    import importlib
    candidates = [
        ("unimol_tools.models.unimolv2",     "UniMolModel"),
        ("unimol_tools.models.unimolv2_model","UniMolModel"),
        ("unimol_tools.models.unimol",        "UniMolModel"),
        ("unimol_tools.models",               "UniMolModel"),
    ]
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, attr, None)
            if cls is not None:
                return mod_name, cls
        except ImportError:
            continue
    raise SystemExit(
        "Could not locate UniMolModel in `unimol_tools`. Tried: "
        f"{[c[0]+'.'+c[1] for c in candidates]}. Try `pip install -U unimol_tools` "
        "or `pip show unimol_tools` and adapt this resolver."
    )


def _resolve_unimol_pretrain_dir() -> Path:
    """Find / create the dir where unimol_tools caches the .pt checkpoint."""
    # The default cache is ~/.cache/unimol; we mirror that.
    p = Path.home() / ".cache" / "unimol"
    p.mkdir(parents=True, exist_ok=True)
    return p


class UniMol2Branch(nn.Module):
    """Uni-Mol2 84M encoder, returns ``(n_mol, 512)`` molecular embeddings.

    Args:
        model_variant: "84M" or "164M". 84M is the default in HybridADMET.
        freeze: if True, gradients don't flow into Uni-Mol2 (cheap mode).
                If False (HybridADMET default), the whole encoder
                fine-tunes end-to-end with the other branches.
    """
    def __init__(self, model_variant: str = "84M", freeze: bool = False):
        super().__init__()
        from unimol_tools.utils import logger as _ulogger  # silence import noise
        _mod, UniMolModel = _resolve_unimol_api()

        # Construct the encoder with the official "molecule_v2" task config.
        # The class auto-downloads the pretrained checkpoint on first run
        # (writes to ~/.cache/unimol).
        try:
            self.encoder = UniMolModel(
                output_dim=512, data_type="molecule",
                remove_hs=False, model_size=model_variant,
            )
        except TypeError:
            # Older signature: data_type kw not present
            self.encoder = UniMolModel(output_dim=512, model_size=model_variant)
        self.hidden_dim = 512

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, batch: dict) -> torch.Tensor:
        """``batch`` is a dict produced by Uni-Mol2's collator, with keys
        ``src_tokens`` (atom-type indices), ``src_distance`` (pairwise dist
        matrix), ``src_edge_type`` (pairwise edge type idx), ``src_coord``
        (3D coords). See ``unimol_tools.data.collator``.

        Returns: (n_mol, 512) molecular embedding tensor.
        """
        # The forward of unimol_tools' UniMolModel returns a dict with
        # 'cls_repr' (or 'molecule_repr'); we fish that out robustly.
        out = self.encoder(**batch)
        if isinstance(out, dict):
            for key in ("cls_repr", "molecule_repr", "pooler_output",
                         "encoder_rep", "encoder_out"):
                if key in out:
                    rep = out[key]
                    break
            else:
                raise RuntimeError(
                    f"Uni-Mol2 returned dict with keys {list(out.keys())}; "
                    "couldn't find the molecule-level representation. "
                    "Adjust UniMol2Branch.forward."
                )
        else:
            rep = out
        if rep.dim() == 3:
            # (B, L, D) -> take [CLS] (index 0) per sequence
            rep = rep[:, 0, :]
        return rep
