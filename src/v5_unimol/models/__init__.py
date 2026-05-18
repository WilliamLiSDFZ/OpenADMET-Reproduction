"""v5_unimol.models: Uni-Mol2 branch (reused) + UnimolOnly main model."""
# Reuse v4's UniMol2Branch verbatim — same encoder, same checkpoint, same
# cls_repr extraction logic. No need to duplicate.
#
# NOTE: three dots, not two. This file's package is ``src.v5_unimol.models``,
# so ``..`` resolves to ``src.v5_unimol`` (not ``src``). We need ``...`` to
# climb up to ``src`` and reach sibling package ``v4_hybridadmet``.
from ...v4_hybridadmet.models.unimol_branch import UniMol2Branch
from .unimol_only import UnimolOnly, MultiTaskHead, masked_mae_loss

__all__ = ["UniMol2Branch", "UnimolOnly", "MultiTaskHead", "masked_mae_loss"]
