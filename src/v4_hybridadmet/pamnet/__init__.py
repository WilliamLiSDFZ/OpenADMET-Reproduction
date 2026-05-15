"""PAMNet, ported verbatim from XieResearchGroup/Physics-aware-Multiplex-GNN.

Files mirror the upstream layout 1-to-1 (just renamed to plain modules so
the package imports cleanly):

  utils/sbf.py                 -> sbf_utils.py
  layers/basic.py              -> basic_layers.py
  layers/global_message_passing.py -> global_mp.py
  layers/local_message_passing.py  -> local_mp.py
  models.py                    -> pamnet_model.py
"""
from .basic_layers import (BesselBasisLayer, Envelope, MLP, Res,
                           SiLU, SphericalBasisLayer)
from .global_mp import Global_MessagePassing
from .local_mp import Local_MessagePassing, Local_MessagePassing_s
from .pamnet_model import PAMNet, PAMNet_s, Config

__all__ = [
    "PAMNet", "PAMNet_s", "Config",
    "BesselBasisLayer", "Envelope", "MLP", "Res", "SiLU", "SphericalBasisLayer",
    "Global_MessagePassing", "Local_MessagePassing", "Local_MessagePassing_s",
]
