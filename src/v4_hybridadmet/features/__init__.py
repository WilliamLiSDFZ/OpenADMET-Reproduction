"""Feature computation: MACCS keys + ErG fingerprint (PubChem TODO)."""
from .fingerprints import compute_fingerprints_batch, FP_DIM
__all__ = ["compute_fingerprints_batch", "FP_DIM"]
