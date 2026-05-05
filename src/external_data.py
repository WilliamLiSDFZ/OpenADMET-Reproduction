"""Load and unit-align external ADMET datasets for augmentation.

External sources (all from public GitHub repos, no HuggingFace needed):

  1. biogen_logS.csv   (Pat Walters, ~2173 measured kinetic solubility)
                       -> augments KSOL  (assumed units: log10(μM))
  2. esol_logSolubility.csv  (~500 ESOL/Delaney measured logS in mol/L)
                       -> augments KSOL  (units: log10(M), so convert to log10(μM))
  3. drugbank_admet_predictions.csv  (admet_ai PREDICTED, ~2845 drugs)
                       -> augments LogD, KSOL, Caco-2 Papp, MPPB, HLM CLint, MLM CLint
                       (lower weight because these are predictions, not measurements)

Returns rows in the same "log space" the LightGBM model trains on, so
they can be concatenated to the log-transformed training set directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

import os as _os
# Per-endpoint sample-weight schedule.
#   measured-data weight : default 0.3 (slightly less confident than challenge)
#   predicted-data weight: default 0.05 (much less; just regularization signal)
W_MEASURED = float(_os.environ.get("W_MEASURED", 0.3))
W_PREDICTED = float(_os.environ.get("W_PREDICTED", 0.05))


def _df_with_weight(smiles: pd.Series, label: pd.Series, weight: float,
                    source: str) -> pd.DataFrame:
    """Convenience: build the augmentation row format expected downstream."""
    out = pd.DataFrame({
        "SMILES": smiles.astype(str).str.strip(),
        "label": label.astype(float),
        "weight": float(weight),
        "source": source,
    })
    out = out.dropna(subset=["SMILES", "label"]).reset_index(drop=True)
    out = out[out["SMILES"].str.len() > 0]
    return out


# ============================================================
# Per-endpoint augmentation tables
# ============================================================
def _aug_LogS_biogen() -> pd.DataFrame:
    """biogen_logS appears to be log10(KSOL in μM).

    Challenge's ``Log_KSOL`` (short_name 'LogS') = log10((KSOL_μM + 1) * 1e-6).
    Convert with:    log_chall = log10(10^biogen + 1) - 6
    """
    f = EXT_DIR / "biogen_logS.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    converted = np.log10(np.power(10.0, df["logS"].astype(float)) + 1.0) - 6.0
    return _df_with_weight(df["SMILES"], converted, W_MEASURED, "biogen_logS")


def _aug_LogS_esol() -> pd.DataFrame:
    """ESOL/Delaney logSolubility is in log10(mol/L).

    Convert: KSOL_μM = 10^logS * 1e6,
             log_chall = log10(10^logS * 1e6 + 1) - 6
    """
    f = EXT_DIR / "esol_logSolubility.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    ksol_uM = np.power(10.0, df["logSolubility"].astype(float)) * 1e6
    converted = np.log10(ksol_uM + 1.0) - 6.0
    return _df_with_weight(df["smiles"], converted, W_MEASURED, "esol_logS")


def _aug_LogS_drugbank() -> pd.DataFrame:
    """admet_ai's Solubility_AqSolDB prediction is log10(mol/L)."""
    f = EXT_DIR / "drugbank_admet_predictions.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if "Solubility_AqSolDB" not in df.columns:
        return pd.DataFrame()
    ksol_uM = np.power(10.0, df["Solubility_AqSolDB"].astype(float)) * 1e6
    converted = np.log10(ksol_uM + 1.0) - 6.0
    return _df_with_weight(df["smiles"], converted, W_PREDICTED, "drugbank_solubility")


def _aug_LogD_drugbank() -> pd.DataFrame:
    """admet_ai's Lipophilicity_AstraZeneca is LogD7.4 -- the same scale we want."""
    f = EXT_DIR / "drugbank_admet_predictions.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if "Lipophilicity_AstraZeneca" not in df.columns:
        return pd.DataFrame()
    return _df_with_weight(df["smiles"], df["Lipophilicity_AstraZeneca"],
                           W_PREDICTED, "drugbank_lipo")


def _aug_Caco_Papp_drugbank() -> pd.DataFrame:
    """admet_ai's Caco2_Wang is log10(cm/s); challenge wants log10((Papp_um + 1)*1e-6)
    where Papp_um is in 1e-6 cm/s.

    Papp_cm_s = 10^Caco2_Wang
    Papp_um   = Papp_cm_s * 1e6 = 10^(Caco2_Wang + 6)
    log_chall = log10(Papp_um + 1) - 6 = log10(10^(Caco2_Wang+6) + 1) - 6

    For typical magnitudes (Caco2_Wang ~ -5 → Papp_um ~ 10), this simplifies to
    Caco2_Wang itself once 10^(Caco2_Wang+6) >> 1, so it's basically a passthrough.
    """
    f = EXT_DIR / "drugbank_admet_predictions.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if "Caco2_Wang" not in df.columns:
        return pd.DataFrame()
    papp_um = np.power(10.0, df["Caco2_Wang"].astype(float) + 6.0)
    converted = np.log10(papp_um + 1.0) - 6.0
    return _df_with_weight(df["smiles"], converted, W_PREDICTED, "drugbank_caco")


def _aug_MPPB_drugbank() -> pd.DataFrame:
    """admet_ai's PPBR_AZ is in % bound to plasma protein.

    Challenge's MPPB column (in train.csv) is also % (typically interpreted as
    fraction unbound × 100). Both AZ and challenge values cluster 0-100, so we
    pass the value through the same log10(x+1) transform used for MPPB.

    NB: PPBR_AZ in admet_ai can be slightly outside [0,100] (-125 to 127) because
    it's a regression prediction. We clip to [0, 100] before transforming.
    """
    f = EXT_DIR / "drugbank_admet_predictions.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if "PPBR_AZ" not in df.columns:
        return pd.DataFrame()
    val = df["PPBR_AZ"].astype(float).clip(0, 100)
    converted = np.log10(val + 1.0)
    return _df_with_weight(df["smiles"], converted, W_PREDICTED, "drugbank_ppbr")


def _aug_HLM_CLint_drugbank() -> pd.DataFrame:
    """admet_ai's Clearance_Microsome_AZ is in μL/min/mg (matches HLM/MLM units)."""
    f = EXT_DIR / "drugbank_admet_predictions.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if "Clearance_Microsome_AZ" not in df.columns:
        return pd.DataFrame()
    val = df["Clearance_Microsome_AZ"].astype(float).clip(0, None)
    converted = np.log10(val + 1.0)
    return _df_with_weight(df["smiles"], converted, W_PREDICTED, "drugbank_clint")


# ============================================================
# Public API
# ============================================================
# Endpoint short-name -> list of loader callables.
# Toggle with env var EXT_PROFILE: "all" | "selective" | "none"
_PROFILE = _os.environ.get("EXT_PROFILE", "all").lower()

if _PROFILE == "none":
    AUGMENTATION_LOADERS: Dict[str, List] = {ep: [] for ep in [
        "LogD", "LogS", "Log_HLM_CLint", "Log_MLM_CLint",
        "Log_Caco_Papp_AB", "Log_Caco_ER",
        "Log_Mouse_PPB", "Log_Mouse_BPB", "Log_Mouse_MPB",
    ]}
elif _PROFILE == "selective":
    # First run (W=0.5/0.2) showed external HELPED these endpoints (ΔRAE < 0):
    #   LogD (-0.015), MLM CLint (-0.009), MPPB (-0.051)
    # External HURT these endpoints (ΔRAE > 0), so we skip them:
    #   KSOL (+0.020), HLM CLint (+0.016), Caco-2 Papp (+0.048)
    AUGMENTATION_LOADERS = {
        "LogD":             [_aug_LogD_drugbank],
        "LogS":             [],
        "Log_HLM_CLint":    [],
        "Log_MLM_CLint":    [_aug_HLM_CLint_drugbank],
        "Log_Caco_Papp_AB": [],
        "Log_Caco_ER":      [],
        "Log_Mouse_PPB":    [_aug_MPPB_drugbank],
        "Log_Mouse_BPB":    [],
        "Log_Mouse_MPB":    [],
    }
else:  # "all"
    AUGMENTATION_LOADERS = {
        "LogD":             [_aug_LogD_drugbank],
        "LogS":             [_aug_LogS_biogen, _aug_LogS_esol, _aug_LogS_drugbank],
        "Log_HLM_CLint":    [_aug_HLM_CLint_drugbank],
        "Log_MLM_CLint":    [_aug_HLM_CLint_drugbank],
        "Log_Caco_Papp_AB": [_aug_Caco_Papp_drugbank],
        "Log_Caco_ER":      [],
        "Log_Mouse_PPB":    [_aug_MPPB_drugbank],
        "Log_Mouse_BPB":    [],
        "Log_Mouse_MPB":    [],
    }


def load_augmentation(short_name: str) -> pd.DataFrame:
    """Return concatenated external rows for one challenge endpoint.

    Columns: SMILES, label (already in challenge log-space), weight, source.
    Returns an empty DataFrame if no external data is configured.
    """
    pieces = []
    for loader in AUGMENTATION_LOADERS.get(short_name, []):
        try:
            df = loader()
            if not df.empty:
                pieces.append(df)
        except Exception as e:
            print(f"  WARN: {loader.__name__} failed: {e}")
    if not pieces:
        return pd.DataFrame(columns=["SMILES", "label", "weight", "source"])
    out = pd.concat(pieces, ignore_index=True)
    # Drop SMILES that show up multiple times (keep first/measured)
    out = out.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)
    return out


def summary() -> pd.DataFrame:
    """Quick "what augmentation do we have for each endpoint" report."""
    rows = []
    for ep in AUGMENTATION_LOADERS:
        df = load_augmentation(ep)
        rows.append({
            "endpoint": ep,
            "n_external_total": len(df),
            "sources": ", ".join(sorted(df["source"].unique())) if len(df) else "—",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(summary().to_string(index=False))
