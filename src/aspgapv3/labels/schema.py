"""Declarative registry of every predicted quantity.

One `LabelSpec` per label: which ASPCAP column(s) it can live in, its error and
flag columns, the sigma floor used by the NLL, and the physical valid range
used both for cleaning and for masking. Column names are *candidates* resolved
against the real table at load time, so astra v0.6.0 -> DR21 is a table swap,
not a code change (`labels/loaders.py` stays untouched).

Verified against astraAllStarASPCAP-0.6.0.fits (HDU2, 1,095,480 rows):
    atmosphere   teff / e_teff, logg / e_logg, m_h_atm / e_m_h_atm,
                 alpha_m_atm / e_alpha_m_atm   (raw FERRE values carry raw_)
    elements     fe_h, mg_h, c_h, ... with e_<el>_h and <el>_h_flags
    calibration  teff is the IRFM-calibrated Teff (V band from Gaia XP
                 synthetic photometry => XP leakage); raw_teff is FERRE;
                 irfm_teff is the calibration target itself
    global       gaia_dr3_source_id, snr, flag_bad, ra, dec, g_mag

Leakage note: the canonical TRAINING Teff is therefore `raw_teff`. The
published (IRFM-calibrated) `teff` is kept for reporting and is explicitly
disclaimed as an independent validation quantity.

Tier policy (project definition):
  Tier 1: teff, logg, m_h, alpha_m, fe_h, mg_h  (mg_fe is derived, not fit)
  Tier 2: further ASPCAP individual abundances, added only after each is shown
          to carry XP-transferred information
  Tier 3: weak-information elements (Ce, Na, K, V, Nd, ... C/N ages)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelSpec:
    name: str
    tier: int
    group: str                    # "atm" or "chem"
    unit: str                     # "K" or "dex"
    column: tuple[str, ...]       # first existing candidate wins
    error_column: tuple[str, ...]
    flag_column: tuple[str, ...] = ()   # per-element quality bitmask; may be absent
    sigma_floor: float = 0.02     # raw units (K or dex)
    valid_range: tuple[float, float] = (-5.0, 1.0)
    description: str = ""


def _aux_element(name: str, floor: float,
                 valid_range: tuple[float, float] = (-4.0, 1.5)) -> LabelSpec:
    """Auxiliary element abundance following the astra naming scheme
    `<el>_h`, `e_<el>_h`, `<el>_h_flags` (verified for every <el> in HDU2)."""
    return LabelSpec(
        name=name, tier=3, group="aux", unit="dex",
        column=(name, name.upper()), error_column=(f"e_{name}", f"{name}_err"),
        flag_column=(f"{name}_flags",), sigma_floor=floor, valid_range=valid_range,
    )


LABELS: tuple[LabelSpec, ...] = (
    LabelSpec(
        name="teff", tier=1, group="atm", unit="K",
        column=("raw_teff", "teff_raw", "TEFF"),
        error_column=("raw_e_teff", "e_teff", "TEFF_ERR"),
        sigma_floor=50.0, valid_range=(2800.0, 8000.0),
        description="raw FERRE Teff; auxiliary target (avoids the IRFM/Gaia-XP "
                    "photometry leak baked into the calibrated teff)",
    ),
    LabelSpec(
        name="teff_calibrated", tier=1, group="atm", unit="K",
        column=("teff",),
        error_column=("e_teff",),
        sigma_floor=50.0, valid_range=(2800.0, 8000.0),
        description="published IRFM-calibrated Teff; the calibration's V band "
                    "comes from Gaia XP synthetic photometry, so this is NOT "
                    "an independent validation quantity",
    ),
    LabelSpec(
        name="logg", tier=1, group="atm", unit="dex",
        column=("logg", "LOGG"),
        error_column=("e_logg", "logg_err", "LOGG_ERR"),
        flag_column=("logg_flags", "logg_flag"),
        sigma_floor=0.05, valid_range=(-0.5, 5.5),
        description="asteroseismology-calibrated logg (no XP leak)",
    ),
    LabelSpec(
        name="m_h", tier=1, group="atm", unit="dex",
        column=("m_h_atm", "m_h", "MH"),
        error_column=("e_m_h_atm", "m_h_err"),
        flag_column=("m_h_atm_flags", "m_h_flags"),
        sigma_floor=0.02, valid_range=(-5.0, 1.0),
        description="[M/H] from the atmospheric (stellar-parameter) fit",
    ),
    LabelSpec(
        name="alpha_m", tier=1, group="chem", unit="dex",
        column=("alpha_m_atm", "alpha_m", "ALPHA_M"),
        error_column=("e_alpha_m_atm", "alpha_m_err"),
        flag_column=("alpha_m_atm_flags", "alpha_m_flags", "alpha_m_flag"),
        sigma_floor=0.02, valid_range=(-0.5, 1.0),
        description="[alpha/M]; the headline science label",
    ),
    LabelSpec(
        name="fe_h", tier=1, group="chem", unit="dex",
        column=("fe_h", "FE_H"),
        error_column=("e_fe_h", "fe_h_err", "FE_H_ERR"),
        flag_column=("fe_h_flags",),
        sigma_floor=0.02, valid_range=(-5.0, 1.0),
    ),
    LabelSpec(
        name="mg_h", tier=1, group="chem", unit="dex",
        column=("mg_h", "MG_H"),
        error_column=("e_mg_h", "mg_h_err", "MG_H_ERR"),
        flag_column=("mg_h_flags",),
        sigma_floor=0.02, valid_range=(-3.0, 1.5),
    ),
    # ---- Tier 2 (added to configs only after each earns its place) ----
    LabelSpec(
        name="c_h", tier=2, group="chem", unit="dex",
        column=("c_h", "C_H"), error_column=("e_c_h", "c_h_err"),
        flag_column=("c_h_flags",),
        sigma_floor=0.03, valid_range=(-4.0, 1.5),
    ),
    LabelSpec(
        name="n_h", tier=2, group="chem", unit="dex",
        column=("n_h", "N_H"), error_column=("e_n_h", "n_h_err"),
        flag_column=("n_h_flags",),
        sigma_floor=0.05, valid_range=(-4.0, 1.5),
    ),
    LabelSpec(
        name="o_h", tier=2, group="chem", unit="dex",
        column=("o_h", "O_H"), error_column=("e_o_h", "o_h_err"),
        flag_column=("o_h_flags",),
        sigma_floor=0.03, valid_range=(-4.0, 1.5),
    ),
    LabelSpec(
        name="si_h", tier=2, group="chem", unit="dex",
        column=("si_h", "SI_H"), error_column=("e_si_h", "si_h_err"),
        flag_column=("si_h_flags",),
        sigma_floor=0.02, valid_range=(-4.0, 1.5),
    ),
    LabelSpec(
        name="ca_h", tier=2, group="chem", unit="dex",
        column=("ca_h", "CA_H"), error_column=("e_ca_h", "ca_h_err"),
        flag_column=("ca_h_flags",),
        sigma_floor=0.02, valid_range=(-4.0, 1.5),
    ),
    LabelSpec(
        name="ni_h", tier=2, group="chem", unit="dex",
        column=("ni_h", "NI_H"), error_column=("e_ni_h", "ni_h_err"),
        flag_column=("ni_h_flags",),
        sigma_floor=0.02, valid_range=(-4.0, 1.5),
    ),
    LabelSpec(
        name="al_h", tier=2, group="chem", unit="dex",
        column=("al_h", "AL_H"), error_column=("e_al_h", "al_h_err"),
        flag_column=("al_h_flags",),
        sigma_floor=0.03, valid_range=(-4.0, 1.5),
    ),
    LabelSpec(
        name="mn_h", tier=2, group="chem", unit="dex",
        column=("mn_h", "MN_H"), error_column=("e_mn_h", "mn_h_err"),
        flag_column=("mn_h_flags",),
        sigma_floor=0.03, valid_range=(-4.0, 1.5),
    ),
    # ---- Tier 3 / auxiliary-only elements (P1 latent bottleneck) -----------
    # Never science outputs: trained only through the auxiliary head
    # (`labels.aux`, weight `labels.weight_aux`). Floors are larger for weak /
    # blended lines where formal (repeat-visit) errors are unrealistically small.
    *(_aux_element(*a) for a in (
        # name   floor  (valid range defaults to (-4, 1.5))
        ("s_h", 0.05), ("k_h", 0.06), ("na_h", 0.08), ("ti_h", 0.05),
        ("ti_2_h", 0.08), ("v_h", 0.10), ("cr_h", 0.06), ("co_h", 0.10),
        ("cu_h", 0.12), ("ce_h", 0.15), ("nd_h", 0.15), ("p_h", 0.15),
        ("c_1_h", 0.05),
    )),
)

_BY_NAME: dict[str, LabelSpec] = {s.name: s for s in LABELS}

TIER1: tuple[str, ...] = tuple(s.name for s in LABELS if s.tier == 1)


# Elements usable as auxiliary targets: every registered abundance that is not
# already a Tier-1 science label (fe_h / mg_h are trained in the chem head).
AUX_CANDIDATES: tuple[str, ...] = tuple(
    s.name for s in LABELS if s.tier >= 2 and s.group in ("chem", "aux"))


def spec(name: str) -> LabelSpec:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown label {name!r}; known: {sorted(_BY_NAME)}. "
            "Add new elements to LABELS with tier evidence, not wishful thinking."
        ) from None


def specs(names) -> list[LabelSpec]:
    return [spec(n) for n in names]


# Global (non-per-label) columns, resolved at load time. Verified names first.
GLOBAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "source_id": ("gaia_dr3_source_id", "source_id"),
    "snr": ("snr", "SNR"),
    "flag_bad": ("flag_bad",),
    "teff_for_range": ("raw_teff", "teff"),     # Teff<8000K domain cut (raw scale)
    "teff_calibrated": ("teff",),               # reported alongside, disclaimed
    "irfm_teff": ("irfm_teff",),
    "irfm_teff_flags": ("irfm_teff_flags",),
    "ra": ("ra",),
    "dec": ("dec",),
    "g_mag": ("g_mag", "phot_g_mean_mag"),
    "apogee_id": ("sdss4_apogee_id", "apogee_id"),
    "sdss_id": ("sdss_id",),
    "healpix": ("healpix",),
}

# astraAllStarASPCAP <el>_h_flags bit definitions (DR20 = DR19 APOGEE):
#   bits 0-4  UPPER_LIMIT_T1..T5   (some line an upper limit; value usable)
#   bit  5    CENSORED_HIGH_TEFF
#   bit  6    CENSORED_LOW_TEFF_VMICRO
#   bit  7    CENSORED_UNPHYSICAL
#   bit  8    BAD_GRID_EDGE
#   bit  9    WARN_GRID_EDGE
#   bit 10    WARN_TEFF
#   bit 11    WARN_M_H
FLAG_BITS = {
    "upper_limit": (0, 1, 2, 3, 4),   # recorded, not masked by default
    "hard": (5, 6, 7, 8),             # always masked out
    "warn": (9, 10, 11),              # masked when mask_warn_bits=True
}
