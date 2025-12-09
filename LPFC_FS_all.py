# === Cell 1: Imports for Stability Selection (LOSO, nested, classification) ===
import os, warnings, math
from collections import defaultdict, Counter
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from joblib import Memory

# Scikit-learn core
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,classification_report, 
                            confusion_matrix,  roc_curve, log_loss)
from sklearn.utils import check_random_state
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, StratifiedShuffleSplit, ParameterGrid, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.base import clone

# Optional: multiple-comparison control for biomarker tables
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Parallel utils (optional, for subsampling loops)
from joblib import Parallel, delayed



import json, hashlib
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from scipy.io import loadmat

# ---- Threads cap to avoid hidden oversubscription (can adjust or remove) ----
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "6")
os.environ.setdefault("OMP_NUM_THREADS", "6")

warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
rng = check_random_state(SEED)


# --- Cell 2a: Load XX (N × n_trl × n_feat) and labels from MATLAB exports ---

import os, glob, re
import numpy as np
import pandas as pd
from scipy.io import loadmat

# mat_root = "/Users/sarakamali/Desktop/dementia/Dementia analysis/"
# out_dir  = "./stability_results/LPFC_all"

mat_root = "/home/saraka/simulations/dementia"
out_dir  = "./LPFC_all"

os.makedirs(out_dir, exist_ok=True)

mat_files = sorted(glob.glob(os.path.join(mat_root, "tensor_pure_feats_*.mat")))
if not mat_files:
    raise FileNotFoundError(f"No MAT files found in {mat_root} matching 'tensor_pure_feats_*.mat'")

data_by_cluster = {}
for mf in mat_files:
    dd = loadmat(mf, squeeze_me=True, struct_as_record=False)

    base = os.path.splitext(os.path.basename(mf))[0]
    cluster_name = re.sub(r'^tensor_pure_feats_', '', base, flags=re.IGNORECASE)

    # Core tensor
    XX = np.asarray(dd["XX"], dtype=float)  # (N, n_trl, n_feat)

    subject_ids = np.asarray(dd.get("subject_id",  np.nan), dtype=float)

    # Labels
    yCD = np.asarray(dd.get("yCD", np.nan), dtype=float)   # Control vs dementia (binary)
    yAF = np.asarray(dd.get("yAF", np.nan), dtype=float)   # AD vs FTD (binary)
    yCE = np.asarray(dd.get("yCE", np.nan), dtype=float)   # hard 0/1 with NaNs
    yEL = np.asarray(dd.get("yEL", np.nan), dtype=float)   # hard 0/1 with NaNs
    mmse = np.asarray(dd.get("ymmse", np.nan), dtype=float)

    # Feature names
    if "feature_names" in dd:
        raw = dd["feature_names"]
        if isinstance(raw, (list, tuple)):
            feat_names = [str(x).strip() for x in raw]
        else:
            arr = np.asarray(raw)
            if arr.dtype.kind in ("U", "S", "O"):
                feat_names = [str(x).strip() for x in arr.ravel().tolist()]
            elif arr.dtype == np.uint8 and arr.ndim == 2:  # char matrix from MATLAB
                feat_names = [
                    "".join(chr(arr[r, c]) for r in range(arr.shape[0])).strip()
                    for c in range(arr.shape[1])
                ]
            else:
                feat_names = [f"feat_{i+1}" for i in range(XX.shape[2])]
    else:
        feat_names = [f"feat_{i+1}" for i in range(XX.shape[2])]

    data_by_cluster[cluster_name] = {
        "XX": XX,
        "ybin": yCD,            # Control vs dementia
        "yAF": yAF,
        "yCE": yCE,
        "yEL": yEL,
        "mask_CD": ~np.isnan(yCD),
        "mask_AF": ~np.isnan(yAF),
        "mask_CE": ~np.isnan(yCE),
        "mask_EL": ~np.isnan(yEL),
        "y_mmse": mmse,
        "feature_names": feat_names,
        "file": mf,
        "subject_ids": subject_ids,
    }

AVAILABLE = list(data_by_cluster.keys())
print("Available clusters:", AVAILABLE)

CLUSTER = 'LPFC' 
D = data_by_cluster[CLUSTER]

# --- Assign variables for CNN/LSTM step ---
X_all = D["XX"]            # (N, n_trials, n_feat)
y_CD  = D["ybin"]          # (N,)
y_AF  = D["yAF"]           # (N,)
y_CE  = D["yCE"]           # (N,) hard with NaN masked
y_EL  = D["yEL"]           # (N,) hard with NaN masked
mask_CD = D["mask_CD"]
mask_AF = D["mask_AF"]
mask_CE = D["mask_CE"]
mask_EL = D["mask_EL"]
subject_ids = D["subject_ids"]

print(f"[{CLUSTER}] X_all shape: {X_all.shape} (N, n_trials, n_feat)")
print(f"[{CLUSTER}] y_CD counts:", pd.Series(y_CD).value_counts(dropna=False).to_dict())
print(f"[{CLUSTER}] y_AF counts:", pd.Series(y_AF).value_counts(dropna=False).to_dict())
print(f"[{CLUSTER}] y_CE counts:", pd.Series(y_CE).value_counts(dropna=False).to_dict())
print(f"[{CLUSTER}] y_EL counts:", pd.Series(y_EL).value_counts(dropna=False).to_dict())

feature_names = D.get("feature_names", [f"feat_{i+1}" for i in range(X_all.shape[2])])


# In[75]:


# === Cell 2b: Cohort summary (from MAT fields in dd) — subject_id, age, gender ===

# Re-open the MAT for the selected cluster to access group/age/gender
mf = D["file"]  # path saved in Cell 2 for the chosen CLUSTER
dd = loadmat(mf, squeeze_me=True, struct_as_record=False)

# Extract exactly as specified (from dd, not from D)
group  = np.asarray(dd.get("group_str", np.nan)).reshape(-1).astype(object)
age    = np.asarray(dd.get("age", np.nan), dtype=float).reshape(-1)
gender = np.asarray(dd.get("gender_raw", np.nan)).reshape(-1).astype(object)

# Basic integrity checks against X_all
N = X_all.shape[0]
if group.size != N or age.size != N or gender.size != N:
    raise ValueError(
        f"Length mismatch: group({group.size}), age({age.size}), gender({gender.size}) vs N={N}"
    )

# Build tidy DataFrame
sid = np.asarray(subject_ids).reshape(-1).astype(object)
meta = pd.DataFrame({"subject_id": sid, "group": group, "age": age, "gender": gender})

# Drop rows with unknown group labels (if any)
meta_clean = meta.dropna(subset=["group"]).copy()

# --- Group percentages ---
grp_counts = meta_clean["group"].value_counts()
grp_pct = 100.0 * grp_counts / grp_counts.sum()

print(f"[{CLUSTER}] Subject counts by group:\n{grp_counts.to_string()}")
print(f"\n[{CLUSTER}] Subject percentages by group (%):\n{grp_pct.round(2).to_string()}")

# Save CSV
pct_table = pd.DataFrame({"count": grp_counts, "percent": grp_pct.round(2)})
pct_table.to_csv(os.path.join(out_dir, f"{CLUSTER}_group_percentages.csv"))

# --- Age summary by group ---
age_summary = (
    meta_clean.groupby("group")["age"]
    .agg(
        n="count",
        mean="mean",
        std="std",
        median="median",
        q25=lambda s: np.nanpercentile(s, 25),
        q75=lambda s: np.nanpercentile(s, 75),
    )
    .round(2)
)
print(f"\n[{CLUSTER}] Age summary by group:\n{age_summary.to_string()}")
age_summary.to_csv(os.path.join(out_dir, f"{CLUSTER}_age_summary.csv"))

# --- Gender counts per group (expects 'F'/'M') ---
gender_ct = (
    meta_clean.groupby(["group", "gender"])
    .size()
    .unstack(fill_value=0)
    .astype(int)
)
print(f"\n[{CLUSTER}] Gender composition (counts):\n{gender_ct.to_string()}")
gender_ct.to_csv(os.path.join(out_dir, f"{CLUSTER}_gender_counts.csv"))

# ================== PLOTS ==================
# Use the same group order as counts for consistent labeling
group_order = grp_counts.index.tolist()

# 1) Group percentages (bar)
fig1, ax1 = plt.subplots(figsize=(5, 4), dpi=130)
bars = ax1.bar(group_order, grp_pct[group_order].values)
ax1.set_ylabel("Percentage of subjects (%)")
ax1.set_title(f"{CLUSTER}: Subject distribution by group (N={grp_counts.sum()})")
for b, v in zip(bars, grp_pct[group_order].values):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
fig1.tight_layout()
fig1.savefig(os.path.join(out_dir, f"{CLUSTER}_group_percentages.png"), bbox_inches="tight")
plt.close(fig1)

# 2) Age by group (boxplot)
age_data = [meta_clean.loc[meta_clean["group"] == g, "age"].dropna().to_numpy() for g in group_order]
fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=130)
ax2.boxplot(age_data, labels=group_order, showfliers=False)
ax2.set_ylabel("Age (years)")
ax2.set_title(f"{CLUSTER}: Age distribution by group")
fig2.tight_layout()
fig2.savefig(os.path.join(out_dir, f"{CLUSTER}_age_boxplot.png"), bbox_inches="tight")
plt.close(fig2)

# 3) Gender composition per group (stacked bar)
fig3, ax3 = plt.subplots(figsize=(6, 4), dpi=130)
bottom = np.zeros(len(gender_ct.index))
for col in gender_ct.columns:  # typically 'F', 'M'
    vals = gender_ct[col].reindex(group_order).to_numpy()
    ax3.bar(group_order, vals, bottom=bottom, label=col)
    bottom += vals
ax3.set_ylabel("Count")
ax3.set_title(f"{CLUSTER}: Gender composition by group")
ax3.legend(title="Gender", bbox_to_anchor=(1.02, 1), loc="upper left")
fig3.tight_layout()
fig3.savefig(os.path.join(out_dir, f"{CLUSTER}_gender_stacked_bar.png"), bbox_inches="tight")
plt.close(fig3)

print("Saved figures and CSVs to:", out_dir)


# In[76]:


# === Cell 2c: Create & append ratio features by name ===

RATIOS_TO_ADD = [
    # ('powermean_Delta','powermean_Alpha'),
    # ('powermean_Theta', 'powermean_Low Gamma'),
    ('powermean_Theta', 'powermean_Alpha'),
]


names = np.array(feature_names, dtype=object)

# Case-insensitive exact name lookup
name2idx = {n.lower(): i for i, n in enumerate(names)}

added = []
skipped = []

for A_name, B_name in RATIOS_TO_ADD:
    A_key = A_name.strip().lower()
    B_key = B_name.strip().lower()

    if A_key not in name2idx or B_key not in name2idx:
        missing = []
        if A_key not in name2idx: missing.append(A_name)
        if B_key not in name2idx: missing.append(B_name)
        skipped.append((A_name, B_name, f"Missing: {', '.join(missing)}"))
        continue

    iA = name2idx[A_key]
    iB = name2idx[B_key]

    A = X_all[:, :, iA]   # (N, n_trl)
    B = X_all[:, :, iB]   # (N, n_trl)

    # Safe ratio: NaN where denominator is (near) zero or NaN
    denom_ok = (~np.isnan(B)) & (~np.isclose(B, 0.0))
    ratio = np.full_like(A, np.nan)
    ratio[denom_ok] = A[denom_ok] / B[denom_ok]

    # Append as a new feature (axis=2)
    X_all = np.concatenate([X_all, ratio[:, :, None]], axis=2)

    new_name = f"{A_name}/{B_name}"
    feature_names.append(new_name)
    added.append(new_name)

# ---- Report ----
if added:
    print(f"Added {len(added)} ratio feature(s): {added}")
    print(f"New X_all shape: {X_all.shape} | Features now: {len(feature_names)}")
else:
    print("No ratio features added.")

if skipped:
    for A_name, B_name, reason in skipped:
        print(f"Skipped '{A_name}/{B_name}' → {reason}")

# Optional: keep mapping of original→current indices after appending
KEPT_FEATURE_IDX = np.where(np.isin(feature_names_orig, feature_names))[0].tolist() if 'feature_names_orig' in globals() else None


# In[77]:


# === Cell 2d: Drop or reset features by name ===
# Example usage:
#   DROP_FEATURES = ["HD(A)/S(A)", "SampEn(HG)/SampEn(LG)"]
#   RESET_FEATURES = True   # to undo and go back to original

# 81-82

DROP_FEATURES= [
                # 'M_Theta','M_Alpha','M_Beta','M_Low Gamma','M_High Gamma',
                # 'HDA_Theta','HDA_Alpha','HDA_Beta','HDA_Low Gamma','HDA_High Gamma',
                # 'HA_Theta','HA_Alpha','HA_Beta','HA_Low Gamma','HA_High Gamma',
                # 'HD_Theta','HD_Alpha','HD_Beta','HD_Low Gamma','HD_High Gamma',
                # 'sampen_Theta','sampen_Alpha','sampen_Beta', 'sampen_Low Gamma','sampen_High Gamma',
                # 'KFD_Theta','KFD_Alpha','KFD_Beta','KFD_Low Gamma','KFD_High Gamma',
                'powermean_Delta','powermean_Theta', 'powermean_Alpha','powermean_Beta','powermean_Low Gamma','powermean_High Gamma',
                  # 'signalmean_Delta',
                # 'signalmean_Theta','signalmean_Alpha','signalmean_Beta','signalmean_Low Gamma','signalmean_High Gamma',
                # 'HFD_Theta','HFD_Alpha','HFD_Beta','HFD_Low Gamma','HFD_High Gamma',
                # 'Hurst_Theta','Hurst_Alpha','Hurst_Beta','Hurst_Low Gamma','Hurst_High Gamma',
                # 'MI_Tort_alpha×lowgamma','MI_Tort_alpha×highgamma','PrefPhase_alpha×lowgamma','PrefPhase_alpha×highgamma'
]



RESET_FEATURES = False  # <-- set True to restore original feature set

# ---- Preserve originals once (for easy reset) ----
if 'X_all_orig' not in globals():
    X_all_orig = X_all.copy()
    feature_names_orig = feature_names.copy()

if RESET_FEATURES:
    X_all = X_all_orig.copy()
    feature_names = feature_names_orig.copy()
    print(f"Reset done. X_all shape: {X_all.shape} | Features: {len(feature_names)}")

else:
    # Options
    CASE_INSENSITIVE = True   # match names ignoring case
    ALLOW_SUBSTRING  = False  # if True, treat entries in DROP_FEATURES as substrings

    names = np.array(feature_names, dtype=object)

    def _match_mask(names, patterns, ci=True, substr=False):
        if not patterns:
            return np.zeros(len(names), dtype=bool)
        mask = np.zeros(len(names), dtype=bool)
        for p in patterns:
            if ci:
                p_cmp = p.lower()
                if substr:
                    mask |= np.array([p_cmp in n.lower() for n in names], dtype=bool)
                else:
                    mask |= np.array([n.lower() == p_cmp for n in names], dtype=bool)
            else:
                if substr:
                    mask |= np.array([p in n for n in names], dtype=bool)
                else:
                    mask |= (names == p)
        return mask

    drop_mask = _match_mask(names, DROP_FEATURES, ci=CASE_INSENSITIVE, substr=ALLOW_SUBSTRING)
    keep_mask = ~drop_mask

    dropped_idx = np.where(drop_mask)[0].tolist()
    kept_idx    = np.where(keep_mask)[0].tolist()
    dropped_names = names[dropped_idx].tolist()

    # ---- Apply selection along feature axis (axis=2 of X_all: N x n_trl x n_feat) ----
    if len(dropped_idx) > 0:
        X_all = X_all[:, :, kept_idx]
        feature_names = names[keep_mask].tolist()
        print(f"Removed {len(dropped_idx)} feature(s): {dropped_names}")
        print(f"New X_all shape: {X_all.shape} | Features left: {len(feature_names)}")
    else:
        print("No features matched the drop list. X_all unchanged.")
        print(f"X_all shape: {X_all.shape} | Features: {len(feature_names)}")

# For downstream bookkeeping (optional):
KEPT_FEATURE_IDX = np.where(np.isin(feature_names_orig, feature_names))[0].tolist()
DROPPED_FEATURE_IDX = np.where(~np.isin(feature_names_orig, feature_names))[0].tolist()
DROPPED_FEATURE_NAMES = [feature_names_orig[i] for i in DROPPED_FEATURE_IDX]


# In[79]:


# === Cell 3: Prepare subject-level matrix for LOSO + Stability Selection ===


# ---- 3.1 Choose target (you can switch to y_AF / y_CE / y_EL as needed) ----
TARGET = "y_CD"   # options: "y_CD", "y_AF", "y_CE", "y_EL"
if TARGET == "y_CD":
    y_raw  = y_CD
    mask_y = mask_CD
elif TARGET == "y_AF":
    y_raw  = y_AF
    mask_y = mask_AF
elif TARGET == "y_CE":
    y_raw  = y_CE
    mask_y = mask_CE
elif TARGET == "y_EL":
    y_raw  = y_EL
    mask_y = mask_EL
else:
    raise ValueError("Unknown TARGET")

# ---- 3.2 Aggregate trials → subject-level features (robust) ----
def aggregate_trials(X, aggs=("median","iqr","cv"), eps=1e-8, feature_names=None):
    """
    X: (N_subjects, n_trials, n_features)
    Returns: X_agg (N_subjects, n_features * len(aggs)), aggregated_names
    """
    if X.ndim != 3:
        raise ValueError("X must be 3D: (subjects, trials, features)")
    N, T, F = X.shape
    if feature_names is None:
        feature_names = [f"feat_{i+1}" for i in range(F)]
    feature_names = list(feature_names)

    outs = []
    out_names = []

    X_mean = np.nanmean(X, axis=1)      # for CV
    X_std  = np.nanstd(X, axis=1)

    for agg in aggs:
        if agg == "median":
            val = np.nanmedian(X, axis=1)                 # (N, F)
            outs.append(val)
            out_names += [f"{n}|med" for n in feature_names]
        elif agg == "iqr":
            q75 = np.nanpercentile(X, 75, axis=1)
            q25 = np.nanpercentile(X, 25, axis=1)
            val = q75 - q25
            outs.append(val)
            out_names += [f"{n}|iqr" for n in feature_names]
        elif agg == "cv":
            # coefficient of variation (std / |mean|)
            val = X_std / (np.abs(X_mean) + eps)
            outs.append(val)
            out_names += [f"{n}|cv" for n in feature_names]
        elif agg == "mean":
            val = X_mean
            outs.append(val)
            out_names += [f"{n}|mean" for n in feature_names]
        elif agg == "std":
            val = X_std
            outs.append(val)
            out_names += [f"{n}|std" for n in feature_names]
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

    X_agg = np.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]
    return X_agg.astype(float), out_names

# Pick your aggregation set (robust defaults)
AGGS = ("median", "iqr", "cv")  # you can change to ("median",) if you prefer fewer features
X_subj, agg_names = aggregate_trials(X_all, aggs=AGGS, feature_names=feature_names)

# ---- 3.3 Filter valid subjects (non-NaN label) and build subject_id for LOSO ----
valid = np.isfinite(y_raw) & np.isfinite(X_subj).any(axis=1)
X_subj = X_subj[valid]
y_labels  = y_raw[valid].astype(int)
subject_id = subject_ids[valid]

# If subject_ids contain NaNs or non-integers, fall back to consecutive IDs
if not np.isfinite(subject_id).all():
    subject_id = np.arange(len(subject_id))
else:
    # Ensure integers (for LeaveOneGroupOut)
    subject_id = subject_id.astype(int)

agg_names = np.array(agg_names)

print(f"Target = {TARGET}")
print(f"X_subj shape: {X_subj.shape}  (subjects × aggregated features)")
print("Class balance:", pd.Series(y_labels).value_counts().to_dict())
print("Unique subjects:", len(np.unique(subject_id)))
print("Example aggregated feature names:", agg_names[:8].tolist())


# In[50]:


# === Cell 4: Stability Selection helper (Elastic-Net / L1 logistic, saga) ===

# ---------- helpers ----------
def _stratified_subsample_indices(y, frac, rng):
    idx = np.arange(len(y))
    take = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        n_take = max(2, int(np.floor(frac * len(cls_idx))))
        sel = rng.choice(cls_idx, size=n_take, replace=False)
        take.append(sel)
    return np.sort(np.concatenate(take))


def _sweep_C_path_right_edge(Xs, y, C_grid, q, l1_ratio=None, class_weight='balanced', max_iter=5000, tol=1e-4, seed=0):
    penalty = 'l1' if l1_ratio is None else 'elasticnet'

    clf = LogisticRegression(  penalty=penalty, solver='saga', C=float(C_grid[0]),
        l1_ratio=(None if l1_ratio is None else float(l1_ratio)), class_weight=class_weight, max_iter=max_iter, tol=tol,
        n_jobs=1, random_state=seed, warm_start=True)

    nz = np.empty(len(C_grid), dtype=int)
    models = []

    clf.fit(Xs, y)
    nz[0] = int(np.count_nonzero(clf.coef_))
    models.append(deepcopy(clf))

    for k in range(1, len(C_grid)):
        clf.set_params(C=float(C_grid[k]))
        clf.fit(Xs, y)
        nz[k] = int(np.count_nonzero(clf.coef_))
        models.append(deepcopy(clf))

    idx = np.where(nz == q)[0]
    if idx.size:
        j = idx[-1]
    else:
        d = np.abs(nz - q)
        j = np.flatnonzero(d == d.min())[-1]

    boundary_hit = (j == 0) or (j == len(C_grid)-1)
    chosen = models[j]
    mask = (np.abs(chosen.coef_.ravel()) > 1e-6)
    return float(C_grid[j]), mask, nz, boundary_hit

# ---------- main stability selection function ----------
def stability_selection_logreg( X, y, C_grid, n_subsamples=500, subsample_frac=0.5, l1_ratio=None, pi_thr=0.7, 
    class_weight='balanced',rng=None, verbose=False, return_refit=False, PFER_base= 3):
    rng = check_random_state(rng)
    n, p = X.shape

    # scale once
    Xs_full = StandardScaler().fit_transform(X)
    q_target = max(1, int(np.sqrt( PFER_base * (2 * pi_thr - 1) * p)))

    counts = np.zeros(p, dtype=int)
    q_per_run = []
    C_per_run = []
    boundary_hits = 0

    for t in range(n_subsamples):
        idx = _stratified_subsample_indices(y, frac=subsample_frac, rng=rng)
        Xh = Xs_full[idx]
        yh = y[idx]

        C_sel, mask, nz, bh = _sweep_C_path_right_edge( Xh, yh, C_grid=C_grid,
            q=q_target, l1_ratio=l1_ratio, class_weight=class_weight, seed=int(rng.randint(1_000_000_000)))

        counts += mask.astype(int)
        q_per_run.append(int(mask.sum()))
        C_per_run.append(float(C_sel))
        boundary_hits += bh

        if verbose and (t + 1) % max(1, n_subsamples // 10) == 0:
            print(f"  subsample {t+1}/{n_subsamples}: selected={q_per_run[-1]}")

    freq = counts / float(n_subsamples)
    q_hat = float(np.mean(q_per_run))
    boundary_rate = boundary_hits / float(n_subsamples)

    pfer_bound = (q_hat ** 2) / ((2 * pi_thr - 1 ) * p)
    selected_idx = np.where(freq >= pi_thr)[0]

    out = {'freq': freq, 'selected_idx': selected_idx,'pi_thr': float(pi_thr),
        'q_hat': q_hat,'pfer_bound': pfer_bound, 'C_per_run': np.array(C_per_run),
        'q_per_run': np.array(q_per_run, dtype=int),'boundary_rate': boundary_rate}

    if verbose:
        print(f"----->> l1_ratio:{l1_ratio} <<-------------------------------------------------------")
        print(f"(pi_thr: {pi_thr}, q_hat:{q_hat} and q_target: {q_target} on  C_grid: [{np.min(C_grid)},...,{np.max(C_grid)}] )")
        print(f"unique C = {np.unique(C_per_run)}")
        print("max freq:", freq.max(), "mean nonzero freq:", freq[freq > 0].mean())

    if return_refit and selected_idx.size > 0:
        Xs = StandardScaler().fit_transform(X[:, selected_idx])
        refit = LogisticRegression(
            penalty=None,        # <-- no regularization
            solver='lbfgs',
            class_weight=class_weight,
            max_iter=2000
        )
        refit.fit(Xs, y)
        out['refit_coef'] = refit.coef_.ravel()
        out['refit_intercept'] = float(refit.intercept_)


    return out 

print("Stability selection helper ready (plateau-aware C picking + refit option).")


# In[51]:


def youden_threshold_oof(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    keep = np.isfinite(thr)
    if not np.any(keep):
        return 0.5
    J = tpr[keep] - fpr[keep]
    return float(thr[keep][int(np.argmax(J))])


# # LOSO with Stability slection 

# In[52]:


# === Cell 5: helpers ===

SCALERS = { "robust": RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)),
    "standard": StandardScaler() }


def _stable_seed(params: dict, outer_fold: int, base: int = SEED):
    key = json.dumps(params, sort_keys=True, default=str)
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 1_000_000_000
    return (base + outer_fold * 1_000_000 + h) % (2**31 - 1)


def _oof_probs_with_C(X_tr_sel, y_tr, C, seed, inner_folds, scaler_key="standard", penalty='l2', l1_ratio=None):
    
    n = len(y_tr); oof = np.full(n, np.nan, dtype=float)
    k = max(2, min(inner_folds, int(np.bincount(y_tr).min())))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scaler = SCALERS[scaler_key]

    if penalty in ('l1','elasticnet'):
        solver, l1r = 'saga', (None if penalty=='l1' else float(l1_ratio))
    elif penalty in ('l2',None,'none'):
        solver, l1r = 'lbfgs', None
    else:
        raise ValueError("penalty must be one of: 'l1','elasticnet','l2','none'.")

    base = Pipeline(steps=[ ("imp", SimpleImputer(strategy="median")),  ("sc", scaler),
        ("clf", LogisticRegression( penalty=penalty_arg, solver=solver, C=C, l1_ratio=l1r, class_weight="balanced",
            max_iter=5000, tol=1e-4,  warm_start=False, random_state=seed)) ])
    for tr_i, va_i in skf.split(X_tr_sel, y_tr):
        pipe = clone(base)
        pipe.fit(X_tr_sel[tr_i], y_tr[tr_i])
        oof[va_i] = pipe.predict_proba(X_tr_sel[va_i])[:, 1]
    return oof


# Compute inner-CV balanced accuracies for a given C and preprocessing setup
def _inner_baccs_for_C(X_tr_sel, y_tr, C, seed, inner_folds, scaler_key="standard", penalty='l2', l1_ratio=None, class_weight="balanced"):
    baccs = []
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    scaler = SCALERS[scaler_key]

    if penalty in ('l1', 'elasticnet'):
        solver, l1r = 'saga', (None if penalty == 'l1' else float(l1_ratio))
    elif penalty in ('l2', 'none', None):
        solver, l1r = 'lbfgs', None
    else:
        raise ValueError("penalty must be one of: 'l1','elasticnet','l2','none'.")

    base_pipe = Pipeline(steps=[ ("imp", SimpleImputer(strategy="median")), ("sc", scaler),
        ("clf", LogisticRegression(penalty=penalty_arg, solver=solver, C=C, l1_ratio=l1r, class_weight=class_weight, 
                                   max_iter=5000, tol=1e-4, warm_start=False, random_state=seed)) ])

    for tr_i, va_i in skf.split(X_tr_sel, y_tr):
        pipe = clone(base_pipe)
        pipe.fit(X_tr_sel[tr_i], y_tr[tr_i])
        prob = pipe.predict_proba(X_tr_sel[va_i])[:, 1]
        y_va = y_tr[va_i]
        preds = (prob >= 0.5).astype(int)
        baccs.append(balanced_accuracy_score(y_va, preds))

    return np.asarray(baccs, float)


# In[53]:


# === Cell 6: LOSO  ===

def run_loso_once(X_subj, y_labels, subject_id, STAB_SEARCH_GRID, FINAL_PENALTIES=('none','l2'), FINAL_C_GRID=np.logspace(-6, 3, 200),
                  GLOBAL_MIN_FREQ=0.9, GLOBAL_TOP_K=None, INNER_FOLDS=5, SCALERS=SCALERS,  agg_names=None, # optional names for features (len = p)
                  out_dir=None,   cluster_label=None,    target_label=None,  SEED=42, verbose=True ):

    n, p = X_subj.shape #number of subjects and number of features
    selection_counts = np.zeros(p, dtype=int)
    coef_sum = np.zeros(p, dtype=float)
    coef_abs_sum = np.zeros(p, dtype=float)
    coef_n = np.zeros(p, dtype=int)

    per_fold_freq_full = []
    per_fold_selected = []
    y_true_all, y_prob_all, y_pred_all = [], [], []
    fold_rows = []

    outer_fold = 0
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X_subj, y, subject_id):
        outer_fold += 1
        print(f"----------------------  Fold: {outer_fold}/{n} --------------------- ")
        X_tr_raw, X_te_raw = X_subj[train_idx], X_subj[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # --- initialise per-fold bests (use BAcc, not AUC) ---
        best_inner_bacc = -np.inf
        BEST_STAB = None
        best_sel_idx_filt = None
        best_q_hat = np.nan
        best_pfer_bound = np.nan
        BEST_SCALER_KEY, BEST_C_FINAL = None, None
        BEST_L1_RATIO, BEST_PENALTY = None, None
        BEST_FREQ_FULL = None
        CHOSEN_BOUNDARY = np.nan

        # TRAIN-only zero-variance filter
        tr_var = np.var(X_tr_raw, axis=0)
        idx_keep = np.where(tr_var > 1e-16)[0]
        X_tr, X_te = X_tr_raw[:, idx_keep], X_te_raw[:, idx_keep]

        best_inner_auc = -np.inf
        BEST_STAB = None
        best_sel_idx_filt = None
        best_q_hat = np.nan
        best_pfer_bound = np.nan
        BEST_SCALER_KEY, BEST_C_FINAL = None, None
        BEST_L1_RATIO, BEST_PENALTY = None, None
        BEST_FREQ_FULL = None
        CHOSEN_BOUNDARY = np.nan

        # 1) Stability selection over grid + quick screen
        screened = []
        for params in ParameterGrid(STAB_SEARCH_GRID):
            params_local = params.copy()
            params_local["rng"] = _stable_seed(params_local, outer_fold)

            stab = stability_selection_logreg( X_tr, y_tr, C_grid=params_local["C_grid"],
                n_subsamples=params_local["n_subsamples"], subsample_frac=params_local["subsample_frac"],
                l1_ratio=params_local["l1_ratio"], pi_thr=params_local["pi_thr"],
                class_weight=params_local["class_weight"], rng=params_local["rng"],
                verbose=params_local["verbose"], return_refit=False)
            """ stab: 'freq', 'selected_idx' ,'pi_thr','q_hat','pfer_bound', 'C_per_run', 'q_per_run','boundary_rate'"""

            # map freq to FULL feature space
            f_local = np.asarray(stab.get("freq", []), float)
            freq_full = np.zeros(p, dtype=float)
            if f_local.size == X_tr.shape[1]:
                freq_full[idx_keep] = f_local
            elif f_local.size == p:
                freq_full = f_local.copy()
            elif verbose and outer_fold == 1 and f_local.size > 0:
                print(f"[warn] freq len {f_local.size} != {X_tr.shape[1]} or {p}")

            sel_idx_filt = np.array(stab.get("selected_idx", []), dtype=int)
            if sel_idx_filt.size == 0 and f_local.size:
                m = max(1, int(round(max(1 , stab.get("q_hat", 1)))))
                sel_idx_filt = np.argsort(f_local)[::-1][:m]
                print(f" [fold {outer_fold}] no features above pi_thr; forcing top-{m} by freq.")
            if sel_idx_filt.size == 0:
                print(f"  [fold {outer_fold}] candidate skipped (no features selected).")
                continue

            screened.append({ "params": params_local,  "sel_idx_filt": sel_idx_filt,
                "q_hat": float(stab.get("q_hat", np.nan)),  "pfer": float(stab.get("pfer_bound", np.nan)),
                "C_per_run": stab.get("C_per_run", None),  "freq_full": freq_full,
                "boundary_rate": float(stab.get("boundary_rate", np.nan)) })

        if not screened: raise RuntimeError("stability_selection produced no viable candidates.")

        # 2) Inner FINAL sweep with 1-SE rule
        for cand in screened:
            sel_idx_filt = cand["sel_idx_filt"]
            X_tr_sel = X_tr[:, sel_idx_filt]
        
            # best inner-CV BAcc for this stability-selection candidate
            best_bacc_this = -np.inf
            best_C_this, best_scaler_key, best_l1r_this, best_penalty_this = None, None, None, None
        
            for scaler_key in SCALERS.keys():
                for pen in FINAL_PENALTIES:
                    if pen in ("none", "l2"):
                        l1r_try = None
                    elif pen == "elasticnet":
                        l1r_try = cand["params"]["l1_ratio"]
                        if l1r_try is None:
                            continue
                    elif pen == "l1":
                        l1r_try = None
                    else:
                        raise ValueError(f"Unknown penalty: {pen}")
        
                    stats = []  # (C, mean_bacc, se_bacc)
                    for C in FINAL_C_GRID:
                        baccs = _inner_baccs_for_C( X_tr_sel, y_tr, C, SEED, INNER_FOLDS, scaler_key=scaler_key, penalty=pen, 
                                                    l1_ratio=l1r_try)
                        mean_bacc = baccs.mean()
                        se_bacc = baccs.std(ddof=1) / np.sqrt(len(baccs)) if baccs.size > 1 else 0.0
                        stats.append((C, mean_bacc, se_bacc))
        
                    means = np.array([m for (_, m, _) in stats], float)
                    best_i = int(np.argmax(means))
                    best_mean, best_se = stats[best_i][1], stats[best_i][2]
        
                    # 1-SE rule on BAcc, then choose smallest C among acceptable-> we want the least overfitting
                    thr = best_mean - best_se
                    acceptable = [(C, m) for (C, m, se) in stats if m >= thr]
                    C_choice, score_choice = min(acceptable, key=lambda cm: cm[0])
        
                    if (score_choice > best_bacc_this) or ( np.isclose(score_choice, best_bacc_this) and 
                                                            (best_C_this is None or C_choice < best_C_this) ):
                        best_bacc_this = score_choice
                        best_C_this = C_choice
                        best_scaler_key = scaler_key
                        best_penalty_this = pen
                        best_l1r_this = l1r_try
                            
                    if best_bacc_this > best_inner_bacc:
                        best_inner_bacc = best_bacc_this
                        BEST_STAB = cand["params"]
                        best_sel_idx_filt = sel_idx_filt
                        best_q_hat = cand["q_hat"]
                        best_pfer_bound = cand["pfer"]
                        BEST_SCALER_KEY, BEST_C_FINAL = best_scaler_key, best_C_this
                        BEST_PENALTY, BEST_L1_RATIO = best_penalty_this, best_l1r_this
                        BEST_FREQ_FULL = cand["freq_full"]
                        CHOSEN_BOUNDARY = cand["boundary_rate"]
                
                # defensive check (optional)
                if best_sel_idx_filt is None:
                    raise RuntimeError("Inner FINAL sweep failed to select any viable model.")


        # Map back to full index space
        sel_idx_full = idx_keep[best_sel_idx_filt]
        selection_counts[sel_idx_full] += 1
        if agg_names is not None:
            per_fold_selected.append([agg_names[i] for i in sel_idx_full])
        else:
            per_fold_selected.append(sel_idx_full.tolist())
        per_fold_freq_full.append(BEST_FREQ_FULL if BEST_FREQ_FULL is not None else np.zeros(p, float))

        # OOF train probabilities for metrics + threshold
        X_tr_sel = X_tr[:, best_sel_idx_filt]
        X_te_sel = X_te[:, best_sel_idx_filt]

        oof_probs = _oof_probs_with_C( X_tr_sel, y_tr, BEST_C_FINAL, SEED, INNER_FOLDS, scaler_key=BEST_SCALER_KEY, 
                                       penalty=BEST_PENALTY, l1_ratio=BEST_L1_RATIO)
        thr = 0.5
        # thr = youden_threshold_oof(y_tr, oof_probs)

        loss_tr_oof = log_loss(y_tr, oof_probs, labels=[0, 1])
        yhat_tr_oof = (oof_probs >= thr).astype(int)
        acc_tr_oof  = accuracy_score(y_tr, yhat_tr_oof)
        bacc_tr_oof = balanced_accuracy_score(y_tr, yhat_tr_oof)
        auc_tr_oof  = roc_auc_score(y_tr, oof_probs) if len(np.unique(y_tr)) == 2 else np.nan

        solver = 'lbfgs' if BEST_PENALTY in ('l2', 'none', None) else 'saga'
        l1r_use = (float(BEST_L1_RATIO) if BEST_PENALTY == 'elasticnet' else None)
        C_final = (1.0 if BEST_PENALTY == 'none' else (BEST_C_FINAL if BEST_C_FINAL is not None else 1.0))
        penalty_arg = None if BEST_PENALTY in ('none', None) else BEST_PENALTY
        
        final_pipe = Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", SCALERS[BEST_SCALER_KEY]),
            ("clf", LogisticRegression(
                penalty=penalty_arg,    # <---
                solver=solver,
                C=C_final,
                l1_ratio=l1r_use,
                class_weight="balanced",
                max_iter=5000,
                tol=1e-4,
                warm_start=False,
                random_state=SEED))
        ])

        final_pipe.fit(X_tr_sel, y_tr)

        prob_te = final_pipe.predict_proba(X_te_sel)[:, 1]
        yhat_te = (prob_te >= thr).astype(int)
        loss_te = log_loss(y_te, prob_te, labels=[0, 1])
        acc_te  = accuracy_score(y_te, yhat_te)

        clf = final_pipe.named_steps["clf"]
        if hasattr(clf, "coef_"):
            coef_sel = clf.coef_.ravel()
            coef_sum[sel_idx_full]     += coef_sel
            coef_abs_sum[sel_idx_full] += np.abs(coef_sel)
            coef_n[sel_idx_full]       += 1

        # pooled-outer vectors (first element if LOSO yields single test sample)
        y_true_all.append(int(y_te[0]))
        y_prob_all.append(float(prob_te[0]))
        y_pred_all.append(int(yhat_te[0]))

        bacc_te = acc_te

        C_report = float(BEST_C_FINAL if BEST_C_FINAL is not None else 1.0)
        l1r_report = (None if BEST_PENALTY != 'elasticnet' else (None if BEST_L1_RATIO is None else float(BEST_L1_RATIO)))

        if verbose:
            print(
                f"left_out={int(test_idx[0])} | feats={sel_idx_full.size} | "
                f"pen={BEST_PENALTY}{'' if l1r_report is None else f' (l1_ratio={l1r_report:.2f})'}| "
                f"C={C_report:.4g}| scaler={BEST_SCALER_KEY}| "
                f"AUC(tr-OOF)={auc_tr_oof:.3f} | "
                f"BAcc(tr-OOF)={bacc_tr_oof:.3f}  BAcc(te)={bacc_te:.0f} | "
                f"Loss(tr-OOF)={loss_tr_oof:.3f}  Loss(te)={loss_te:.3f}"
            )

        fold_rows.append({
            "fold": outer_fold,  "left_out_idx": int(test_idx[0]),  "n_selected": int(sel_idx_full.size),
            "C_used": C_report, "q_hat": float(best_q_hat),  "pfer": float(best_pfer_bound),
            "thr_train": float(thr),  "auc_train":  float(auc_tr_oof),  "acc_train":  float(acc_tr_oof),
            "bacc_train": float(bacc_tr_oof), "logloss_train": float(loss_tr_oof),
            "acc_test":   float(acc_te),  "bacc_test":  float(bacc_te), "logloss_test": float(loss_te),
            "boundary_rate": float(CHOSEN_BOUNDARY), "stab_params": BEST_STAB, "scaler": BEST_SCALER_KEY,
        })

    # Outer pooled metrics
    y_true_all = np.array(y_true_all, dtype=int)
    y_prob_all = np.array(y_prob_all, dtype=float)
    y_pred_all = np.array(y_pred_all, dtype=int)
    
    AUC  = roc_auc_score(y_true_all, y_prob_all)
    ACC  = accuracy_score(y_true_all, y_pred_all)
    BACC = balanced_accuracy_score(y_true_all, y_pred_all)
    F1   = f1_score(y_true_all, y_pred_all)

    # --- Outer metrics at Youden threshold (post-hoc, optimistic) ---
    thr_youden_outer = youden_threshold_oof(y_true_all, y_prob_all)
    y_pred_youden    = (y_prob_all >= thr_youden_outer).astype(int)
    ACC_y  = accuracy_score(y_true_all, y_pred_youden)
    BACC_y = balanced_accuracy_score(y_true_all, y_pred_youden)
    F1_y   = f1_score(y_true_all, y_pred_youden)


    if verbose:
        tag = target_label if target_label is not None else "—"
        print(f"[Stability-LOSO | TARGET={tag}] N={len(y_true_all)}  Acc={ACC:.3f}  BalAcc={BACC:.3f}  F1={F1:.3f}  AUC={AUC:.3f}")

    fold_df = pd.DataFrame(fold_rows)

    # Feature stability across folds
    n_folds = int(fold_df.shape[0])
    stability_freq = selection_counts / max(1, n_folds)
    coef_mean = np.zeros(p, dtype=float)
    coef_abs_mean = np.zeros(p, dtype=float)
    np.divide(coef_sum, np.maximum(coef_n, 1), out=coef_mean, where=(coef_n > 0))
    np.divide(coef_abs_sum, np.maximum(coef_n, 1), out=coef_abs_mean, where=(coef_n > 0))

    names = agg_names if agg_names is not None else [f"f{i}" for i in range(p)]
    stab_df = pd.DataFrame({
        "feature": names,
        "fold_select_count": selection_counts,
        "fold_select_freq": stability_freq,
        "coef_mean": coef_mean,
        "coef_abs_mean": coef_abs_mean,
        "coef_n": coef_n,
    }).sort_values(
        ["fold_select_freq", "fold_select_count", "coef_abs_mean"],
        ascending=[False, False, False]
    )

    # Global stable set
    stab_nonzero = stab_df[stab_df["fold_select_freq"] >= float(GLOBAL_MIN_FREQ)].copy()
    stab_top = stab_nonzero.head(GLOBAL_TOP_K) if GLOBAL_TOP_K is not None else stab_nonzero
    global_stable_features = stab_top["feature"].tolist()
    name_to_idx = {name: i for i, name in enumerate(names)}
    global_stable_idx = np.array([name_to_idx[n] for n in global_stable_features], dtype=int)

    if verbose:
        print(f"\n[GLOBAL STABLE SET] {len(global_stable_idx)} features")
        for n_ in global_stable_features:
            print("  -", n_)

    # Optional save
    if out_dir is not None and cluster_label is not None and target_label is not None:
        stab_path = os.path.join(out_dir, f"stability_table_{cluster_label}_{target_label}.csv")
        global_path = os.path.join(out_dir, f"global_stable_features_{cluster_label}_{target_label}.csv")
        stab_df.to_csv(stab_path, index=False)
        pd.DataFrame({"feature": global_stable_features}).to_csv(global_path, index=False)
        if verbose:
            print("\nSaved:")
            print(" ", stab_path)
            print(" ", global_path)

    outer_preds = pd.DataFrame({  "left_out_idx": fold_df["left_out_idx"].values,  "y_true": y_true_all,
                                  "y_prob": y_prob_all,  "y_pred": y_pred_all})

    return { "fold_df": fold_df,  "stab_df": stab_df, 
             "outer_metrics": {"AUC": AUC, "ACC": ACC, "BACC": BACC, "F1": F1, "thr_youden_outer": float(thr_youden_outer),
                               "ACC_youden_outer":  ACC_y, "BACC_youden_outer": BACC_y,"F1_youden_outer":   F1_y},
        "outer_preds": outer_preds,  "global_stable_idx": global_stable_idx, "global_stable_features": global_stable_features,
        "per_fold_freq_full": per_fold_freq_full, "per_fold_selected_names": per_fold_selected }


# In[82]:


#========== Cell 7: evaluation for hyperparam tuning =================
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# =====================================================================
# === 1. Evaluate one (pfer, pi_thr, l1_ratio) with 5-fold CV       ===
# =====================================================================

def evaluate_block(X, y, pfer, pi_thr, l1_ratio, C_grid, cv_folds=5):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)

    fold_baccs = []
    fold_accs = []
    fold_scalers = []
    C_used = []
    fold_metrics = []

    SCALERS = {"standard": StandardScaler(), "robust": RobustScaler()}

    for fold_n, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # --------------------------
        # Stability selection inside this fold
        # --------------------------
        stab_res = stability_selection_logreg(X_tr, y_tr,C_grid=C_grid,n_subsamples=1000,subsample_frac=0.5,
            l1_ratio=l1_ratio, pi_thr=pi_thr, class_weight="balanced",PFER_base=pfer)

        for c in stab_res.get("C_per_run", []):
            if c not in C_used:
                C_used.append(c)
        stable_idx = stab_res["selected_idx"]

        # No features selected
        if stable_idx.size == 0:
            fold_baccs.append(np.nan)
            fold_accs.append(np.nan)
            fold_scalers.append("none")
            fold_metrics.append({ "fold": fold_n,  "scaler": "none", "bacc": np.nan, "acc": np.nan, "features": 0 })
            continue

        X_tr_s = X_tr[:, stable_idx]
        X_val_s = X_val[:, stable_idx]
        n_features = stable_idx.size

        # --------------------------
        # Test both scalers
        # --------------------------
        best_bacc = -np.inf
        best_acc = 0.5
        best_scaler = None

        for scaler_name, scaler_obj in SCALERS.items():

            pipe = Pipeline([ ("imp", SimpleImputer(strategy="median")),("sc", scaler_obj),
                ("clf", LogisticRegression( penalty=None, solver="lbfgs", C=1e5, class_weight="balanced", max_iter=5000 ))
            ])

            pipe.fit(X_tr_s, y_tr)
            y_pred = pipe.predict(X_val_s)

            bacc = balanced_accuracy_score(y_val, y_pred)
            acc = accuracy_score(y_val, y_pred)

            # Save fold result (for plotting later)
            fold_metrics.append({  "fold": fold_n, "scaler": scaler_name, "bacc": bacc, "acc": acc, "features": n_features })

            if bacc > best_bacc:
                best_bacc = bacc
                best_acc = acc
                best_scaler = scaler_name

        fold_baccs.append(best_bacc)
        fold_accs.append(best_acc)
        fold_scalers.append(best_scaler)

    avg_bacc = float(np.nanmean(fold_baccs))
    avg_acc  = float(np.nanmean(fold_accs))

    return avg_bacc, avg_acc, fold_scalers, C_used, fold_metrics


# In[83]:


#============= Cell 8: Tuning call ==================
print("\n==============================")
print(" SINGLE-STAGE GRID SEARCH ")
print("==============================\n")

all_results_log = []

# Search grids
PFER_GRID     = [1, 2, 3, 4, 5, 6, 7]
PI_THR_GRID   = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85,0.9,0.95]
L1_RATIO_GRID = [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

C_stab_grid = np.logspace(-5, 3, 50) # a wide range for scanning

n_pfer = len(PFER_GRID)
n_pi   = len(PI_THR_GRID)
n_l1   = len(L1_RATIO_GRID)

# 3D grids for BAcc and Acc
bacc_grid = np.full((n_pfer, n_pi, n_l1), np.nan)
acc_grid  = np.full((n_pfer, n_pi, n_l1), np.nan)

# For later retrieval of C path and scalers
C_values_dict      = {}  # key: (pfer, pi, l1) -> C_values list
scaler_votes_dict  = {}  # key: (pfer, pi, l1) -> fold_scalers list

# Store accuracy grouped by (PFER, pi) pair
accuracies_by_pair = {}   # key = (pfer,pi), value = list of {"l1":x, "acc":y, "bacc":z}

# ----------------------------------------------------------------------
# 1) Search all combinations, fill grids and logs
# ----------------------------------------------------------------------
for ip, pfer in enumerate(PFER_GRID):
    for ii, pi in enumerate(PI_THR_GRID):
        for il, l1 in enumerate(L1_RATIO_GRID):

            print(f"now checking pfer={pfer}, pi={pi}, l1={l1}")

            avg_bacc, avg_acc, scalers, Cvals, fold_metrics = evaluate_block(X_subj, y_labels, 
                pfer=pfer, pi_thr=pi, l1_ratio=l1,C_grid=C_stab_grid, cv_folds= 5,)

            print(f"avg_bacc={avg_bacc}")

            bacc_grid[ip, ii, il] = avg_bacc
            acc_grid[ip,  ii, il] = avg_acc


if out_dir is not None:
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "stability_grid_LPFC_all.npz"),
        bacc_grid=bacc_grid,
        acc_grid=acc_grid,
        PFER_GRID=np.array(PFER_GRID),
        PI_THR_GRID=np.array(PI_THR_GRID),
        L1_RATIO_GRID=np.array(L1_RATIO_GRID),)

# ----------------------------------------------------------------------
# 2) Plateau-based selection:
#    - restrict to points within 95% of global max BAcc
#    - among them, choose the one with highest mean BAcc in its 3×3×3 neighborhood
#    - tie-break: higher center BAcc, then more valid neighbors
#    Then, for the chosen (pfer*, pi*), select up to 3 l1 values
#    with BAcc >= 95% of the best BAcc at that (pfer*, pi*).
# ----------------------------------------------------------------------

if np.all(np.isnan(bacc_grid)):
    raise RuntimeError("All balanced accuracies are NaN – cannot select hyperparameters.")

rel_thresh = 0.95
bacc_max = float(np.nanmax(bacc_grid))

# 2.1: candidate set (near-global maxima)
candidate_mask = (bacc_grid >= rel_thresh * bacc_max) & ~np.isnan(bacc_grid)
if not np.any(candidate_mask):
    # fallback: use all non-NaN points if threshold is too strict
    candidate_mask = ~np.isnan(bacc_grid)

best_score = -np.inf      # neighborhood mean
best_center = -np.inf     # center BAcc
best_n_valid = -1         # number of valid neighbors
best_idx = None

for ip in range(n_pfer):
    for ii in range(n_pi):
        for il in range(n_l1):
            if not candidate_mask[ip, ii, il]:
                continue

            center = bacc_grid[ip, ii, il]

            # 3×3×3 neighborhood (clipped at borders)
            p_lo  = max(0, ip - 1)
            p_hi  = min(n_pfer - 1, ip + 1)
            pi_lo = max(0, ii - 1)
            pi_hi = min(n_pi   - 1, ii + 1)
            l_lo  = max(0, il - 1)
            l_hi  = min(n_l1  - 1, il + 1)

            local = bacc_grid[p_lo:p_hi+1, pi_lo:pi_hi+1, l_lo:l_hi+1]
            valid = local[~np.isnan(local)]
            if valid.size == 0:
                continue

            local_mean = float(np.mean(valid))
            n_valid = int(valid.size)

            # tie-breaking: plateau mean, then center BAcc, then number of neighbors
            if (local_mean > best_score or
                (np.isclose(local_mean, best_score) and center > best_center) or
                (np.isclose(local_mean, best_score) and np.isclose(center, best_center) and n_valid > best_n_valid)):
                best_score = local_mean
                best_center = center
                best_n_valid = n_valid
                best_idx = (ip, ii, il)

# Fallback: if for some reason we didn't find a plateau candidate, use global max
if best_idx is None:
    flat_idx = np.nanargmax(bacc_grid)
    best_idx = np.unravel_index(flat_idx, bacc_grid.shape)
    best_score = float("nan")
    best_n_valid = 0

ip_star, ii_star, il_star = best_idx

pfer_star = PFER_GRID[ip_star]
pi_star   = PI_THR_GRID[ii_star]

# ----------------------------------------------------------------------
# 3) L1 selection at the chosen (pfer*, pi*):
#    keep up to 3 l1 ratios with BAcc >= 95% of the best BAcc on that slice
# ----------------------------------------------------------------------
bacc_l1 = bacc_grid[ip_star, ii_star, :]
acc_l1  = acc_grid[ip_star, ii_star, :]

if np.all(np.isnan(bacc_l1)):
    # pathological case: slice is NaN; fall back to il_star from plateau
    l1_star_idx = il_star
    top_l1_idx = [il_star]
    best_l1_bacc = float(bacc_grid[ip_star, ii_star, il_star])
else:
    best_l1_idx = int(np.nanargmax(bacc_l1))
    best_l1_bacc = float(bacc_l1[best_l1_idx])
    l1_bacc_thresh = 0.95 * best_l1_bacc

    candidates_l = [
        (il, float(bacc_l1[il]))
        for il in range(n_l1)
        if (not np.isnan(bacc_l1[il])) and (bacc_l1[il] >= l1_bacc_thresh)
    ]
    candidates_l_sorted = sorted(candidates_l, key=lambda x: x[1], reverse=True)

    if not candidates_l_sorted:
        # at least keep the best l1 for this slice
        top_l1_idx = [best_l1_idx]
    else:
        # keep up to 3 best l1 values
        top_l1_idx = [il for (il, _) in candidates_l_sorted[:3]]

    l1_star_idx = top_l1_idx[0]  # representative l1 for "main" triple
    best_l1_bacc = float(bacc_l1[l1_star_idx])

l1_star = L1_RATIO_GRID[l1_star_idx]
best_bacc = float(bacc_grid[ip_star, ii_star, l1_star_idx])
best_acc  = float(acc_l1[l1_star_idx])

key3_star = (pfer_star, pi_star, l1_star)

# dominant scaler from the winning combination
votes = [s for s in scaler_votes_dict.get(key3_star, []) if s != "none"]
dom_scaler = Counter(votes).most_common(1)[0][0] if votes else "robust"

top_l1_ratios_95 = [float(L1_RATIO_GRID[il]) for il in top_l1_idx]

best_params = {
    "pfer": float(pfer_star),
    "pi_thr": float(pi_star),
    "best_l1_ratio": float(l1_star),
    "top_l1_ratios_95pct": top_l1_ratios_95,
    "best_bacc": best_bacc,
    "best_acc": best_acc,
    "dominant_scaler": dom_scaler,
    "C_values": C_values_dict.get(key3_star, []),
    "plateau_score": float(best_score),
    "plateau_n_valid": int(best_n_valid),
    "global_max_bacc": bacc_max,
    "rel_thresh": rel_thresh,
    "l1_slice_best_bacc": best_l1_bacc,
    "l1_slice_thresh_95pct": l1_bacc_thresh if not np.all(np.isnan(bacc_l1)) else None,
}

print("\n==============================")
print(" BEST PARAMETERS (PLATEAU-BASED)")
print("==============================")
print(
    f"(PFER={pfer_star}, pi={pi_star}, L1={l1_star})  "
    f"-> BAcc={best_bacc:.4f}, Acc={best_acc:.4f}, "
    f"PlateauScore={best_score:.4f}, "
    f"Neighbors={best_n_valid}, "
    f"GlobalMaxBAcc={bacc_max:.4f}"
)
print(f"L1s within 95% of best BAcc at (PFER*, pi*): {top_l1_ratios_95}")
print(best_params)

if out_dir is not None:
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "best_params_LPFC_all.npz"),
        best_params=best_params,)


# # Ploting the grid space

# In[84]:


#========== Cell 9: Plot hyperparam grid space
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

# --------------------------------------------------------------------
# Build df_summary directly from bacc_grid / acc_grid
# --------------------------------------------------------------------
rows = []
for ip, pfer in enumerate(PFER_GRID):
    for ii, pi in enumerate(PI_THR_GRID):
        for il, l1 in enumerate(L1_RATIO_GRID):
            b = bacc_grid[ip, ii, il]
            a = acc_grid[ip, ii, il]
            if np.isnan(b):
                continue
            rows.append({
                "pfer": float(pfer),
                "pi_thr": float(pi),
                "l1_ratio": float(l1),
                "mean_bacc": float(b),
                "mean_acc": float(a),
            })

df_summary = pd.DataFrame(rows)

# --------------------------------------------------------------------
# 1) 1D PLOTS WITH IQR SHADED REGIONS
# --------------------------------------------------------------------

def compute_iqr(df, var):
    grp = df.groupby(var)[["mean_bacc", "mean_acc"]]
    out = grp.agg([
        ("mean", "mean"),
        ("q1", lambda x: x.quantile(0.25)),
        ("q3", lambda x: x.quantile(0.75)),
    ])
    out.columns = ["_".join(col) for col in out.columns]
    out = out.reset_index()
    return out

df_pfer = compute_iqr(df_summary, "pfer")
df_pi   = compute_iqr(df_summary, "pi_thr")
df_l1   = compute_iqr(df_summary, "l1_ratio")

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for ax, df_iqr, label in zip(
    axes,
    [df_pfer, df_pi, df_l1],
    ["PFER", "Pi_thr", "L1_ratio"]
):
    x = df_iqr.iloc[:, 0]

    ax.plot(x, df_iqr["mean_bacc_mean"], marker="o", label="BAcc", color="blue")
    ax.fill_between(x, df_iqr["mean_bacc_q1"], df_iqr["mean_bacc_q3"],
                    color="blue", alpha=0.2)

    ax.plot(x, df_iqr["mean_acc_mean"], marker="x", label="Acc", color="red")
    ax.fill_between(x, df_iqr["mean_acc_q1"], df_iqr["mean_acc_q3"],
                    color="red", alpha=0.2)

    ax.set_title(f"{label} vs Performance")
    ax.set_xlabel(label)
    ax.set_ylabel("Score")
    ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(out_dir, "1D_performance_curves_IQR.png"), dpi=200)
plt.show()

# --------------------------------------------------------------------
# 2) HEATMAPS — AVERAGED OVER THIRD VARIABLE
# --------------------------------------------------------------------

def pivot_mean(df, row, col, metric):
    return df.pivot_table(index=row, columns=col, values=metric, aggfunc="mean")

pairs = [
    ("pfer", "pi_thr", "l1_ratio"),
    ("pfer", "l1_ratio", "pi_thr"),
    ("pi_thr", "l1_ratio", "pfer"),
]

fig, axes = plt.subplots(3, 2, figsize=(18, 20))

for (row, col, other), (ax_b, ax_a) in zip(pairs, axes):
    piv_b = pivot_mean(df_summary, row, col, "mean_bacc")
    piv_a = pivot_mean(df_summary, row, col, "mean_acc")

    sns.heatmap(piv_b, cmap="coolwarm", annot=True, fmt=".3f", ax=ax_b)
    sns.heatmap(piv_a, cmap="coolwarm", annot=True, fmt=".3f", ax=ax_a)

    ax_b.set_title(f"{row} vs {col} — Mean BAcc")
    ax_a.set_title(f"{row} vs {col} — Mean Acc")

fig.tight_layout()
fig.savefig(os.path.join(out_dir, "2D_heatmaps_mean_performance.png"), dpi=200)
plt.show()

# --------------------------------------------------------------------
# 3) BEST BAcc SURFACES FOR EACH PAIR
# --------------------------------------------------------------------

def pivot_best(df, row, col):
    df_best = df.groupby([row, col]).agg(best_bacc=("mean_bacc", "max")).reset_index()
    return df_best.pivot(index=row, columns=col, values="best_bacc")

fig, axes = plt.subplots(3, 1, figsize=(12, 20))

for (row, col, other), ax in zip(pairs, axes):
    piv_best = pivot_best(df_summary, row, col)
    sns.heatmap(piv_best, cmap="coolwarm", annot=True, fmt=".3f", ax=ax)
    ax.set_title(f"BEST BAcc — {row} vs {col}")

fig.tight_layout()
fig.savefig(os.path.join(out_dir, "2D_best_BAcc_surfaces.png"), dpi=200)
plt.show()


# In[85]:


#============= Cell 10: Plot 3D of the hyperparam grid
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused, needed for 3D

# Make coordinate grid
P, Pi, L1 = np.meshgrid(
    PFER_GRID, PI_THR_GRID, L1_RATIO_GRID,
    indexing="ij"   # so P[ip,ii,il] matches bacc_grid[ip,ii,il]
)

# Flatten
x = P.ravel()
y = Pi.ravel()
z = L1.ravel()
c = bacc_grid.ravel()

# Drop NaNs
mask = ~np.isnan(c)
x = x[mask]
y = y[mask]
z = z[mask]
c = c[mask]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(x, y, z, c=c, cmap="viridis")
cb = fig.colorbar(sc, ax=ax, shrink=0.8)
cb.set_label("Balanced accuracy")

ax.set_xlabel("PFER")
ax.set_ylabel("pi_thr")
ax.set_zlabel("l1_ratio")
ax.set_title("3D hyperparameter grid — BAcc")

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "3D_bacc_scatter.png"), dpi=200)
plt.show()


# # ***Running LOSO*** 

# In[88]:


#========== Cell 11: The LOSO call =================
from copy import deepcopy

SCALERS = {'standard': StandardScaler(),'robust': RobustScaler()}

best_params = {
    "pfer": float(pfer_star),
    "pi_thr": float(pi_star),
    "best_l1_ratio": float(l1_star),
    "top_l1_ratios_95pct": top_l1_ratios_95,
    "best_bacc": best_bacc,
    "best_acc": best_acc,
    "dominant_scaler": dom_scaler,
    "C_values": C_values_dict.get(key3_star, []),
    "plateau_score": float(best_score),
    "plateau_n_valid": int(best_n_valid),
    "global_max_bacc": bacc_max,
    "rel_thresh": rel_thresh,
    "l1_slice_best_bacc": best_l1_bacc,
    "l1_slice_thresh_95pct": l1_bacc_thresh if not np.all(np.isnan(bacc_l1)) else None,
}

# ---- Read best stability parameters from the search ----
pfer_best = best_params["pfer"]
pi_best   = best_params["pi_thr"]
l1_best = best_params["top_l1_ratios_95pct"]

c_range = best_params.get("C_values", [])

if len(c_range) > 0:
    cmin = float(np.min(c_range))
    cmax = float(np.max(c_range))
    log_cmin = np.log10(cmin)
    log_cmax = np.log10(cmax)

    # 1) Stability C grid, focused on the range where stability-selection actually operated
    C_STAB = np.logspace(log_cmin-0.5, log_cmax+0.5, 50)
    print(f"[REFINED C GRIDS] STAB: 10^[{log_cmin-1:.2f},{log_cmax+1:.2f}] ")
else:
    # Fallback: your previous manual ranges
    C_STAB  = np.logspace(-4, 3, 50)
    print("[REFINED C GRIDS] No c_grid_range_for_refinement found; using default ranges.")


C_FINAL = np.logspace(-5, 3, 200) #This is where we can give the final boost to  the model performance with l2

FINAL_PENALTIES = ('none', 'l2',)

# ---- Stability-selection grid, now driven by final_best_setting ----
STAB_SEARCH_GRID = {
    "n_subsamples": [1000],
    "subsample_frac": [0.5],
    "C": [None],          # unused by stability_selection_logreg
    "C_grid": [C_STAB],   # refined stability path
    "l1_ratio": l1_best,
    "pi_thr": [pi_best],
    "class_weight": ["balanced"],
    "verbose": [False],
    "PFER": [pfer_best],  # passed inside stability_selection_logreg as PFER_base
}

# ---- LOSO with the tuned stability hyperparameters ----
res = run_loso_once(
    X_subj, y_labels, subject_id=subject_ids,
    STAB_SEARCH_GRID=STAB_SEARCH_GRID,
    FINAL_PENALTIES=FINAL_PENALTIES,
    FINAL_C_GRID=C_FINAL,
    GLOBAL_MIN_FREQ=0.85,
    GLOBAL_TOP_K=None,
    INNER_FOLDS=5,
    SCALERS=SCALERS,
    agg_names=agg_names,        # or None if not available
    out_dir=out_dir,            # or None
    cluster_label=CLUSTER,      # or None
    target_label=TARGET,        # or None
    SEED=42,
    verbose=True
)

fold_df    = res["fold_df"]
stab_df    = res["stab_df"]
metrics    = res["outer_metrics"]
glob_idx   = res["global_stable_idx"]
glob_feats = res["global_stable_features"]

print("\nOuter pooled metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.3f}")

print(f"\nNumber of globally stable features (fold_select_freq >= 0.85): {len(glob_idx)}")
if len(glob_feats) > 0:
    print("Top few globally stable features:")
    for name in glob_feats[:10]:
        print("  -", name)


# # **Visualization and reporting of the results**

# In[89]:


# ============ Cell 12: Create table of top stable features =================
import numpy as np
import pandas as pd
from IPython.display import display, HTML

def make_pretty_label(feat):
    f = str(feat)

    # --- Replace TAR and Ŝ ---
    f = f.replace("powermean_Theta/powermean_Alpha", "TAR")
    f = f.replace("signalmean", r"$\hat{S}$")

    # Small cleanup
    f = f.replace("_|", "|")
    return f


# ---------------------------------------------------------------------
# Use the tuned pi_thr from the stability search, not a hard-coded value
# ---------------------------------------------------------------------
pi_thr = float(best_params["pi_thr"])   # from plateau-based selection
p      = stab_df.shape[0]              # total number of features

if (2 * pi_thr - 1) > 0:
    # PFER-based q_target heuristic
    n_top = max(1, int(np.sqrt((2 * pi_thr - 1) * p)))
else:
    n_top = 7  # fallback

print(f"Showing top {n_top} features (p={p}, pi_thr={pi_thr:.2f})")

# If you prefer a hard cap at 7, you can override:
# n_top = 7


# ---------------------------------------------------------------------
# Create table of top stable features (sorted by selection frequency)
# ---------------------------------------------------------------------
# If stab_df is not already sorted, enforce it here:
stab_sorted = stab_df.sort_values("fold_select_freq", ascending=False)

top_df = stab_sorted.head(n_top).copy()
top_df["feature"] = [make_pretty_label(f) for f in top_df["feature"]]

# Rename columns for clarity (must match stab_df column names)
top_df = top_df.rename(columns={
    "feature":         "Feature",
    "fold_select_freq":"Selection Frequency",
    "coef_mean":       "Coefficient (mean)",
    "coef_abs_mean":   "|Coefficient| (abs mean)"
})

# Round numeric values for readability
cols_to_round = ["Selection Frequency", "Coefficient (mean)", "|Coefficient| (abs mean)"]
top_df[cols_to_round] = top_df[cols_to_round].round(3)


# ---------------------------------------------------------------------
# Display as a styled HTML table (nice for paper screenshots / SI)
# ---------------------------------------------------------------------
caption_txt = (
    f"<b>Top {n_top} Stable Features — {CLUSTER}</b><br>"
    f"(PFER={best_params['pfer']:.0f}, "
    f"π<sub>thr</sub>={best_params['pi_thr']:.2f}, "
    f"L1 ∈ {best_params['top_l1_ratios_95pct']})"
)

styled = (
    top_df.style
    .set_table_attributes('style="font-size:14px; border-collapse:collapse;"')
    .set_caption(caption_txt)
    .set_properties(**{
        'font-weight': 'bold',
        'text-align': 'center',
        'border': '1px solid black',
        'padding': '4px'
    })
)

display(HTML(styled.to_html()))

tex_path = os.path.join(out_dir, f"top_features_{CLUSTER}.tex")

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(top_df.to_latex(index=False, escape=False))

print(f"LaTeX table saved to: {tex_path}")


# In[90]:


# ======= Cell 16: Publication plots & tables (enhanced reporting, polished) ========
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score
)

# Try to detect if we can display tables nicely
try:
    from IPython.display import display
    _CAN_DISPLAY = True
except Exception:
    _CAN_DISPLAY = False

# --------------------------------------------------------------------
# Output directory (assumed defined earlier as out_dir)
# --------------------------------------------------------------------
OUT_DIR = globals().get("out_dir", ".")
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Get outputs from run_loso_once
# --------------------------------------------------------------------
loso_res = res  # or keep your own assignment
fold_df     = loso_res["fold_df"]
stab_df     = loso_res["stab_df"]
outer_preds = loso_res["outer_preds"]

# 🔧 User-toggles / labels
USE_BOOTSTRAP   = True
N_BOOT          = 1500
CLASS_NAMES     = globals().get("CLASS_NAMES", {0: "Healthy", 1: "Case"})
CLUSTER_LABEL   = str(globals().get("CLUSTER", "cluster"))
TARGET_LABEL    = str(globals().get("TARGET", "target"))
TITLE_PREFIX    = f"{CLUSTER_LABEL} | {TARGET_LABEL}"

# ----- Prepare arrays -----
y_true = outer_preds["y_true"].to_numpy()
y_prob = outer_preds["y_prob"].to_numpy()
y_pred = outer_preds["y_pred"].to_numpy()
mask   = np.isfinite(y_true) & np.isfinite(y_prob) & np.isfinite(y_pred)
has_binary = (mask.sum() >= 2) and (np.unique(y_true[mask]).size == 2)

# ------------------------- Helpers -------------------------
def _bootstrap_curve_stats(y, s, *, n_boot=1000, seed=42, grid=None, mode="roc"):
    rng = np.random.RandomState(seed)
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    if grid is None:
        grid = np.linspace(0., 1., 101)
    collects = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)  # resample subjects with replacement
        yy = y[idx]; ss = s[idx]
        if np.unique(yy).size < 2:
            continue
        if mode == "roc":
            fpr, tpr, _ = roc_curve(yy, ss)
            fpr_u, idx_u = np.unique(fpr, return_index=True)
            tpr_u = tpr[idx_u]
            collects.append(np.interp(grid, fpr_u, tpr_u))
        elif mode == "pr":
            prec, rec, _ = precision_recall_curve(yy, ss)
            rec_u, idx_u = np.unique(rec, return_index=True)
            prec_u = prec[idx_u]
            collects.append(np.interp(grid, rec_u, prec_u))
        else:
            raise ValueError("mode must be 'roc' or 'pr'")
    if len(collects) == 0:
        return grid, None, None, None
    M = np.vstack(collects)
    mean = np.nanmean(M, axis=0)
    q25  = np.nanpercentile(M, 25, axis=0)
    q75  = np.nanpercentile(M, 75, axis=0)
    return grid, mean, q25, q75

def _summ_stats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=np.nan, median=np.nan, iqr=np.nan, min=np.nan, max=np.nan)
    q25, q75 = np.percentile(x, [25, 75])
    return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                iqr=float(q75 - q25), min=float(np.min(x)), max=float(np.max(x)))

# ===== 0) Text summary block (outer unbiased + fold means) =====
summary_outer = {}
if has_binary:
    y_t = y_true[mask]
    y_p = y_prob[mask]
    y_hat = y_pred[mask]

    # Pooled AUC and AP
    summary_outer["AUC_outer"] = roc_auc_score(y_t, y_p)
    summary_outer["AP_outer"]  = average_precision_score(y_t, y_p)

    # --- Metrics at fixed threshold 0.5 (primary, matches y_pred) ---
    summary_outer["ACC_thr05"] = accuracy_score(y_t, y_hat)
    tn, fp, fn, tp = confusion_matrix(y_t, y_hat).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    summary_outer["BACC_thr05"] = 0.5 * (sens + spec)

    # For convenience: alias as ACC_outer / BACC_outer (used later)
    summary_outer["ACC_outer"]  = summary_outer["ACC_thr05"]
    summary_outer["BACC_outer"] = summary_outer["BACC_thr05"]

    # --- Metrics at Youden threshold on pooled outer scores (post-hoc, optimistic) ---
    thr_youden_outer = youden_threshold_oof(y_t, y_p)
    y_pred_youden    = (y_p >= thr_youden_outer).astype(int)
    summary_outer["thr_youden_outer"] = float(thr_youden_outer)
    summary_outer["ACC_youden_outer"] = accuracy_score(y_t, y_pred_youden)

    tn_y, fp_y, fn_y, tp_y = confusion_matrix(y_t, y_pred_youden).ravel()
    sens_y = tp_y / max(tp_y + fn_y, 1)
    spec_y = tn_y / max(tn_y + fp_y, 1)
    summary_outer["BACC_youden_outer"] = 0.5 * (sens_y + spec_y)

    summary_outer["n_subjects"] = int(mask.sum())
else:
    summary_outer = {
        "AUC_outer": np.nan,
        "AP_outer":  np.nan,
        "ACC_thr05": np.nan,
        "BACC_thr05": np.nan,
        "ACC_outer": np.nan,
        "BACC_outer": np.nan,
        "thr_youden_outer": np.nan,
        "ACC_youden_outer": np.nan,
        "BACC_youden_outer": np.nan,
        "n_subjects": int(mask.sum()),
    }

# Per-fold means (whatever columns are available)
means_fold = {}
for col in ["auc_train","auc_test","bacc_train","bacc_test",
            "logloss_train","logloss_test"]:
    if col in fold_df.columns:
        means_fold[col] = float(np.nanmean(fold_df[col]))
    else:
        means_fold[col] = np.nan

print("=== Outer LOSO (pooled subjects, thr=0.5) ===")
for k in ["AUC_outer","AP_outer","ACC_outer","BACC_outer","n_subjects"]:
    val = summary_outer.get(k, np.nan)
    if isinstance(val, (int, float)):
        if isinstance(val, int) and k == "n_subjects":
            print(f"{k}: {val}")
        else:
            print(f"{k}: {val:.3f}")
    else:
        print(f"{k}: {val}")

print("\n=== Outer LOSO (pooled, Youden threshold) ===")
for k in ["thr_youden_outer","ACC_youden_outer","BACC_youden_outer"]:
    val = summary_outer.get(k, np.nan)
    if isinstance(val, (int, float)):
        print(f"{k}: {val:.3f}")
    else:
        print(f"{k}: {val}")

print("\n=== Per-fold means ===")
for k,v in means_fold.items():
    if isinstance(v, float) and not np.isnan(v):
        print(f"{k}: {v:.3f}")
    else:
        print(f"{k}: {v}")

# ===== 1) ROC curve (outer pooled) with optional subject-bootstrap mean + IQR =====
fig, ax = plt.subplots(figsize=(6.0, 4.8), dpi=120)
if has_binary:
    fpr, tpr, _ = roc_curve(y_true[mask], y_prob[mask])
    auc_val = roc_auc_score(y_true[mask], y_prob[mask])

    if USE_BOOTSTRAP:
        grid, mean_tpr, tpr_q25, tpr_q75 = _bootstrap_curve_stats(
            y_true[mask], y_prob[mask],
            n_boot=N_BOOT, seed=int(globals().get("SEED", 42)),
            grid=np.linspace(0,1,201), mode="roc"
        )
        if mean_tpr is not None:
            ax.fill_between(grid, tpr_q25, tpr_q75, alpha=0.2,
                            label="IQR band (subject bootstrap)")
            ax.plot(grid, mean_tpr, lw=2, label="Mean ROC (subject bootstrap)")

    ax.plot(fpr, tpr, lw=1, alpha=0.8, label=f"Pooled ROC (AUC={auc_val:.3f})")
    ax.plot([0,1],[0,1], "--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{TITLE_PREFIX} | ROC (outer LOSO pooled)")
    ax.legend(loc="lower right")
else:
    ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, f"roc_outer_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
            dpi=300, bbox_inches="tight")
if _CAN_DISPLAY: plt.show()
else: plt.close(fig)

# ===== 2) Precision–Recall curve (outer pooled) with optional subject-bootstrap mean + IQR =====
fig, ax = plt.subplots(figsize=(6.0, 4.8), dpi=120)
if has_binary:
    precision, recall, _ = precision_recall_curve(y_true[mask], y_prob[mask])
    ap = average_precision_score(y_true[mask], y_prob[mask])

    if USE_BOOTSTRAP:
        grid, mean_prec, prec_q25, prec_q75 = _bootstrap_curve_stats(
            y_true[mask], y_prob[mask],
            n_boot=N_BOOT, seed=int(globals().get("SEED", 42)),
            grid=np.linspace(0,1,201), mode="pr"
        )
        if mean_prec is not None:
            ax.fill_between(grid, prec_q25, prec_q75, alpha=0.2,
                            label="IQR band (subject bootstrap)")
            ax.plot(grid, mean_prec, lw=2, label="Mean PR (subject bootstrap)")

    ax.plot(recall, precision, lw=1, alpha=0.8, label=f"Pooled PR (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{TITLE_PREFIX} | Precision–Recall (outer LOSO pooled)")
    ax.legend(loc="lower left")
else:
    ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, f"pr_outer_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
            dpi=300, bbox_inches="tight")
if _CAN_DISPLAY: plt.show()
else: plt.close(fig)

# ===== 3) Confusion matrix (outer pooled decision) =====
if has_binary:
    labels = [0, 1]
    cm = confusion_matrix(y_true[mask], y_pred[mask], labels=labels)
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=120)
    im = ax.imshow(cm_norm, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels([CLASS_NAMES.get(0,"Class 0"), CLASS_NAMES.get(1,"Class 1")])
    ax.set_yticklabels([CLASS_NAMES.get(0,"Class 0"), CLASS_NAMES.get(1,"Class 1")])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{TITLE_PREFIX} | Confusion Matrix (outer LOSO pooled)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "black")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"cm_outer_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)

# ===== 4) Per-fold AUC (train vs test) =====
if {"auc_train","fold"}.issubset(fold_df.columns):
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=120)
    ax.plot(fold_df["fold"], fold_df["auc_train"], marker="o", label="Train AUC (OOF)")
    if "auc_test" in fold_df.columns and not fold_df["auc_test"].isna().all():
        ax.plot(fold_df["fold"], fold_df["auc_test"], marker="o", label="Test AUC (held-out)")
    ax.set_xlabel("Outer fold"); ax.set_ylabel("AUC"); ax.set_ylim(0,1.05)
    ax.set_title(f"{TITLE_PREFIX} | Per-fold AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"per_fold_auc_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)

# ===== 5) Per-fold #selected features =====
if {"fold","n_selected"}.issubset(fold_df.columns):
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=120)
    ax.bar(fold_df["fold"], fold_df["n_selected"])
    ax.set_xlabel("Outer fold"); ax.set_ylabel("# selected features")
    ax.set_title(f"{TITLE_PREFIX} | Selected features per fold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"per_fold_n_features_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)

# ===== 6) Feature stability histogram + top-k barh =====
fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=120)
ax.hist(stab_df["fold_select_freq"].to_numpy(), bins=20, edgecolor="black")
ax.set_xlabel("Fold selection frequency"); ax.set_ylabel("Number of features")
ax.set_title(f"{TITLE_PREFIX} | Stability frequency histogram")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, f"stab_hist_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
            dpi=300, bbox_inches="tight")
if _CAN_DISPLAY: plt.show()
else: plt.close(fig)

TOP_K = min(20, stab_df.shape[0])
top_df_plot = stab_df.head(TOP_K)[["feature","fold_select_freq"]].iloc[::-1]
fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=120)
ax.barh(top_df_plot["feature"], top_df_plot["fold_select_freq"])
ax.set_xlabel("Fold selection frequency")
ax.set_title(f"{TITLE_PREFIX} | Top-{TOP_K} stable features")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, f"top{TOP_K}_features_barh_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
            dpi=300, bbox_inches="tight")
if _CAN_DISPLAY: plt.show()
else: plt.close(fig)

# ===== 7) Accuracy and per-class recall per fold =====
per_fold = []
if "left_out_idx" in fold_df.columns and "left_out_idx" in outer_preds.columns:
    for _, row in fold_df.iterrows():
        idx = int(row["left_out_idx"])
        m = (outer_preds["left_out_idx"] == idx)
        if m.any():
            yt = int(outer_preds.loc[m, "y_true"].iloc[0])
            yp = int(outer_preds.loc[m, "y_pred"].iloc[0])
            acc = float(yt == yp)
            sens = 1.0 if (yt == 1 and yp == 1) else (0.0 if yt == 1 else np.nan)
            spec = 1.0 if (yt == 0 and yp == 0) else (0.0 if yt == 0 else np.nan)
            per_fold.append({"fold": int(row["fold"]), "acc": acc, "sens": sens, "spec": spec})
per_fold_df = pd.DataFrame(per_fold).sort_values("fold")

if not per_fold_df.empty:
    # Accuracy per fold
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=120)
    ax.plot(per_fold_df["fold"], per_fold_df["acc"], marker="o")
    ax.set_xlabel("Outer fold"); ax.set_ylabel("Accuracy"); ax.set_ylim(-0.05,1.05)
    ax.set_title(f"{TITLE_PREFIX} | Accuracy per fold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"per_fold_acc_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)

    # Sensitivity & specificity per fold
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=120)
    ax.plot(per_fold_df["fold"], per_fold_df["sens"], marker="o", label="Sensitivity (Recall 1)")
    ax.plot(per_fold_df["fold"], per_fold_df["spec"], marker="o", label="Specificity (Recall 0)")
    ax.set_xlabel("Outer fold"); ax.set_ylabel("Recall"); ax.set_ylim(-0.05,1.05)
    ax.set_title(f"{TITLE_PREFIX} | Per-class recall per fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"per_fold_recall_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)

# ===== 8) Per-fold log-loss (train OOF vs test) =====
if {"fold","logloss_train","logloss_test"}.issubset(fold_df.columns):
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=120)
    ax.plot(fold_df["fold"], fold_df["logloss_train"], marker="o", label="Train LogLoss (OOF)")
    ax.plot(fold_df["fold"], fold_df["logloss_test"],  marker="o", label="Test LogLoss (held-out)")
    ax.set_xlabel("Outer fold"); ax.set_ylabel("Log loss")
    ax.set_title(f"{TITLE_PREFIX} | Per-fold LogLoss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"per_fold_logloss_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)

# ===== 9) PFER reporting =====
if "pfer" in fold_df.columns:
    pfer_stats = _summ_stats(fold_df["pfer"])
    pfer_df = pd.DataFrame([{
        "PFER_mean":   pfer_stats["mean"],
        "PFER_median": pfer_stats["median"],
        "PFER_IQR":    pfer_stats["iqr"],
        "PFER_min":    pfer_stats["min"],
        "PFER_max":    pfer_stats["max"],
    }])
    print("=== PFER summary across outer folds (Meinshausen–Bühlmann bound) ===")
    if _CAN_DISPLAY:
        display(pfer_df)
    pfer_df.to_csv(os.path.join(OUT_DIR, f"pfer_summary_{CLUSTER_LABEL}_{TARGET_LABEL}.csv"),
                   index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=120)
    vals = np.asarray(fold_df["pfer"], float)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        ax.hist(vals, bins=min(20, max(5, int(np.sqrt(vals.size)))), edgecolor="black")
        ax.set_xlabel("PFER bound"); ax.set_ylabel("# folds")
        ax.set_title(f"{TITLE_PREFIX} | PFER bound (per fold)")
    else:
        ax.text(0.5, 0.5, "No finite PFER values", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"pfer_hist_{CLUSTER_LABEL}_{TARGET_LABEL}.png"),
                dpi=300, bbox_inches="tight")
    if _CAN_DISPLAY: plt.show()
    else: plt.close(fig)

# ===== 10) Compact overall summary table =====
overall = {
    "AUC_outer":  summary_outer.get("AUC_outer", np.nan),
    "AP_outer":   summary_outer.get("AP_outer", np.nan),
    "ACC_outer":  summary_outer.get("ACC_outer", np.nan),
    "BACC_outer": summary_outer.get("BACC_outer", np.nan),
    "thr_youden_outer": summary_outer.get("thr_youden_outer", np.nan),
    "ACC_youden_outer": summary_outer.get("ACC_youden_outer", np.nan),
    "BACC_youden_outer": summary_outer.get("BACC_youden_outer", np.nan),
    "mean_auc_train":  means_fold.get("auc_train", np.nan),
    "mean_auc_test":   means_fold.get("auc_test", np.nan),
    "mean_bacc_train": means_fold.get("bacc_train", np.nan),
    "mean_bacc_test":  means_fold.get("bacc_test", np.nan),
    "mean_logloss_train": means_fold.get("logloss_train", np.nan),
    "mean_logloss_test":  means_fold.get("logloss_test", np.nan),
    "n_subjects": summary_outer.get("n_subjects", np.nan),
}
overall_df = pd.DataFrame([overall])
print("=== Overall summary ===")
if _CAN_DISPLAY:
    display(overall_df)
overall_df.to_csv(os.path.join(OUT_DIR, f"overall_summary_{CLUSTER_LABEL}_{TARGET_LABEL}.csv"),
                  index=False)


# In[98]:


#============== Cell 17: Features plot ===============
import os
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Patch

# Output dir (same convention as above)
OUT_DIR = globals().get("out_dir", ".")
os.makedirs(OUT_DIR, exist_ok=True)

# Try to detect if we can display
try:
    from IPython.display import display  # noqa
    _CAN_DISPLAY = True
except Exception:
    _CAN_DISPLAY = False

CLUSTER_LABEL = str(globals().get("CLUSTER", "cluster"))
TARGET_LABEL  = str(globals().get("TARGET", "target"))

def make_pretty_label(feat):
    f = str(feat)
    # --- Replace TAR and S̄ ---
    f = f.replace("powermean_Theta/powermean_Alpha", "TAR")
    f = f.replace("signalmean", r"$\overline{S}$")
    f = f.replace("_|", "|")
    return f

# ---- Build top-K table from stab_df ----
TOP_K = min(20, stab_df.shape[0])
top_df = (
    stab_df
    .head(TOP_K)[["feature", "fold_select_freq", "coef_mean", "coef_abs_mean"]]
    .iloc[::-1]
    .copy()
)

# Pretty labels for y-axis
x_labels = [make_pretty_label(f) for f in top_df["feature"]]

coef_vals = top_df["coef_mean"].to_numpy()   # signed coefficients
abs_vals  = np.abs(coef_vals)

vmax_raw = float(np.nanmax(abs_vals)) if np.isfinite(np.nanmax(abs_vals)) else 0.0
if vmax_raw == 0.0:
    vmax_raw = 1.0

mapped = abs_vals / vmax_raw                  # 0..1
cmap   = cm.get_cmap("Purples")
cols   = cmap(mapped)
norm01 = colors.Normalize(vmin=0.0, vmax=1.0)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
bars = ax.barh(
    x_labels,
    top_df["fold_select_freq"],
    color=cols,
    edgecolor="black",
    linewidth=0.8
)

ax.set_xlabel("Fold selection frequency", fontsize=16, fontweight="bold")
ax.set_title(f"{CLUSTER_LABEL} | {TARGET_LABEL} | Top-{TOP_K} stable features", 
             fontsize=18, fontweight="bold")
ax.grid(axis="x", alpha=0.2)

# --- Hatch overlay for negative coefficients ---
for bar, c in zip(bars, coef_vals):
    if c < 0:
        ax.barh(
            y=bar.get_y() + bar.get_height() / 2,
            width=bar.get_width(),
            height=bar.get_height(),
            left=bar.get_x(),
            align="center",
            facecolor="none",
            edgecolor="orange",
            linewidth=0.0,
            hatch="//",
        )

# Colorbar for |coef| normalized 0–1
sm = cm.ScalarMappable(norm=norm01, cmap=cmap)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("|Coefficient| (normalized)", fontsize=16, fontweight="bold")
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.ax.tick_params(labelsize=14)
for tick in cb.ax.get_yticklabels():
    tick.set_fontweight("bold")

# Legend for sign meaning
legend_patches = [
    Patch(facecolor=cmap(0.7), edgecolor="black", label="Positive coef"),
    Patch(facecolor=cmap(0.8), edgecolor="orange", hatch="//", label="Negative coef"),
]
ax.legend(handles=legend_patches, loc="lower right", frameon=False)

# Axis formatting
ax.tick_params(axis='x', labelsize=16)
for ticklabel in ax.get_xticklabels():
    ticklabel.set_fontweight("bold")

ax.tick_params(axis='y', labelsize=12)
for ticklabel in ax.get_yticklabels():
    ticklabel.set_fontweight("bold")

fig.tight_layout()

# ---- Save and (optionally) display ----
fig_path = os.path.join(OUT_DIR, f"top{TOP_K}_features_fancy_{CLUSTER_LABEL}_{TARGET_LABEL}.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Saved feature plot to: {fig_path}")

if _CAN_DISPLAY:
    plt.show()
else:
    plt.close(fig)


# In[ ]:




