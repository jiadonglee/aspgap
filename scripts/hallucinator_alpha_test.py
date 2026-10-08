#!/usr/bin/env python
"""[alpha/M]-focused control-vs-treatment test of the v1-latent 'hallucinator' distillation head.

Arms (3 seeds each, identical but for model.latent_head): control = aspgap110_aug_rand (seed 42 is the
pre-existing aspgap110_aug_rand_r2, seeds 1,2 are hall_ctrl_s*), treatment = hall_v1lat_s{42,1,2}.
Everything is computed from saved predictions_<split>.npz (native G, raw units), metrics_degraded_*.csv
and summary.md of every run. Truth/masks come from APOGEEDataset (as compare_subsets.py).

Metrics per (split, subset): alpha RMSE and 1.4826*MAD (about the median residual); [M/H] bins;
giants/dwarfs; faint_xp; never-in-DR17; high/low-alpha sequence classification and bimodality in
-0.8<[M/H]<-0.1; Teff/logg/[M/H] RMSE; degraded-G' alpha RMSE/MAD; residual-recovery R2 from summary.md.
Statistic: arm mean +- std (ddof=1, 3 seeds), diff = treat - ctrl, z = diff/sqrt(sc^2/3+st^2/3),
plus a paired bootstrap over test stars of the seed-averaged predictions of the two arms.

Sequence definition: on TRAIN true labels with -0.8<[M/H]<-0.1, in 14 [M/H] bins find the KDE
minimum of the true [alpha/M] distribution between its two main modes; the dividing line is the
straight-line fit of those trough positions vs [M/H].
Classification uses the predicted [alpha/M] against the line evaluated at the TRUE [M/H] (so it
isolates [alpha/M] quality); the alternative with predicted [M/H] is also reported.
Bimodality: in two true-[M/H] windows (-0.8,-0.5) and (-0.5,-0.1), Gaussian KDE (bw 0.01) of the
predicted [alpha/M]; peak positions = modes of the TRUE-alpha KDE below/above the divider;
trough_ratio = min KDE between the peaks / min(KDE at the two peaks) (lower = more bimodal; the
truth value is reported alongside as trough_true).
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

MAIN = Path("/nexus/posix0/MIA-astro-env/hxr/jdli/aspgap_v3")
RUNS = MAIN / "runs"
ROOT = MAIN / "data/processed/aspcap_v0.6.0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aspgapv3.data.dataset import APOGEEDataset  # noqa: E402

LABELS = ["teff", "logg", "m_h", "alpha_m", "fe_h", "mg_h"]
ARMS = {"ctrl": ["aspgap110_aug_rand_r2", "hall_ctrl_s1", "hall_ctrl_s2"],
        "treat": ["hall_v1lat_s42", "hall_v1lat_s1", "hall_v1lat_s2"]}
GDEG = [15.0, 16.0, 17.0, 17.5]
A = 3  # alpha index
MH = 2
rng = np.random.default_rng(0)


def rmse(r):
    return float(np.sqrt(np.mean(r ** 2))) if len(r) else np.nan


def mad(r):
    return float(1.4826 * np.median(np.abs(r - np.median(r)))) if len(r) else np.nan


# ---------- sequences ----------
def fit_line(root):
    d = APOGEEDataset(root, "train", input="coef110n", labels=LABELS, noise=False, require_xp_valid=True)
    y, m = d.y_raw, d.mask
    s = m[:, MH] & m[:, A] & (y[:, MH] > -0.8) & (y[:, MH] < -0.1)
    X = np.c_[y[s, MH], y[s, A]]
    # per-[M/H]-bin trough of the train alpha distribution between the two sequences
    edges = np.linspace(-0.8, -0.1, 15)
    ag = np.linspace(-0.1, 0.4, 251)
    xs, ts = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = X[(X[:, 0] >= lo) & (X[:, 0] < hi), 1]
        k = kde(v, ag, bw=0.012)
        p1 = np.argmax(k)
        far = np.abs(ag - ag[p1]) > 0.07
        p2 = np.argmax(np.where(far, k, -1))
        a, b = sorted([p1, p2])
        t = a + np.argmin(k[a:b + 1])
        xs.append(0.5 * (lo + hi)); ts.append(ag[t])
    a1, a0 = np.polyfit(xs, ts, 1)
    print(f"[line] train n={s.sum()}; per-bin troughs {np.round(ts, 3).tolist()}; divider alpha = {a0:.4f} + {a1:.4f}*[M/H]")
    return a0, a1, dict(n_train=int(s.sum()), troughs=[float(t) for t in ts])


def kde(x, grid, bw=0.008, **_):
    out = np.zeros_like(grid)
    for i in range(0, len(x), 20000):
        out += np.exp(-0.5 * ((grid[:, None] - x[None, i:i + 20000]) / bw) ** 2).sum(1)
    return out / (len(x) * bw * np.sqrt(2 * np.pi))


AG = np.linspace(-0.1, 0.4, 251)
WINS = [(-0.8, -0.5), (-0.5, -0.1)]


def peaks_from_truth(y, m, line):
    """per window: true-alpha KDE modes below / above the divider at the window mid-[M/H]."""
    a0, a1 = line
    pk = []
    for lo, hi in WINS:
        s = m[:, MH] & m[:, A] & (y[:, MH] > lo) & (y[:, MH] < hi)
        k = kde(y[s, A], AG, bw=0.01)
        dv = a0 + a1 * 0.5 * (lo + hi)
        pk.append((AG[AG < dv][np.argmax(k[AG < dv])], AG[AG >= dv][np.argmax(k[AG >= dv])]))
    return pk


def trough_ratio(v, pk):
    k = kde(v, AG, bw=0.01)
    sel = (AG >= pk[0]) & (AG <= pk[1])
    kp = [k[np.argmin(np.abs(AG - p))] for p in pk]
    return float(k[sel].min() / min(kp))


# ---------- per-run metrics ----------
def run_metrics(mu, T, ctx, split, sel_never):
    """returns dict name->value for one run on one split."""
    y, m, cat = T["y"], T["m"], T["cat"]
    out = {}
    logg = y[:, 1]
    mh = y[:, MH]
    subsets = {"all": np.ones(len(y), bool)}
    for lo, hi, nm in [(-9, -1, "mh<-1"), (-1, -0.5, "-1<mh<-0.5"), (-0.5, 0, "-0.5<mh<0"), (0, 9, "mh>0")]:
        subsets[nm] = m[:, MH] & (mh >= lo) & (mh < hi)
    subsets["giants"] = m[:, 1] & (logg < 3.5)
    subsets["dwarfs"] = m[:, 1] & (logg >= 3.5)
    subsets["faint_xp"] = cat["is_faint_xp"].to_numpy(bool)
    allsub = dict(subsets)
    for k, v in subsets.items():
        allsub[f"never17|{k}"] = v & sel_never
    for sn, s in allsub.items():
        for k, lab in enumerate(LABELS):
            if lab not in ("alpha_m",) and sn != "all" and not sn.startswith("never17|all"):
                continue
            ss = s & m[:, k] & np.isfinite(mu[:, k])
            r = mu[ss, k] - y[ss, k]
            out[f"{lab}|{sn}|rmse"] = rmse(r)
            out[f"{lab}|{sn}|mad"] = mad(r)
            out[f"{lab}|{sn}|n"] = int(ss.sum())
    # sequences
    a0, a1 = ctx["line"]
    for tag, extra in [("seq", np.ones(len(y), bool)), ("never17|seq", sel_never)]:
        s = extra & m[:, MH] & m[:, A] & (mh > -0.8) & (mh < -0.1) & np.isfinite(mu[:, A])
        true_hi = y[s, A] > a0 + a1 * mh[s]
        pred_hi = mu[s, A] > a0 + a1 * mh[s]
        pred_hi2 = mu[s, A] > a0 + a1 * mu[s, MH]
        out[f"{tag}|acc"] = float((true_hi == pred_hi).mean())
        out[f"{tag}|acc_predmh"] = float((true_hi == pred_hi2).mean())
        out[f"{tag}|acc_hi"] = float(pred_hi[true_hi].mean())
        out[f"{tag}|acc_lo"] = float((~pred_hi[~true_hi]).mean())
        out[f"{tag}|n"] = int(s.sum())
        out[f"{tag}|d_rmse"] = rmse(mu[s, A] - y[s, A])
        for (lo, hi), pk in zip(WINS, ctx["peaks"]):
            w = s & (mh > lo) & (mh < hi)
            if w.sum() > 50:
                out[f"{tag}|trough_pred[{lo},{hi}]"] = trough_ratio(mu[w, A], pk)
                out[f"{tag}|trough_true[{lo},{hi}]"] = trough_ratio(y[w, A], pk)
    return out


def load_truth(root, split, dr17):
    d = APOGEEDataset(root, split, input="coef110n", labels=LABELS, noise=False, require_xp_valid=True)
    cat = d.catalog.merge(dr17, on="source_id", how="left")
    assert len(cat) == len(d.catalog)
    never = ~cat["in_dr17_any"].fillna(False).to_numpy(bool)
    return dict(y=d.y_raw, m=d.mask, cat=cat, sid=cat["source_id"].to_numpy()), never


def parse_summary(f):
    txt = Path(f).read_text()
    out = {}
    for lab in ("alpha_m", "mg_fe"):
        mm = re.search(rf"\*\*{lab}\*\*.*?residual recovery \(label-valid rows\): R² = ([-\d.]+)", txt, re.S)
        if mm:
            out[f"resid_recov_R2|{lab}"] = float(mm.group(1))
    return out


def degraded(run, split):
    f = RUNS / run / f"metrics_degraded_{split}.csv"
    df = pd.read_csv(f)
    out = {}
    for _, r in df[df.label.isin(LABELS)].iterrows():
        g = "native" if not np.isfinite(r.g_prime) else f"G'{r.g_prime:g}"
        out[f"{r.label}|deg_{g}|rmse"] = r.rmse
        out[f"{r.label}|deg_{g}|mad"] = r.scatter_mad
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS / "hallucinator_test"))
    ap.add_argument("--nboot", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dr17 = pd.read_parquet(RUNS / "aspgap_v1_ref" / "source_in_dr17.parquet")
    a0, a1, linfo = fit_line(str(ROOT))
    ctx = dict(line=(a0, a1))
    rows, boot_rows = [], []
    for split in ["iid_test", "sky_test"]:
        T, never = load_truth(str(ROOT), split, dr17)
        ctx["peaks"] = peaks_from_truth(T["y"], T["m"], (a0, a1))
        print(f"[{split}] true-d peaks {ctx['peaks']}")
        mus = {}
        for arm, runs in ARMS.items():
            mus[arm] = []
            for run in runs:
                z = np.load(RUNS / run / f"predictions_{split}.npz")
                assert np.array_equal(z["source_id"], T["sid"]), run
                mu = z["mu"]
                mus[arm].append(mu)
                res = run_metrics(mu, T, ctx, split, never)
                res.update(degraded(run, split))
                if split == "iid_test":
                    pass
                res.update(parse_summary(RUNS / run / "summary.md") if split == "iid_test" else {})
                for k, v in res.items():
                    rows.append(dict(split=split, run=run, arm=arm, metric=k, value=v))
        # paired bootstrap on seed-averaged predictions
        avg = {a: np.mean(mus[a], 0) for a in mus}
        N = len(T["y"])
        bidx = [rng.integers(0, N, N) for _ in range(args.nboot)]
        def sub_metrics(mu, idx):
            T2 = dict(y=T["y"][idx], m=T["m"][idx], cat=T["cat"].iloc[idx].reset_index(drop=True))
            r = run_metrics(mu[idx], T2, ctx, split, never[idx])
            return {k: v for k, v in r.items() if k.endswith(("|rmse", "|mad", "|acc", "|d_rmse")) or "trough_pred" in k}
        base = {a: sub_metrics(avg[a], np.arange(N)) for a in avg}
        bd = {k: [] for k in base["ctrl"]}
        for ii, idx in enumerate(bidx):
            rc, rt = sub_metrics(avg["ctrl"], idx), sub_metrics(avg["treat"], idx)
            for k in bd:
                bd[k].append(rt[k] - rc[k])
            if ii % 50 == 0:
                print(f"  boot {ii}/{args.nboot}", flush=True)
        for k, v in bd.items():
            v = np.array(v, float)
            boot_rows.append(dict(split=split, metric=k, avgpred_ctrl=base["ctrl"][k], avgpred_treat=base["treat"][k],
                                  boot_diff=base["treat"][k] - base["ctrl"][k], boot_se=np.nanstd(v),
                                  boot_p_lt0=float(np.nanmean(v < 0)), boot_z=(base["treat"][k] - base["ctrl"][k]) / (np.nanstd(v) + 1e-12)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics_per_run.csv", index=False)
    g = df.groupby(["split", "metric", "arm"]).value.agg(["mean", lambda x: x.std(ddof=1), "count"]).reset_index()
    g.columns = ["split", "metric", "arm", "mean", "std", "n"]
    c = g[g.arm == "ctrl"].set_index(["split", "metric"])
    t = g[g.arm == "treat"].set_index(["split", "metric"])
    m = c.join(t, lsuffix="_c", rsuffix="_t")
    m["diff"] = m["mean_t"] - m["mean_c"]
    m["z_seed"] = m["diff"] / np.sqrt(m["std_c"] ** 2 / 3 + m["std_t"] ** 2 / 3 + 1e-18)
    m = m.reset_index().drop(columns=["arm_c", "arm_t"])
    b = pd.DataFrame(boot_rows)
    m = m.merge(b, on=["split", "metric"], how="left")
    m.to_csv(out / "metrics.csv", index=False)
    (out / "line_info.txt").write_text(f"alpha = {a0:.5f} + {a1:.5f}*[M/H]; {linfo}\n")
    print("wrote", out / "metrics.csv")


if __name__ == "__main__":
    main()
