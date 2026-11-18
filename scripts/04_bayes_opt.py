"""Step 4 - Bayesian hyperparameter optimisation of the selected model (LightGBM).

Search      : Optuna TPE sampler (Bayesian: models good vs bad regions of the space)
Objective   : mean observation-level Top-1 accuracy over 5-fold GroupKFold by trip
              (same folds, rows and features as 03_train_models.py)
Pruning     : median pruner after each fold, so clearly worse trials stop early
Data        : train partition only; student_test is used once at the end to compare
              the tuned model with the untuned one. The evaluator partition is never used.

Outputs
  results/bayes_opt_trials.csv, results/bayes_opt_best_params.json,
  results/bayes_opt_comparison.csv, results/figures/bayes_opt_*.png,
  artifacts/lightgbm_tuned.joblib
"""
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from _paths import ARTIFACTS, FIGURES, PROCESSED, RESULTS, SEED
from pathsnap.config import EXCLUDED_TRAIN_TRIPS, TRAIN_POINT_STRIDE
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.metrics import candidate_metrics, top1_accuracy, top1_table

N_TRIALS = 30
UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8)


def make_model(params: dict) -> LGBMClassifier:
    return LGBMClassifier(**{"subsample_freq": 1, **params}, n_jobs=16, random_state=SEED, verbose=-1)


def search_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def main():
    cands = pd.read_pickle(PROCESSED / "candidates.pkl")
    points = pd.read_pickle(PROCESSED / "points.pkl")
    is_train = (cands.split == "train") & ~cands.trip_id.isin(EXCLUDED_TRAIN_TRIPS)
    train_df = cands[is_train & (cands.obs % TRAIN_POINT_STRIDE == 0)].reset_index(drop=True)
    test_df = cands[cands.split == "student_test"].reset_index(drop=True)
    tr_points = points[(points.split == "train") & ~points.trip_id.isin(EXCLUDED_TRAIN_TRIPS)
                       & (points.obs % TRAIN_POINT_STRIDE == 0)]
    n_test_points = int((points.split == "student_test").sum())

    X, y = train_df[FEATURE_COLUMNS], train_df.label.to_numpy()
    splits = list(GroupKFold(n_splits=5).split(train_df, groups=train_df.trip_id))
    val_points = [int(tr_points.trip_id.isin(train_df.trip_id.iloc[va].unique()).sum()) for _, va in splits]

    def cv_scores(params: dict, trial: optuna.Trial | None = None) -> dict:
        top1s, f1s, lls = [], [], []
        for f, (tr, va) in enumerate(splits):
            m = make_model(params).fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[va])[:, 1]
            top1s.append(top1_accuracy(top1_table(train_df.iloc[va], p), val_points[f])[0])
            met = candidate_metrics(y[va], p)
            f1s.append(met["macro_f1"]); lls.append(met["log_loss"])
            if trial is not None:
                trial.report(float(np.mean(top1s)), step=f)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return {"cv_top1": float(np.mean(top1s)), "cv_macro_f1": float(np.mean(f1s)),
                "cv_macro_f1_std": float(np.std(f1s, ddof=1)), "cv_log_loss": float(np.mean(lls))}

    def objective(trial: optuna.Trial) -> float:
        s = cv_scores(search_space(trial), trial)
        for k, v in s.items():
            trial.set_user_attr(k, v)
        return s["cv_top1"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="pathsnap_lightgbm",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    study.enqueue_trial({**{k: v for k, v in UNTUNED.items() if k != "subsample_freq"},
                         "min_child_samples": 20, "reg_alpha": 1e-3, "reg_lambda": 1e-3})  # untuned start point
    t0 = time.time()

    def log(st, tr):
        print(f"trial {tr.number:2d} {tr.state.name:8s} value={tr.value if tr.value is not None else float('nan'):.4f} "
              f"best={st.best_value:.4f} elapsed={time.time() - t0:.0f}s", flush=True)

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[log])
    search_s = time.time() - t0

    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    trials["duration_s"] = trials.pop("duration").dt.total_seconds()
    trials.to_csv(RESULTS / "bayes_opt_trials.csv", index=False)
    best = study.best_trial
    best_params = {**best.params, "subsample_freq": 1}
    (RESULTS / "bayes_opt_best_params.json").write_text(json.dumps({
        "objective": "mean 5-fold GroupKFold observation-level Top-1 accuracy (train partition)",
        "sampler": "Optuna TPESampler(seed=42, n_startup_trials=8)", "pruner": "MedianPruner(n_startup_trials=8, n_warmup_steps=1)",
        "n_trials": len(study.trials), "n_pruned": int(sum(t.state.name == "PRUNED" for t in study.trials)),
        "search_seconds": round(search_s, 1), "best_trial": best.number, "best_cv_top1": best.value,
        "best_params": best_params, "best_cv_scores": best.user_attrs,
    }, indent=2), encoding="utf-8")

    # ------------------------------------------------ untuned vs tuned, same data and metrics
    untuned_cv = cv_scores({**UNTUNED})
    rows = []
    for name, params, cv in (("LightGBM untuned", UNTUNED, untuned_cv), ("LightGBM tuned", best_params, best.user_attrs)):
        t1 = time.time(); m = make_model(params).fit(X, y); fit_s = time.time() - t1
        t1 = time.time(); p = m.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]; inf_s = time.time() - t1
        te = candidate_metrics(test_df.label.to_numpy(), p)
        t1u = top1_accuracy(top1_table(test_df, p), n_test_points)
        rows.append({"model": name, "cv_macro_f1": cv["cv_macro_f1"], "cv_top1": cv["cv_top1"], "cv_log_loss": cv["cv_log_loss"],
                     "test_top1": t1u[0], "test_top1_undirected": t1u[1], "test_macro_f1": te["macro_f1"],
                     "test_log_loss": te["log_loss"], "fit_time_s": fit_s, "infer_time_s": inf_s,
                     "infer_ms_per_1k_points": inf_s / n_test_points * 1e6})
        if name == "LightGBM tuned":
            joblib.dump({"model": m, "features": FEATURE_COLUMNS, "params": params}, ARTIFACTS / "lightgbm_tuned.joblib")
    comp = pd.DataFrame(rows)
    comp.to_csv(RESULTS / "bayes_opt_comparison.csv", index=False)

    # ------------------------------------------------ plots
    done = trials[trials.state == "COMPLETE"].sort_values("number")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].scatter(done.number, done.value, color="#4575b4", label="completed trial")
    pr = trials[trials.state == "PRUNED"]
    axes[0].scatter(pr.number, [done.value.min()] * len(pr), marker="x", color="#999", label="pruned (stopped early)")
    axes[0].plot(done.number, done.value.cummax(), color="#d73027", label="best so far")
    axes[0].set_xlabel("trial"); axes[0].set_ylabel("CV Top-1 accuracy"); axes[0].legend()
    axes[0].set_title("Bayesian optimisation history (TPE, GroupKFold CV on train)")
    try:
        imp = optuna.importance.get_param_importances(study)
        axes[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="#2166ac")
        axes[1].set_title("Hyperparameter importance (fANOVA)")
    except Exception as exc:  # importance needs enough completed trials
        axes[1].text(0.1, 0.5, f"importance not available: {exc}")
    fig.tight_layout(); fig.savefig(FIGURES / "bayes_opt_history.png", dpi=130); plt.close(fig)

    print(json.dumps(best_params, indent=2))
    pd.set_option("display.width", 220)
    print(comp.round(4).to_string(index=False))
    print(f"search took {search_s:.0f}s")


if __name__ == "__main__":
    main()
