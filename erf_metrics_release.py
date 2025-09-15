# -*- coding: utf-8 -*-
"""
erf_metrics_release.py
Version: 1.0 (2025-09-09)

Purpose
-------
This script reproduces the numeric evaluation reported in the manuscript for the
Enhanced Random Forest (eRF) classifier without generating any plots. It computes:

1) Overall metrics (Accuracy, macro-Precision, macro-Recall, macro-F1) for
   Train / Test and (optionally) Blind sets;
2) Per-class metrics for the 18 lithologies on Train / Test and (optionally) Blind:
   - One-vs-rest Accuracy (Acc_ovr)
   - Precision (macro over the single target class vs rest)
   - Recall   (macro over the single target class vs rest)
   - F1       (macro over the single target class vs rest)
3) Train vs Test one-vs-rest Accuracy comparison for each class.

No figures are produced. All outputs are CSV files in the specified output folder.

Reproducibility & Alignment with the Manuscript
-----------------------------------------------
- Features: five conventional logs, default names: GR, CNL, DEN, AC, RLA5
  (can be changed via --features).
- Preprocessing: Z-score standardization (fit on Train only).
- Imbalance handling: Borderline-SMOTE applied on Train only.
- Classifier: RandomForestClassifier with criterion='entropy' and max_features='sqrt'.
  Note: scikit-learn does not provide C4.5 gain-ratio; using 'entropy' is a practical
  proxy for information gain splits. The Kendall’s W stability logic is described
  methodologically in the manuscript; this script focuses on the numeric outputs.
- Train/Test partition:
  * If --split_col is provided and exists in the CSV, rows with split_col == 'train'
    form the training set and split_col == 'test' form the test set.
  * Else if --group is provided, a GroupShuffleSplit (by well) is used.
  * Else a stratified random split (by label) is used.
- Blind inference: an optional blind CSV is standardized with the scaler fit on Train
  and is evaluated if labels are present; otherwise predictions are saved only.

Usage examples
--------------
Example 1: grouped split by well, with blind evaluation
    python erf_metrics_release.py \
        --data all_wells.csv \
        --features GR CNL DEN AC RLA5 \
        --label litho \
        --group well_id \
        --blind_csv blind_wells.csv \
        --outdir outputs_metrics

Example 2: using a pre-defined split column in the main CSV
    # The CSV must contain a column 'split' with values 'train' or 'test'
    python erf_metrics_release.py \
        --data all_wells_with_split.csv \
        --features GR CNL DEN AC RLA5 \
        --label litho \
        --split_col split \
        --outdir outputs_metrics

Dependencies
------------
- Python 3.8+
- numpy, pandas, scikit-learn, imbalanced-learn

Outputs
-------
- metrics_overall.csv                  : overall metrics for Train / Test / (Blind if labels present)
- per_class_metrics_train.csv          : per-class metrics on Train
- per_class_metrics_test.csv           : per-class metrics on Test
- class_train_vs_test_ovr_accuracy.csv : per-class one-vs-rest Accuracy comparison (Train vs Test)
- per_class_metrics_blind.csv          : per-class metrics on Blind (if blind labels available)
- blind_predictions.csv                : per-sample predictions for Blind (if no labels in blind)

Author: Your research team
License: CC-BY 4.0 (or choose a license consistent with the journal's policy)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from imblearn.over_sampling import BorderlineSMOTE


# ------------------------------ Utility functions ------------------------------

def safe_read_csv(path: str) -> pd.DataFrame:
    """Read a CSV file with basic validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.shape[0] == 0:
        raise ValueError(f"Empty CSV: {path}")
    return df


def drop_na_rows(df: pd.DataFrame, cols: List[str], tag: str) -> pd.DataFrame:
    """Drop rows with NA in specified columns, reporting how many were removed."""
    before = df.shape[0]
    out = df.dropna(subset=cols)
    removed = before - out.shape[0]
    if removed > 0:
        print(f"[INFO] Dropped {removed} rows with NA in {cols} for {tag}. "
              f"Remaining: {out.shape[0]}", flush=True)
    return out


def split_train_test(df: pd.DataFrame, label_col: str,
                     group_col: Optional[str] = None,
                     split_col: Optional[str] = None,
                     test_size: float = 0.2,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a train/test split using one of the following strategies:
    1) If split_col is provided, it must contain 'train' and 'test' labels.
    2) Else if group_col is provided, use GroupShuffleSplit (by group).
    3) Else use stratified random split by label.
    """
    if split_col is not None and split_col in df.columns:
        tr = df[df[split_col].astype(str).str.lower() == "train"].copy()
        te = df[df[split_col].astype(str).str.lower() == "test"].copy()
        if tr.empty or te.empty:
            raise ValueError(f"[ERROR] split_col='{split_col}' must partition rows into 'train' and 'test'.")
        print(f"[INFO] Using predefined split column '{split_col}': train={tr.shape[0]}, test={te.shape[0]}", flush=True)
        return tr, te

    if group_col is not None and group_col in df.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        idx = np.arange(len(df))
        groups = df[group_col].values
        for train_idx, test_idx in gss.split(idx, groups=groups):
            tr = df.iloc[train_idx].copy()
            te = df.iloc[test_idx].copy()
            print(f"[INFO] Grouped split by '{group_col}': train={tr.shape[0]}, test={te.shape[0]}", flush=True)
            return tr, te

    tr, te = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[label_col]
    )
    print(f"[INFO] Stratified random split: train={tr.shape[0]}, test={te.shape[0]}", flush=True)
    return tr.copy(), te.copy()


def one_vs_rest_accuracy(y_true: np.ndarray, y_pred: np.ndarray, pos_label) -> float:
    """Compute one-vs-rest accuracy for a single class label."""
    y_true_bin = (y_true == pos_label).astype(int)
    y_pred_bin = (y_pred == pos_label).astype(int)
    return float((y_true_bin == y_pred_bin).mean())


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List) -> pd.DataFrame:
    """
    Compute per-class metrics in a one-vs-rest manner:
    - Acc_ovr: one-vs-rest accuracy
    - Prec, Rec, F1: computed with labels=[class] and average='macro' so that
      each class is evaluated vs the rest consistently.
    """
    rows = []
    for c in classes:
        row = {
            "class": c,
            "Acc_ovr": one_vs_rest_accuracy(y_true, y_pred, c),
            "Prec": precision_score(y_true, y_pred, labels=[c], average='macro', zero_division=0),
            "Rec": recall_score(y_true, y_pred, labels=[c], average='macro', zero_division=0),
            "F1": f1_score(y_true, y_pred, labels=[c], average='macro', zero_division=0),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute overall Accuracy, macro-Precision, macro-Recall, macro-F1."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


# ------------------------------ Core pipeline ------------------------------

def fit_erf_and_predict(train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        features: List[str],
                        label_col: str,
                        smote_kind: str = "borderline-1",
                        k_neighbors: int = 5,
                        m_neighbors: int = 10,
                        n_estimators: int = 300,
                        max_features: str = "sqrt",
                        random_state: int = 42) -> Dict:
    """
    Fit the eRF pipeline on training data and predict on train/test.

    Steps:
    - LabelEncode labels
    - Standardize features (fit on Train only)
    - Borderline-SMOTE on Train only
    - RandomForest (entropy) fit on rebalanced Train
    - Predict on standardized Train/Test
    """
    # Label encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(train_df[label_col].values)
    y_test_enc = le.transform(test_df[label_col].values)

    X_train = train_df[features].values
    X_test = test_df[features].values

    # Standardization (fit on Train only)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Imbalance handling (Train only)
    sm = BorderlineSMOTE(kind=smote_kind, k_neighbors=k_neighbors,
                         m_neighbors=m_neighbors, random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train_std, y_train_enc)

    # Enhanced RF core (entropy as proxy for information gain)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        criterion='entropy',
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_res, y_res)

    # Predictions (inverse transform to original labels)
    y_pred_train_lbl = le.inverse_transform(clf.predict(X_train_std))
    y_pred_test_lbl = le.inverse_transform(clf.predict(X_test_std))

    bundle = {
        "scaler": scaler,
        "clf": clf,
        "label_encoder": le,
        "class_names": list(le.classes_),
        "y_train_true": train_df[label_col].values,
        "y_train_pred": y_pred_train_lbl,
        "y_test_true": test_df[label_col].values,
        "y_test_pred": y_pred_test_lbl,
    }
    return bundle


def predict_blind(bundle: Dict,
                  blind_df: pd.DataFrame,
                  features: List[str],
                  label_col: Optional[str] = None) -> Dict:
    """Apply trained pipeline to a blind set; evaluate if labels are available."""
    scaler = bundle["scaler"]
    clf = bundle["clf"]
    le = bundle["label_encoder"]

    Xb = blind_df[features].values
    Xb_std = scaler.transform(Xb)
    yb_pred_lbl = le.inverse_transform(clf.predict(Xb_std))

    out = {"y_blind_pred": yb_pred_lbl}
    if label_col is not None and label_col in blind_df.columns:
        out["y_blind_true"] = blind_df[label_col].values
    return out


# ------------------------------ Main ------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute numeric metrics for eRF (Train/Test/(Blind)) without plotting."
    )
    ap.add_argument("--data", required=True, help="Main CSV containing Train/Test data.")
    ap.add_argument("--features", nargs="+", required=True,
                    help="Feature column names, e.g., GR CNL DEN AC RLA5")
    ap.add_argument("--label", required=True, help="Label column name.")
    ap.add_argument("--group", default=None, help="Group column (e.g., well_id) for grouped splitting.")
    ap.add_argument("--split_col", default=None,
                    help="Optional column marking pre-defined split: values must be 'train' or 'test'.")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test size fraction if splitting is required.")
    ap.add_argument("--blind_csv", default=None, help="Optional blind CSV for zero-shot evaluation.")

    # SMOTE
    ap.add_argument("--smote_kind", default="borderline-1", choices=["borderline-1", "borderline-2"],
                    help="Borderline-SMOTE variant.")
    ap.add_argument("--k_neighbors", type=int, default=5, help="SMOTE k_neighbors.")
    ap.add_argument("--m_neighbors", type=int, default=10, help="SMOTE m_neighbors.")

    # RF
    ap.add_argument("--n_estimators", type=int, default=300, help="Number of trees in RF.")
    ap.add_argument("--max_features", default="sqrt",
                    help="RF max_features (e.g., 'sqrt', 'log2', or an integer).")

    ap.add_argument("--outdir", default="outputs_metrics", help="Output directory.")
    ap.add_argument("--random_state", type=int, default=42, help="Random seed.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    df = safe_read_csv(args.data)

    # Validate columns
    for col in args.features + [args.label]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in data CSV.")

    # Drop NA rows in features/label
    df = drop_na_rows(df, args.features + [args.label], tag="main")

    # Create Train/Test split
    train_df, test_df = split_train_test(
        df, label_col=args.label, group_col=args.group, split_col=args.split_col,
        test_size=args.test_size, random_state=args.random_state
    )

    # Fit and predict
    bundle = fit_erf_and_predict(
        train_df, test_df, features=args.features, label_col=args.label,
        smote_kind=args.smote_kind, k_neighbors=args.k_neighbors, m_neighbors=args.m_neighbors,
        n_estimators=args.n_estimators, max_features=args.max_features,
        random_state=args.random_state
    )

    classes = bundle["class_names"]

    # Overall metrics (Train/Test)
    overall_train = overall_metrics(bundle["y_train_true"], bundle["y_train_pred"])
    overall_test = overall_metrics(bundle["y_test_true"], bundle["y_test_pred"])
    overall_df = pd.DataFrame([
        {"Split": "Train", **overall_train},
        {"Split": "Test", **overall_test},
    ])

    # Per-class (Train/Test)
    per_train = per_class_metrics(bundle["y_train_true"], bundle["y_train_pred"], classes)
    per_test = per_class_metrics(bundle["y_test_true"], bundle["y_test_pred"], classes)

    # Train vs Test one-vs-rest Accuracy comparison
    acc_comp = pd.DataFrame({
        "class": classes,
        "Acc_train_ovr": [one_vs_rest_accuracy(bundle["y_train_true"], bundle["y_train_pred"], c) for c in classes],
        "Acc_test_ovr":  [one_vs_rest_accuracy(bundle["y_test_true"],  bundle["y_test_pred"],  c) for c in classes],
    })

    # Blind evaluation (optional)
    per_blind = None
    if args.blind_csv is not None:
        blind_df = safe_read_csv(args.blind_csv)
        # ensure feature columns exist
        for col in args.features:
            if col not in blind_df.columns:
                raise KeyError(f"[Blind] Column '{col}' not found in blind CSV.")
        # drop NA
        blind_df = drop_na_rows(blind_df, args.features, tag="blind")
        bout = predict_blind(bundle, blind_df, args.features, label_col=args.label if args.label in blind_df.columns else None)

        if "y_blind_true" in bout:
            # Overall + per-class
            overall_blind = overall_metrics(bout["y_blind_true"], bout["y_blind_pred"])
            overall_df = pd.concat([overall_df, pd.DataFrame([{"Split": "Blind", **overall_blind}])],
                                   ignore_index=True)
            per_blind = per_class_metrics(bout["y_blind_true"], bout["y_blind_pred"], classes)
        else:
            # No labels in blind: save predictions only
            pd.DataFrame({"y_pred": bout["y_blind_pred"]}).to_csv(
                os.path.join(args.outdir, "blind_predictions.csv"), index=False, encoding="utf-8-sig"
            )
            print("[INFO] Blind CSV has no label column. Saved predictions only.", flush=True)

    # Save outputs
    overall_path = os.path.join(args.outdir, "metrics_overall.csv")
    per_train_path = os.path.join(args.outdir, "per_class_metrics_train.csv")
    per_test_path = os.path.join(args.outdir, "per_class_metrics_test.csv")
    acc_comp_path = os.path.join(args.outdir, "class_train_vs_test_ovr_accuracy.csv")

    overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")
    per_train.to_csv(per_train_path, index=False, encoding="utf-8-sig")
    per_test.to_csv(per_test_path, index=False, encoding="utf-8-sig")
    acc_comp.to_csv(acc_comp_path, index=False, encoding="utf-8-sig")

    if per_blind is not None:
        per_blind_path = os.path.join(args.outdir, "per_class_metrics_blind.csv")
        per_blind.to_csv(per_blind_path, index=False, encoding="utf-8-sig")

    print("[DONE] Saved CSVs:")
    for p in [overall_path, per_train_path, per_test_path, acc_comp_path]:
        print(" -", p)
    if per_blind is not None:
        print(" -", per_blind_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
