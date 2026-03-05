import json, argparse, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import language_tool_python

from utils import build_feature_matrix

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train essay score classifier")
    parser.add_argument("csv_path", help="Path to training CSV")
    parser.add_argument("--output_dir", default="./model", help="Directory to save model artifacts")
    parser.add_argument("--test_size", type=float, default=0.05, help="Held-out test fraction")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    print(f"  Rows: {len(df)}  |  Score distribution:\n{df['score'].value_counts().sort_index()}\n")

    lang_tool = language_tool_python.LanguageTool("en-US")

    X, feature_names = build_feature_matrix(df, lang_tool)
    y_raw = df["score"].values.astype(int)

    unique_labels = sorted(np.unique(y_raw))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    y = np.array([label_to_idx[v] for v in y_raw])
    num_classes = len(unique_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', class_weight='balanced', probability=True, C=1.0, gamma='scale', random_state=42)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Map predictions back to original labels for readable output
    y_test_orig = np.array([idx_to_label[v] for v in y_test])
    y_pred_orig = np.array([idx_to_label[v] for v in y_pred])

    print("eval results")
    print(classification_report(y_test_orig, y_pred_orig, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_orig, y_pred_orig))
    qwk = cohen_kappa_score(y_test_orig, y_pred_orig, weights="quadratic")
    print(f"\nQuadratic Weighted Kappa: {qwk:.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', class_weight='balanced', probability=True, C=1.0, gamma='scale', random_state=42)),
        ]),
        X, y, cv=cv, scoring="accuracy",
    )
    print(f"  Mean accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Save artifacts ───────────────────────────────────────────
    print(f"saved to {output_dir}")
    joblib.dump(model, str(output_dir / "svc_pipeline.pkl"))
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump({
            "label_to_idx": {int(k): int(v) for k, v in label_to_idx.items()},
            "idx_to_label": {int(k): int(v) for k, v in idx_to_label.items()},
        }, f)

    lang_tool.close()
    print("training complete")


if __name__ == "__main__":
    main()
