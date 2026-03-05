import sys, json, argparse, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import language_tool_python

from utils import (
    parse_embedding,
    hand_crafted_features,
    embedding_distance_features,
    build_feature_matrix,
)

warnings.filterwarnings("ignore")


def load_model(model_dir: str):
    model_dir = Path(model_dir)
    model = joblib.load(str(model_dir / "svc_pipeline.pkl"))
    with open(model_dir / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(model_dir / "label_mapping.json") as f:
        mapping = json.load(f)
    # JSON keys are always strings, convert back to int
    idx_to_label = {int(k): int(v) for k, v in mapping["idx_to_label"].items()}
    return model, feature_names, idx_to_label


def predict_single(
    essay_text: str,
    prompt_embed: np.ndarray,
    essay_embed: np.ndarray,
    model,
    idx_to_label,
    lang_tool,
):
    hc = hand_crafted_features(essay_text, lang_tool)
    dist = embedding_distance_features(prompt_embed, essay_embed)

    features = {**dist, **hc}
    X = np.array([list(features.values())], dtype=np.float32)

    pred_idx = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    original_label = idx_to_label[int(pred_idx)]
    label_proba = {idx_to_label[i]: round(float(p), 4) for i, p in enumerate(proba)}
    return original_label, label_proba


def predict_batch(csv_path: str, model, idx_to_label, lang_tool, output_path: str = None):
    print(f"Loading essays from {csv_path} …")
    df = pd.read_csv(csv_path)

    print("Building features …")
    X, _ = build_feature_matrix(df, lang_tool)

    print("Predicting …")
    pred_indices = model.predict(X)
    probas = model.predict_proba(X)

    # Map back to original labels
    df["predicted_score"] = [idx_to_label[int(i)] for i in pred_indices]
    for idx, original_label in sorted(idx_to_label.items()):
        df[f"prob_score_{original_label}"] = probas[:, idx].round(4)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✓ Predictions saved to {output_path}")

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Predict essay scores")
    parser.add_argument("--model_dir", default="./model", help="Directory containing saved model artifacts")
    parser.add_argument("--csv", default=None, help="Path to CSV of essays to score")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path (batch mode)")
    parser.add_argument("--essay", default=None, help="Essay text string (single mode)")
    parser.add_argument("--prompt_embed", default=None, help="Prompt embedding as string list (single mode)")
    parser.add_argument("--essay_embed", default=None, help="Essay embedding as string list (single mode)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_dir}/ …")
    model, feature_names, idx_to_label = load_model(args.model_dir)

    print("Initialising LanguageTool …")
    lang_tool = language_tool_python.LanguageTool("en-US")

    if args.csv:
        df = predict_batch(args.csv, model, idx_to_label, lang_tool, args.output)
        print(f"\nScore distribution:\n{df['predicted_score'].value_counts().sort_index()}")

    elif args.essay and args.prompt_embed and args.essay_embed:
        prompt_emb = parse_embedding(args.prompt_embed)
        essay_emb = parse_embedding(args.essay_embed)

        pred, proba = predict_single(
            args.essay, prompt_emb, essay_emb, model, idx_to_label, lang_tool
        )
        print(f"\n  Predicted score : {pred}")
        print(f"  Probabilities   : {proba}")

    else:
        print("Error: provide either --csv for batch mode, or --essay + --prompt_embed + --essay_embed for single mode.")
        sys.exit(1)

    lang_tool.close()
    print("\n✓ Done.")


if __name__ == "__main__":
    main()
