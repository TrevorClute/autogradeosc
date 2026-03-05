from flask import Flask, request, jsonify
import os, json, warnings
from dotenv import load_dotenv
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import language_tool_python

from model.utils import hand_crafted_features, embedding_distance_features

warnings.filterwarnings("ignore")

env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

app = Flask(__name__)
INTERNAL_API_KEY = os.getenv('FLASK_API_KEY')

PROMPT_MAP = {
    1: "Why would you be a good candidate for the program?",
    2: "How might you benefit from participation in the program?",
    3: "Give an example of your work on a group project. Describe your role, any successes, and how you handled any frustrations.",
    4: "Please look at the past student projects in the archives on this website and detail which ones are of interest to you and why."
}

MODEL_DIR = Path(__file__).resolve().parent / 'model'

svc_model = joblib.load(str(MODEL_DIR / "svc_pipeline.pkl"))


with open(MODEL_DIR / "label_mapping.json") as f:
    mapping = json.load(f)

idx_to_label = {int(k): int(v) for k, v in mapping["idx_to_label"].items()}
embedder = SentenceTransformer('all-MiniLM-L6-v2')
lang_tool = language_tool_python.LanguageTool("en-US")

def validate_request(func):
    def wrapper(*args, **kwargs):
        incoming_key = request.headers.get('X-Internal-Secret')
        if incoming_key != INTERNAL_API_KEY:
            return jsonify({"error": "Unauthorized System Access"}), 401
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


def evaluate_essay(essay_text, prompt_text):
    prompt_embed = embedder.encode(prompt_text)
    essay_embed = embedder.encode(essay_text)

    hc = hand_crafted_features(essay_text, lang_tool)
    dist = embedding_distance_features(prompt_embed, essay_embed)

    features = {**dist, **hc}
    X = np.array([list(features.values())], dtype=np.float32)

    pred_idx = svc_model.predict(X)[0]
    score = idx_to_label[int(pred_idx)]

    word_count = int(hc["word_count"])
    spelling_errors = round(hc["spelling_error_ratio"] * word_count)
    grammar_errors = round(hc["grammar_error_ratio"] * word_count)

    return {
        "score": score,
        "word_count": word_count,
        "spelling_errors": spelling_errors,
        "grammar_errors": grammar_errors,
    }


@app.route('/health')
def health():
    return "OK", 200


@app.route('/evaluate', methods=['POST'])
@validate_request
def analyze_essay():
    data = request.json
    essay_text = data.get('essay_text', '')
    prompt_id = data.get('prompt_id', 1)
    prompt_text = PROMPT_MAP.get(prompt_id, PROMPT_MAP[1])

    result = evaluate_essay(essay_text, prompt_text)
    return jsonify(result)


@app.route('/evaluate/batch', methods=['POST'])
@validate_request
def analyze_essays_batch():
    data = request.json
    essays = data.get('essays', [])

    if not essays:
        return jsonify({"error": "No essays provided"}), 400

    results = []
    for item in essays:
        essay_text = item.get('essay_text', '')
        prompt_id = item.get('prompt_id', 1)
        essay_id = item.get('essay_id', None)
        prompt_text = PROMPT_MAP.get(prompt_id, PROMPT_MAP[1])

        result = evaluate_essay(essay_text, prompt_text)
        result["essay_id"] = essay_id
        results.append(result)

    return jsonify({"results": results})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
