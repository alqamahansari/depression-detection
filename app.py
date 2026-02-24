import os
import sys
import torch
import pickle
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from models.bilstm_model import BiLSTMModel
from src.preprocessing import clean_text, text_to_sequence, pad_sequence

app = Flask(__name__)

# Load vocabulary
with open(os.path.join(BASE_DIR, "vocab.pkl"), "rb") as f:
    vocab = pickle.load(f)

# Load trained model
model = BiLSTMModel(vocab_size=len(vocab))
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "best_bilstm_model.pth")))
model.eval()

max_len = 200


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_text = data.get("text", "")

    if user_text.strip() == "":
        return jsonify({"error": "Empty input"}), 400

    text = clean_text(user_text)
    sequence = pad_sequence(text_to_sequence(text, vocab), max_len)
    input_tensor = torch.tensor([sequence], dtype=torch.long)

    with torch.no_grad():
        output = model(input_tensor).item()

    return jsonify({
        "probability": round(output * 100, 2),
        "prediction": "Depressed" if output >= 0.5 else "Not Depressed"
    })


if __name__ == "__main__":
    app.run()