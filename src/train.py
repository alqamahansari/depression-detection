import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

from torch.utils.data import DataLoader

import pickle

# Setup base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from preprocessing import clean_text, build_vocab, text_to_sequence, pad_sequence
from dataset import TextDataset
from models.bilstm_model import BiLSTMModel


def main():

    device = torch.device("cpu")
    print("Using device:", device)

    # Load dataset
    data_path = os.path.join(BASE_DIR, "data", "reduced_dataset.csv")
    df = pd.read_csv(data_path)

    # Preprocessing
    df["text"] = df["text"].apply(clean_text)

    vocab = build_vocab(df["text"], max_vocab_size=10000)
    max_len = 200

    with open(os.path.join(BASE_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    print("Vocabulary saved as vocab.pkl")

    df["sequence"] = df["text"].apply(
        lambda x: pad_sequence(text_to_sequence(x, vocab), max_len)
    )

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        df["sequence"], df["label"], test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(TextDataset(list(X_train), list(y_train)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TextDataset(list(X_val), list(y_val)), batch_size=32)
    test_loader = DataLoader(TextDataset(list(X_test), list(y_test)), batch_size=32)

    # Model
    model = BiLSTMModel(vocab_size=len(vocab))
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    patience = 2
    best_val_loss = float("inf")
    counter = 0

    train_losses = []
    val_losses = []

    print("Starting training...")

    for epoch in range(epochs):

        # Training
        model.train()
        total_train_loss = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()

                preds = (outputs >= 0.5).float()
                all_val_preds.extend(preds.numpy())
                all_val_labels.extend(labels.numpy())

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        val_acc = accuracy_score(all_val_labels, all_val_preds)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "best_bilstm_model.pth"))
            print("Best model saved.")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "best_bilstm_model.pth")))
    model.eval()

    print("\nEvaluating on test set...")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()

            probs = outputs.numpy()
            preds = (outputs >= 0.5).float()

            all_probs.extend(probs)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:\n", cm)

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")

    # Loss Curve
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("loss_curve.png")

    print("\nPlots saved: roc_curve.png, loss_curve.png")


if __name__ == "__main__":
    main()