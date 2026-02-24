import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from preprocessing import clean_text


def main():
    print("Loading dataset...")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "reduced_dataset.csv")

    df = pd.read_csv(data_path)

    print("Cleaning text...")
    df["text"] = df["text"].apply(clean_text)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Baseline Results ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()