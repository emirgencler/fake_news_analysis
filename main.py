from data_loader import load_data
from feature_extraction import clean_text_for_linguistic, extract_stylistic_features

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import hstack

from sklearn.preprocessing import StandardScaler
import pandas as pd




if __name__ == "__main__":
    df = load_data("data/Fake.csv", "data/True.csv")

    df["raw_text"] = df["text"].astype(str)

    x = df["raw_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=0,
        stratify=y
    )

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    # Linguistic features
    X_train_clean = X_train.apply(clean_text_for_linguistic)
    X_test_clean = X_test.apply(clean_text_for_linguistic)

    tfidf = TfidfVectorizer(max_features=5000,stop_words="english",ngram_range=(1, 2))

    X_train_linguistic = tfidf.fit_transform(X_train_clean)
    X_test_linguistic = tfidf.transform(X_test_clean)

    print("Linguistic train shape:", X_train_linguistic.shape)
    print("Linguistic test shape:", X_test_linguistic.shape)

    # Stylistic features
    X_train_stylistic = extract_stylistic_features(X_train)
    X_test_stylistic = extract_stylistic_features(X_test)

    # Scale stylistic features for better interpretation
    scaler = StandardScaler()
    X_train_stylistic_scaled = scaler.fit_transform(X_train_stylistic)
    X_test_stylistic_scaled = scaler.transform(X_test_stylistic)


    def evaluate_model(name, y_true, y_pred):
        print(f"\n{name} Results")
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Precision:", precision_score(y_true, y_pred))
        print("Recall:", recall_score(y_true, y_pred))
        print("F1-score:", f1_score(y_true, y_pred))


    # Model 1: Linguistic features only

    linguistic_model = LogisticRegression(max_iter=1000)
    linguistic_model.fit(X_train_linguistic, y_train)

    # Most important words for the linguistic model
    feature_names = tfidf.get_feature_names_out()
    coefficients = linguistic_model.coef_[0]

    feature_scores = list(zip(feature_names, coefficients))

    feature_scores.sort(key=lambda x: x[1])

    top_real_words = feature_scores[:10]
    top_fake_words = feature_scores[-10:]

    print("\nTop words for Fake News:")
    for word, score in reversed(top_fake_words):
        print(word, score)

    print("\nTop words for Real News:")
    for word, score in top_real_words:
        print(word, score)

    linguistic_predictions = linguistic_model.predict(X_test_linguistic)

    evaluate_model("Linguistic Features", y_test, linguistic_predictions)

    # Model 2: Stylistic features only
    stylistic_model = LogisticRegression(max_iter=1000)
    stylistic_model.fit(X_train_stylistic_scaled, y_train)
    stylistic_predictions = stylistic_model.predict(X_test_stylistic_scaled)

    evaluate_model("Stylistic Features", y_test, stylistic_predictions)

    # Stylistic feature importance
    stylistic_importance = pd.DataFrame({
        "Feature": X_train_stylistic.columns,
        "Coefficient": stylistic_model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    print("\nStylistic Feature Importance:")
    print(stylistic_importance)


    # Model 3: Combined features
    X_train_combined = hstack([X_train_linguistic, X_train_stylistic_scaled])
    X_test_combined = hstack([X_test_linguistic, X_test_stylistic_scaled])

    combined_model = LogisticRegression(max_iter=1000)
    combined_model.fit(X_train_combined, y_train)
    combined_predictions = combined_model.predict(X_test_combined)
    evaluate_model("Combined Features", y_test, combined_predictions)

    # Combined feature importance
    combined_feature_names = list(tfidf.get_feature_names_out()) + list(X_train_stylistic.columns)

    combined_importance = pd.DataFrame({
        "Feature": combined_feature_names,
        "Coefficient": combined_model.coef_[0]
    })

    top_combined_fake = combined_importance.sort_values(by="Coefficient", ascending=False).head(15)
    top_combined_real = combined_importance.sort_values(by="Coefficient", ascending=True).head(15)

    print("\nTop Combined Features for Fake News:")
    print(top_combined_fake)

    print("\nTop Combined Features for Real News:")
    print(top_combined_real)