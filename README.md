# Fake News Detection using Linguistic and Stylistic Features

## Abstract

The rapid spread of misinformation on digital media platforms has increased the importance of automatic fake news detection systems. This project investigates the effectiveness of linguistic and stylistic features for fake news classification using Natural Language Processing (NLP) and Machine Learning techniques.

Three different classification approaches were evaluated: linguistic features only, stylistic features only, and combined linguistic-stylistic features. TF-IDF vectorization was used for linguistic representation, while statistical writing characteristics were extracted as stylistic features. Logistic Regression models were trained and evaluated using Accuracy, Precision, Recall, and F1-Score metrics.

Experimental results demonstrated that linguistic features achieved the highest classification performance, while stylistic features provided additional complementary information. The study shows that textual content plays a major role in fake news detection tasks.

---

# 1. Introduction

With the rapid development of online news platforms and social media, fake news has become a major problem affecting society, politics, and public opinion. Since fake news spreads quickly on digital platforms, automatic detection systems have become increasingly important.

Natural Language Processing techniques are commonly used for fake news classification because fake and real news articles usually contain different textual patterns. In addition to textual content, writing style may also provide useful information for distinguishing fake news from reliable journalism.

This project focuses on comparing:

* linguistic features
* stylistic features
* combined feature representations

for fake news detection using Machine Learning methods.

---

# 2. Dataset

The dataset used in this project was obtained from Kaggle:

[Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?utm_source=chatgpt.com)

The dataset consists of two CSV files:

* `Fake.csv`
* `True.csv`

The dataset contains:

* news article titles
* article text content
* labels representing fake or real news

Labels used in the project:

* `1` → Fake News
* `0` → Real News

The fake and real news datasets were merged into a single dataframe before preprocessing and feature extraction.

---

# 3. Data Preprocessing

Before feature extraction, the text data was cleaned and normalized to reduce noise and improve model performance.

The preprocessing pipeline included:

* converting text to lowercase
* removing punctuation
* removing special characters
* removing extra whitespace characters

Example preprocessing operation:

```python
text = text.lower()
```

Regular expressions were also used to preserve only alphabetic characters and spaces. These preprocessing operations helped standardize textual data before feature extraction.

---

# 4. Feature Extraction

The project investigates two different feature categories:

1. Linguistic Features
2. Stylistic Features

---

# 4.1 Linguistic Features

Linguistic features were extracted using TF-IDF (Term Frequency – Inverse Document Frequency) vectorization.

TF-IDF measures the importance of words within documents by considering both word frequency and document frequency.

The following TF-IDF configuration was used:

```python
TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,2)
)
```

The linguistic representation allowed the model to learn:

* important keywords
* phrase frequencies
* textual patterns associated with fake and real news

The final TF-IDF representation contained 5000 dimensions.

---

# 4.2 Stylistic Features

In addition to linguistic content, stylistic writing characteristics were extracted from each news article.

The extracted stylistic features included:

* Word Count
* Character Count
* Sentence Count
* Average Word Length
* Average Sentence Length
* Exclamation Count
* Question Mark Count
* Uppercase Ratio
* Punctuation Ratio

Unlike TF-IDF vectors, stylistic features focus on how the article is written rather than what the article contains.

These features were extracted to investigate whether fake news articles contain distinctive writing behaviors compared to real news articles.

---

# 5. Machine Learning Models

Three different Logistic Regression models were trained and evaluated.

The Logistic Regression configuration used in the experiments was:

```python
LogisticRegression(max_iter=1000)
```

---

# 5.1 Linguistic Features Model

The first model was trained using only TF-IDF linguistic features.

This model learned textual patterns and word distributions associated with fake and real news articles.

---

# 5.2 Stylistic Features Model

The second model used only stylistic writing features.

Before training, stylistic features were standardized to improve training stability and feature consistency.

The purpose of this experiment was to analyze whether writing style alone could distinguish fake news from real news.

---

# 5.3 Combined Features Model

The final model combined:

* TF-IDF linguistic features
* stylistic statistical features

Feature matrices were merged using:

```python
hstack([X_train_linguistic, X_train_stylistic_scaled])
```

The objective of this experiment was to evaluate whether combining both feature categories improved classification performance.

---

# 6. Evaluation Metrics

The models were evaluated using standard classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score

Confusion matrix analysis was also performed to analyze prediction behavior and classification reliability.

These metrics provide insight into:

* classification accuracy
* false positive rates
* false negative rates
* overall model robustness

---

# 7. Experimental Results

## 7.1 Linguistic Features Results

| Metric    | Result |
| --------- | ------ |
| Accuracy  | 98.75% |
| Precision | 99.11% |
| Recall    | 98.75% |
| F1-Score  | 98.93% |

The linguistic features model achieved the highest classification performance because textual content strongly reflects fake news patterns.

---

## 7.2 Stylistic Features Results

| Metric    | Result |
| --------- | ------ |
| Accuracy  | 81.70% |
| Precision | 87.83% |
| Recall    | 75.47% |
| F1-Score  | 81.18% |

The stylistic features model achieved lower performance because writing style alone does not always provide sufficient information for classification.

---

## 7.3 Combined Features Results

| Metric    | Result |
| --------- | ------ |
| Accuracy  | 97.78% |
| Precision | 98.41% |
| Recall    | 97.33% |
| F1-Score  | 97.86% |

The combined model produced strong classification performance, although linguistic features remained the dominant factor.

---

# 8. Discussion

The experimental results showed that linguistic features contributed significantly more to fake news detection than stylistic features.

One possible explanation is the dimensional difference between the feature representations:

* TF-IDF vectors contain thousands of textual dimensions
* stylistic representations contain only a limited number of statistical features

Therefore, the model learns richer semantic information from textual content.

However, stylistic features still provide useful complementary information and may improve robustness in certain classification scenarios.

---

# 9. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SciPy

---

# 10. Project Structure

```text
.
├── main.py
├── data_loader.py
├── feature_extraction.py
├── data/
│   ├── Fake.csv
│   └── True.csv
└── README.md
```

---

# 11. Installation and Usage

## Install dependencies

```bash
pip install pandas numpy scipy scikit-learn
```

## Add dataset files

Place the following dataset files inside the `data/` directory:

```text
Fake.csv
True.csv
```

## Run the project

```bash
python main.py
```

---

# 12. Future Work

Possible future improvements include:

* Deep Learning based classification methods
* Transformer and BERT architectures
* Larger and more diverse datasets
* Explainable AI techniques
* Real-time fake news detection systems
* Advanced stylistic feature engineering

---

# 13. Conclusion

This project investigated fake news detection using linguistic and stylistic features with Machine Learning techniques.

Experimental results demonstrated that:

* linguistic features achieved the highest classification performance
* stylistic features alone were less effective
* combining both feature categories still produced strong results

Overall, the study shows that textual content plays a major role in fake news detection and that NLP techniques can successfully classify fake and real news articles with high accuracy.

---
## Report

Detailed technical report can be found in the repo:


---

# Author

Emir Gençler
