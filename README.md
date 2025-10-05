# Sentiment Analysis on Tweets: Classical Machine Learning vs Transformer (RoBERTa)
This project implements a comprehensive Sentiment Analysis system that compares the performance and accuracy of classical machine learning models and a transformer-based model (RoBERTa) using Twitter text data.
It is developed as an interactive Streamlit web application, allowing users to visualize the preprocessing steps, model evaluations and review deep-learning explainability through gradients, attention heads and hidden state visualizations.

## Abstract
The objective of this project is so design, train and evaluate various sentiment classification models capable of determining whether a tweet is positive, neutral or negative.
The project integrates both traditional ML pieplines (TF-IDF + Scikit-Learn models) and state-of-the-art transformer architecture for comparisons.

## Features
- Data Preprocessing
  - Removing special characters, URLs, hashtags and mentions.
  - Lemmatization and stop word removal using NLTK.
  - TF-IDF feature extraction
- Classical Models
  - Logistic Regression
  - Random Forest
  - Linear SVC
  - K-Nearest Neighbors
  - Multinomial Naive Bayes
  - Complement Naive Bayes
  - Passive Aggressive Classifier
- Transformer Model
  - RoBERTa
- Model Evaluation
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- Explainability & Visualization
  - Gradient Sensitivity Graphs: demonstrates token influence on sentiment prediction.
  - Attention Heatmaps: visualize token attention across heads.
  - Hidden State Activations: analyze token representations strength across layers.
- Interactive Dashboard
  - Implemented on Streamlit with tabbed navigation for model comparison
  - Clear side-by-side visualizations of traditional vs transformer models.

 ## Implementation Details
 - Language: Python
 - Framework: Streamlit
 - Libraries:
   - transformers
   - torch
   - pandas
   - numpy
   - scikit-learn
   - seaborn
   - matplotlib
- Pretrained Models:
  - `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - Tokenizer: `roberta-case`
- Classical ML Models:
  - Trained using TfidfVectorizer + Scikit-Learn pipelines.

## Usage
- Launch the app:
  `streamlit run sentimentAnalysis.py`

## Results
- Traditonal ML Models: Achieved accuracies 40%-65% depending on the algorithm.
- Transformer (RoBERTa): Outperformed all traditional models with a higher overall accuracy of 69% due to contextual understanding of language.

