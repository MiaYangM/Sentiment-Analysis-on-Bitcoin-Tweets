# Sentiment-Analysis-on-Bitcoin-Tweets

This project analyzes the sentiment of 2,000 tweets about Bitcoin using a variety of NLP and machine learning approaches. The pipeline covers everything from data cleaning to advanced deep learning and transformer-based models. All models are evaluated on a held-out test set and summarized for easy comparison.

---

## **Project Structure**

- **Sentiment_Analysis_Bitcoin_tweets_te.ipynb:**  
  Main notebook containing the full workflow and results.

- **requirements.txt:**  
  List of dependencies required to run the notebook.

---

## **Approaches Implemented**

1. **Text Preprocessing:**  
   - Cleans tweets (removes HTML, URLs, unicode, emojis, mentions, etc.)
   - Handles hashtags and combines them with tweet content.

2. **Sentiment Dictionary (VADER):**  
   - Uses NLTK’s VADER for lexicon-based sentiment prediction.

3. **TF-IDF + Logistic Regression:**  
   - Vectorizes tweets using TF-IDF.
   - Trains a Logistic Regression classifier.

4. **LSTM (own embeddings):**  
   - Recurrent Neural Network trained from scratch with custom embeddings.
   - Includes validation split and early stopping.

5. **LSTM (pre-trained GloVe embeddings):**  
   - RNN classifier with pre-trained GloVe word vectors.
   - Includes validation split and early stopping.
   - Pre-trained Word Vectors uses [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/).
     You can download the pre-trained vectors glove.2024.wikigiga.100d.zip  [here](https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.100d.zip)

6. **Pre-trained Transformer (Zero-shot):**  
   - Applies a Hugging Face transformer pipeline (DistilBERT) to the data.

7. **Fine-tuned Transformer:**  
   - Fine-tunes DistilBERT on the training set (1 epoch for computational efficiency).

---

## **Evaluation**

Each model is evaluated on the same labeled test set using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

A summary table at the end of the notebook compares all approaches.

---

## **Usage**

1. **Clone this repository and unzip the archive if needed.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run on Google Colab:**

   This project is intended to be executed in [Google Colab](https://colab.research.google.com/).  
   To get started, simply open the notebook in Colab and run each cell in sequence.

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_NOTEBOOK_LINK_HERE)

4. **Open and execute `Sentiment_Analysis_Bitcoin_tweets_te.ipynb`.**
   - Ensure that all required data files are present in the working directory. If the data files are not included in this repository, please add them manually.

---

## **Python Version**

**Developed and tested with Python 3.11.**  
See the notebook header for the exact version used in this project.

---

## **Notes**

- Random seeds are set wherever possible to improve reproducibility.
- Some models (e.g., LSTM, transformers) benefit significantly from GPU acceleration.
- Data files containing tweets are **not included** in this repository due to privacy and compliance considerations; please add them as needed.

---
For questions or issues, please open an issue on this repository.
