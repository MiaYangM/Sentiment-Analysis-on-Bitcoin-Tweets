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

1. **Clone this repo and unzip the archive (if necessary).**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
4. **Open and run `Sentiment_Analysis_Bitcoin_tweets_te.ipynb`.**
   - Make sure you have the required data files in the same directory, if not included.

---

## **Python Version**

**This project was developed and tested with Python 3.11.**  
Check the top of the notebook for the version used.

---

## **Notes**

- For reproducibility, random seeds are set where possible.
- Some models may require a GPU (recommended for LSTM and transformers).
- Data files (tweets) are not included due to privacy/compliance; please add them as required.

---

## **Contact**

For questions or issues, please open an issue on this repository.
