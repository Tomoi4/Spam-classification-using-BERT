# Spam-classification-using-BERT
###
🚀 Overview
This project leverages BERT (Bidirectional Encoder Representations from Transformers) to classify emails or messages as spam or ham (non-spam). By fine-tuning a pre-trained BERT model, we achieve high accuracy in detecting spam based on contextual understanding of text.

🧠 Why BERT?
- BERT processes text bidirectionally, capturing context from both left and right.
- It’s pre-trained on massive corpora, making it ideal for nuanced NLP tasks like spam detection.
- Unlike traditional models, BERT understands semantic meaning, not just keyword presence.

📂 Dataset
- Source: Enron-Spam Dataset / SMS Spam Collection (UCI)
- Classes: spam, ham
- Preprocessing:
- Removed URLs, HTML tags, and special characters
- Balanced dataset by downsampling spam to match ham count
- No lemmatization or stopword removal (BERT handles context)
