# app.py (Modified to remove confidence score display)

import streamlit as st
from transformers import pipeline
import torch

# --- Configuration ---
SAVED_MODEL_PATH = "./spam_classifier_bert_model"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="BERT Spam/Ham Classifier",
    page_icon="📧",
    layout="centered"
)

# --- Function to Load the Model (with Caching) ---
@st.cache_resource
def load_classifier_model():
    """
    Loads the pre-trained BERT spam classifier model and its tokenizer.
    This function is cached by Streamlit for performance.
    """
    try:
        classifier = pipeline(
            "text-classification",
            model=SAVED_MODEL_PATH,
            tokenizer=SAVED_MODEL_PATH,
            device=0 if torch.cuda.is_available() else -1
        )
        st.success("Model loaded successfully! Ready for classification.")
        return classifier
    except Exception as e:
        st.error(f"ERROR: Could not load the model from '{SAVED_MODEL_PATH}'.")
        st.error(f"Please ensure the folder exists and contains the model files (e.g., pytorch_model.bin, config.json).")
        st.error(f"Details: {e}")
        return None

# Load the model globally when the app starts.
classifier_model = load_classifier_model()

# --- Streamlit App UI Layout ---

st.title("📧 BERT Spam/Ham Classifier")
st.markdown("Type an email message or text below, and BERT will predict if it's **SPAM** or **HAM**.")

user_input = st.text_area(
    "Enter your text here:",
    "",
    height=150,
    placeholder="e.g., 'Congratulations! You've won a free prize. Click this link now to claim!'"
)

if st.button("Analyze Message", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    elif classifier_model is None:
        st.error("The classification model is not loaded. Please check for errors above.")
    else:
        with st.spinner("Analyzing message..."):
            prediction_result = classifier_model(user_input)
            label = prediction_result[0]['label']
            score = prediction_result[0]['score'] # We still extract 'score', but won't display it directly

            predicted_class = ""
            if label == 'LABEL_0':
                predicted_class = 'HAM'
            elif label == 'LABEL_1':
                predicted_class = 'SPAM'
            else:
                predicted_class = label

            st.subheader("Classification Result:")

            if predicted_class == 'SPAM':
                st.markdown(f"**Sentiment:** <span style='color:red; font-size: 24px;'>SPAM 🚨</span>", unsafe_allow_html=True)
            elif predicted_class == 'HAM':
                st.markdown(f"**Sentiment:** <span style='color:green; font-size: 24px;'>HAM ✅</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**Sentiment:** {predicted_class}", unsafe_allow_html=True)

            # --- Lines related to confidence score display and messages ---
            # You can uncomment these if you change your mind later
            # st.markdown(f"**Confidence Score:** `{score:.4f}`") # This line is removed/commented out

            # These messages still depend on the 'score' variable, so if you remove them
            # you also remove any explicit reference to confidence.
            
            # --- End of lines related to confidence score ---