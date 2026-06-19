import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

labels = ["Negative", "Positive"]

# ----------------------------
# UI Layout
# ----------------------------
st.title("🧠 Sentiment Analysis using BERT")
st.write("Analyze text sentiment using a pre-trained BERT model.")

# ----------------------------
# Text Input Section
# ----------------------------
user_input = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        st.subheader("Result:")
        st.write(f"**Sentiment:** {labels[pred]}")
        st.write(f"Confidence: {probs[0][pred].item():.2f}")
    else:
        st.warning("Please enter some text.")

# ----------------------------
# Dataset Upload Section
# ----------------------------
st.header("📊 Upload Dataset for Visualization")

uploaded_file = st.file_uploader("Upload CSV file with 'text' and 'sentiment' columns", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Sentiment Distribution")

    fig, ax = plt.subplots()
    df['sentiment'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Sentiment Distribution")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_ylabel("")
    st.pyplot(fig2)