import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load Pretrained Model
MODEL_PATH = "distilbert-base-uncased"  # Use a public model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)

# Load dataset
DATASET_PATH = "dataset/brain_rot_data.csv"
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()  # Remove extra spaces


# Define Severity Labels
SEVERITY_LABELS = ["Mild", "Intense", "Severe"]

# Function to Predict Severity Using BERT
def predict_severity(text):
    """Uses BERT to predict brain rot severity based on input text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return SEVERITY_LABELS[predicted_class]

# Function to Compare with Dataset
def compare_with_dataset(response, age_group):
    """Checks if a user's response matches dataset entries for severity classification."""
    for _, row in df.iterrows():
        if row["Age_Group"] == age_group and row["Response"].lower() in response.lower():
            return row["Severity"]
    return None  # If no match, use BERT prediction
