import pandas as pd
import torch
from transformers import BertTokenizer
from transformers import AutoModelForSequenceClassification


# Load the model
model_checkpoint = "D:\PyCharm 2023.1\\new_reddit\\distilbert-base-uncased-finetuned-emotion\\checkpoint-500"  # Replace with the path to your checkpoint file
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)


# Load the CSV data into a DataFrame
df = pd.read_csv('D:\PyCharm 2023.1\\new_reddit\datasets\\reddit_posts_processed.csv')

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')  # Use the same tokenizer you used for training.

# Tokenize and encode the 'body' column
encoded_data = [tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt') for text in df['body']]

# Extract input_ids from BatchEncoding objects and create a tensor
input_ids = torch.cat([sample['input_ids'] for sample in encoded_data], dim=0)

with torch.no_grad():
    outputs = model(input_ids.to("cuda" if torch.cuda.is_available() else "cpu"))  # Ensure that your model is on the same device (CPU or GPU) as during training.

# Assuming your model predicts emotion labels (class probabilities)
predicted_emotion_probs = torch.softmax(outputs.logits, dim=1)

# Get the predicted class labels (index of the maximum probability)
predicted_emotion_labels = torch.argmax(predicted_emotion_probs, dim=1)

df['predicted_emotion_label'] = predicted_emotion_labels.tolist()
df['predicted_emotion_probs'] = predicted_emotion_probs.tolist()

df.to_csv('Reddit_Posts_with_Emotion_Classification.csv', index=False)
