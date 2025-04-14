import torch
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
from transformers import Trainer, TrainingArguments
from torch.nn.functional import cross_entropy
from sklearn.metrics import accuracy_score, f1_score

emotions = load_dataset("emotion")
print('----------------emotion dataset----------------')
print(emotions)   # check the emotion datasets

train_ds = emotions["train"]
emotions.set_format(type="pandas")
df = emotions["train"][:]

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

# add a column called the label name
df["label_name"] = df["label"].apply(label_int2str)
print('----------------trainset----------------')
print(df.head())

def tokenize(batch):
    batch["label"] = batch["label"].tolist()
    tokenized_texts = tokenizer(batch["text"].tolist(), padding=True, truncation=True)
    return {"input_ids": tokenized_texts["input_ids"],
            "attention_mask": tokenized_texts["attention_mask"],
            "label": batch["label"]}

def extract_hidden_states(batch):
    # Place model inputs on the GPU
     inputs = {k: v.to(device) for k,v in batch.items()
        if k in tokenizer.model_input_names}
    # Extract last hidden states
     with torch.no_grad():
         last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
     return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}
def forward_pass_with_label(batch):
    # function that returns the loss along with the predicted label #
    # Place all input tensors on the same device as the model
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device),
                             reduction="none")
    # Place outputs on CPU for compatibility with other dataset columns
    return {"loss": loss.cpu().numpy(),
             "predicted_label": pred_label.cpu().numpy()}


# Define the loss function
loss_fn = cross_entropy

# import the model
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

# convert the input_ids and attention_mask columns to the "torch" format
emotions_encoded.set_format("torch",
                             columns=["input_ids", "attention_mask", "label"])

# extract the hidden states across all splits in one go
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

# check the column names
print("----------------column names----------------")
print(emotions_hidden["train"].column_names)

# Creating a feature matrix
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
labels = emotions["train"].features["label"].names
num_labels = 6

# define the classification model
classification_model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))

# define the training parameters
batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=20,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="steps",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  #log_level="info",
                                  #logging_dir="D:\PyCharm 2023.1\\new_reddit\output\logs"  # set the directory to save logs
                                  )

# training the model
trainer = Trainer(model=classification_model,
                  args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train()

# predictions on the validation set
preds_output = trainer.predict(emotions_encoded["validation"])
y_preds = np.argmax(preds_output.predictions, axis=1)
# visualize the prediction matrix
plot_confusion_matrix(y_preds, y_valid, labels)

# Convert our dataset back to PyTorch tensors
emotions_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])
# Compute loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map(
    forward_pass_with_label, batched=True, batch_size=16)

# create a DataFrame with the texts, losses, and predicted/true labels.
emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (df_test["predicted_label"]
                              .apply(label_int2str))

# have a look at the data samples with the highest losses
print("----------------data samples with the highest losses----------------")
print(df_test.sort_values("loss", ascending=False).head(5))
