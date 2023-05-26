import numpy as np 
import torch
import torch.nn as nn
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import random

from .config import LANGUAGES
from .data_extractor import DataExtractor

# Define the sequential model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(Model, self).__init__()
        # self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
    

train = False
if train == True:
    dataextractor = DataExtractor(LANGUAGES)
    X, y = dataextractor.process_data()

    # randomizing
    combined_data = list(zip(X, y))
    random.shuffle(combined_data)
    X, y = zip(*combined_data)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41)

    # Convert X and y to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train_indices = torch.tensor([dataextractor.convert_ohe_to_index(lang) for lang in y_train], dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val_indices = torch.tensor([dataextractor.convert_ohe_to_index(lang) for lang in y_val], dtype=torch.long)

    # Set hyperparameters
    input_size = len(X_train[0])
    hidden_size = 128
    output_size = len(y_train[0])
    learning_rate = 0.001
    dropout_rate = 0.0
    num_epochs = 350

    # Create an instance of the sequential model
    model = Model(input_size, hidden_size, output_size, dropout_rate)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')
    best_epoch = 0

    # training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train_indices)
        loss.backward()
        optimizer.step()
        
        # validation
        model.eval()
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val_indices)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_weights = model.state_dict()
        
        print(f"Epoch: {epoch+1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")
    print("Best epoch: ", best_epoch)
    # load the best weights

    model.load_state_dict(best_weights)

    # calculate F1 score using the best weights
    model.eval()
    val_outputs = model(X_val)
    _, val_predictions = torch.max(val_outputs, 1)
    f1_scores = f1_score(y_val_indices.detach().numpy(), val_predictions.detach().numpy(), average=None)

    # print F1 score for each label
    for label, f1 in enumerate(f1_scores):
        print(f"Label {label}: F1 score = {f1}")

    torch.save(best_weights, "best_model_weights.pt")
    print("Best model weights saved.")

######################################################

def prediction(text: str) -> str: 
    model = Model(125, 256, 28, 0)
    # Load the saved weights
    model.load_state_dict(torch.load('best_model_weights.pt'))
    model.eval()
    preprocessed_text = dataextractor.process_text(text)
    preprocessed_text_tensor = torch.tensor(preprocessed_text, dtype=torch.float32)
    outputs = model(preprocessed_text_tensor.unsqueeze(0))
    _, predictions = torch.max(outputs, 1)
    predicted_label = predictions.item()
    predicted_lang = dataextractor.convert_prediction_to_langname(predicted_label)
    return predicted_lang