import torch
import json
from src.model.train import train
from src.model.evaluate import evaluate

if __name__ == "__main__":
    with open('src/config.json', 'r') as file:
        config = json.load(file)

    lr = config['learning_rate']
    epochs = config['num_epochs']
    hidden_dim = config['hidden_dim']
    layers = config['num_layers']
    dropout = config['dropout']

    train_data = torch.load("data/processed/train_data.pt", weights_only=False)
    val_data = torch.load("data/processed/val_data.pt", weights_only=False)
    model, predictor = train(data=train_data, val_data=val_data, epochs=epochs, hidden_channels=hidden_dim, lr=lr, layers=layers, dropout=dropout)

    val_data = torch.load("data/processed/val_data.pt", weights_only=False)
    evaluate(model, predictor, val_data)