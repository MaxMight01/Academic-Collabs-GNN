import torch
import json
from src.model.train import train

if __name__ == "__main__":
    with open('src/config.json', 'r') as file:
        config = json.load(file)

    lr = config['learning_rate']
    epochs = config['num_epochs']
    hidden_dim = config['hidden_dim']
    layers = config['num_layers']
    dropout = config['dropout']

    data = torch.load("data/processed/train_data.pt", weights_only=False)
    train(data, epochs=epochs, hidden_channels=hidden_dim, lr=lr, layers=layers, dropout=dropout)