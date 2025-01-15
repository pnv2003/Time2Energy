from matplotlib import pyplot as plt
import torch
from dl.regularize import EarlyStopping

def train_model(
        loaders, model, criterion, optimizer, device, 
        num_epochs=20, early_stop_patience=5, early_stop_delta=0.00001):
    
    train_losses, valid_losses = [], []
    early_stopping = EarlyStopping(patience=early_stop_patience, delta=early_stop_delta)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in loaders['train']:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(loaders['train']))

        model.eval()
        with torch.no_grad():
            valid_loss = sum(
                criterion(
                    model(X_batch.to(device)), 
                    y_batch.to(device).unsqueeze(1)
                ).item() 
                for X_batch, y_batch in loaders['valid']
            )
        valid_losses.append(valid_loss / len(loaders['valid']))

        print(f'Epoch {epoch+1}/{num_epochs} | ' 
              f'Train Loss: {train_losses[-1]:.4f} | '
              f'Valid Loss: {valid_losses[-1]:.4f}')
        
        if early_stopping(valid_losses[-1]):
            print('Early stopping triggered')
            break

    # plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.legend()
    plt.show()

    return model
        
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved at {path}')

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f'Model loaded from {path}')
    return model
