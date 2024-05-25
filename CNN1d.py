import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CNN1D(nn.Module):
    def __init__(self, input_size, num_units1, num_units2, num_units3, num_units4, num_units5,num_units6, epochs, batchsize,window_length,maxpooling,dropout):
        super(CNN1D, self).__init__()
        self.cnn1d = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=num_units1, kernel_size=13),
            nn.BatchNorm1d(num_units1),
            nn.RReLU(),
            nn.MaxPool1d(maxpooling),
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=num_units1, out_channels=num_units2, kernel_size=9),
            nn.BatchNorm1d(num_units2),
            nn.RReLU(),
            nn.MaxPool1d(maxpooling),
            nn.Dropout(p=dropout),
            nn.Conv1d(in_channels=num_units2, out_channels=num_units3, kernel_size=5),
            nn.BatchNorm1d(num_units3),
            nn.RReLU(),
            nn.MaxPool1d(maxpooling),
            nn.Dropout(p=dropout),
        )
        test_tensor = self.cnn1d(torch.zeros(batchsize, input_size, window_length))
        test_tensor = test_tensor.view(test_tensor.size()[0], -1)
        self.fullyconnected1 = nn.Sequential(
            nn.Linear(test_tensor.shape[1], num_units4),
            nn.BatchNorm1d(num_units4),
        )
        self.fullyconnected2 = nn.Sequential(
            nn.RReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(num_units4, num_units5),
            nn.BatchNorm1d(num_units5),
            nn.RReLU(),
            nn.Linear(num_units5, num_units6),
            nn.Softmax(dim=1)
        )
        self.epochs = epochs
        self.batchsize = batchsize

    def forward(self, x):
        output = self.cnn1d(x)
        output = output.view(output.size()[0], -1)
        output = self.fullyconnected1(output)
        final_output = self.fullyconnected2(output)
        return final_output

    def fit(self, train_loader, valid_loader, criterion, optimizer):
        train_losses = []
        valid_losses = []
        best_valid_loss = float('inf')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        for epoch in range(self.epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)

            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = self.forward(inputs)
                    loss = criterion(outputs, labels.long())
                    valid_loss += loss.item() * inputs.size(0)
            valid_loss /= len(valid_loader.dataset)
            valid_losses.append(valid_loss)

            print(
                f'Epoch {epoch + 1}/{self.epochs}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {valid_losses[-1]:.4f}')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.state_dict(), 'best_model.pth')