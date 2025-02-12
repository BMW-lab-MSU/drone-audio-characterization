import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class AudioCNN(nn.Module):
    # I think there are only 2 classes, but I can't remember
    # These initialization steps need to be verified
    def __init__(self, num_classes=2, in_channels=1):
        super(AudioCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        
        # LazyLinear will infer the input features
        self.fc1 = nn.LazyLinear(64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.LazyLinear(num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        x = torch.flatten(x, 1)  # Flatten to [batch, features]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # raw logits
        return x



def main():
    # Hyperparameters and setup, consider adding more
    num_epochs = 5
    batch_size = 32
    learning_rate = 1e-3
    
    # Load data (code should be in CNN_audio_feature_extraction.py)
    dataset = XXXXXXXXX(num_samples=1000, height=64, width=64, num_classes=10) # replace XXXXXXX with properly formatted dataset
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Instantiate model, loss, optimizer
    model = AudioCNN(num_classes=10, in_channels=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # move model to gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Model Training (Look into saving the model)
    for epoch in tqdm(num_epochs, desc="Training CNN...", leave=False):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Move data to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print average loss per epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    main()
