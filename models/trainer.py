import torch
import torch.optim as optim
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, device, lr=0.0004):
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        
    def train(self, dataset, epochs=240, batch_size=4):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output, batch)  # dummy loss for illustration
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
                
    def evaluate(self, dataset):
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                output = self.model(batch)
                # Evaluation metrics can be computed here