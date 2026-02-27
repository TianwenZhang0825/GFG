import torch
from models.dpst import DPST
from data.datasets import SSDDataset, HRSIDDataset
from trainer import Trainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    ssd_data = SSDDataset(root="path/to/ssdd")
    hrsid_data = HRSIDDataset(root="path/to/hrsid")
    
    # Initialize model
    model = DPST().to(device)
    
    # Initialize trainer
    trainer = Trainer(model, device)
    
    # Train on SSDD
    trainer.train(ssd_data, epochs=240, batch_size=4)
    
    # Evaluate on HRSID
    trainer.evaluate(hrsid_data)

if __name__ == "__main__":
    main()