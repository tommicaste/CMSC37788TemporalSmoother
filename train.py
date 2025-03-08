import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, test_loader, lr=0.001, tol=1e-4, num_epochs=100, checkpoint_path=None):
    """
    Trains the given model using the provided data loaders.
    
    Args:
        model  The model to train.
        train_loader : list of training batches.
        test_loader : list of testing batches.
        lr : Learning rate.
        tol : Tolerance for stopping criterion.
        num_epochs : Maximum number of epochs.
        checkpoint_path : Path to save/load model checkpoints.
        
    Returns:
        The trained model.
    """
    
    if checkpoint_path is None:
        checkpoint_path = f"{model.__class__.__name__}_checkpoint.pth"
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # Resume training if checkpoint exists.
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        prev_loss = checkpoint["prev_loss"]
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        start_epoch = 0
        prev_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            snapshots = batch.to_data_list()
            out = model(snapshots).unsqueeze(0)  
            label = batch[0].y  
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
        
        avg_loss = epoch_loss / len(train_loader)
        train_accuracy = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}, Train Accuracy = {train_accuracy:.4f}")
        
        # Evaluate on test set.
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch in test_loader:
                snapshots = batch.to_data_list()
                out = model(snapshots).unsqueeze(0)
                label = batch[0].y
                loss = loss_fn(out, label)
                test_loss += loss.item()
                pred = out.argmax(dim=1)
                test_correct += (pred == label).sum().item()
                test_total += label.size(0)
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total if test_total > 0 else 0.0
        print(f"Epoch {epoch+1}: Test Loss = {avg_test_loss:.6f}, Test Accuracy = {test_accuracy:.4f}")
        
        # Save checkpoint.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "prev_loss": avg_loss
        }, checkpoint_path)
        
        if abs(prev_loss - avg_loss) < tol:
            print("Tolerance reached. Stopping training.")
            break
        prev_loss = avg_loss
    
    return model
