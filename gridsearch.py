import itertools
import torch
import torch.nn as nn
from train import train_model

def evaluate_model(model, test_loader):
    model.eval()
    test_correct = 0
    test_total = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in test_loader:
            snapshots = batch.to_data_list()
            outputs = model(snapshots).unsqueeze(0)
            labels = batch[0].y
            pred = outputs.argmax(dim=1)
            test_correct += (pred == labels).sum().item()
            test_total += labels.size(0)
    accuracy = test_correct / test_total if test_total > 0 else 0.0
    return accuracy

def grid_search(model_class, param_grid, train_loader, test_loader, train_func=train_model, **train_kwargs):
    best_metric = -float('inf')
    best_params = None
    best_model = None
    keys = list(param_grid.keys())
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        print("Testing parameters:", params)
        model = model_class(**params)
        trained_model = train_func(model, train_loader, test_loader, **train_kwargs)
        metric = evaluate_model(trained_model, test_loader)
        print(f"Test accuracy for parameters {params}: {metric:.4f}")
        if metric > best_metric:
            best_metric = metric
            best_params = params
            best_model = trained_model
    return best_params, best_model, best_metric
