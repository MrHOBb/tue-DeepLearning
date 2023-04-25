import os
import random
import pickle
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F

def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def check_accuracy(model, loader, device, dtype=torch.float32):
    """A helper function used to calculate loss and accuracy on an entire dataset.
        DO NOT MODIFY!

        Args:
            model: torch model.
            loader: torch dataloader.

        Returns:
            accuracy, loss (List[]): accuracy and loss for the whole dataset.
    """
    total_samples = 0
    loss = 0
    correct_samples = 0
    samples = len(loader.dataset)
    model.eval()
    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(device=device, dtype=dtype) 
            y = y.to(device=device, dtype=dtype)
            scores = torch.argmax(model(x), dim=1).to(dtype=dtype)
            loss += F.cross_entropy(scores, y)
            correct_samples += (scores == y).sum().item()
            total_samples += y.shape[0]
            
    return np.array([float(correct_samples) / total_samples, loss / samples])
    
  
def convert_model_to_dict(model, name='model'):
    """Convert either a model or model_state to a python dictionary to save space.

    Args:
        model (model or model_state): model to be converted
        name (str, optional): Name of the output file (without extension). Defaults to 'model'.
    """
    if type(model) is OrderedDict:
        model_dict = model
    else:
        model_dict = model.state_dict()
        
    dictionary = {}
    
    for key, value in model_dict.items():
        dictionary.update({key: value.shape})
    
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
        
    # with open('saved_dictionary.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)
    
    