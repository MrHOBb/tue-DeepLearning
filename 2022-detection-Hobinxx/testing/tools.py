import logging as logger
import torch
from collections import OrderedDict

def compare_models(model1, model2, weights=False):
    
    if type(model1) is OrderedDict:
        dict1 = model1
    else:
        dict1 = model1.state_dict()
        
    if type(model2) is OrderedDict:
        dict2 = model2
    else:
        dict2 = model2.state_dict()
    
    if len(dict1) != len(dict2):
        print(f"Length mismatch: {len(dict1)}, {len(dict2)}"
        )
        return False
    
    for ((k_1, v_1), (k_2, v_2)) in zip(
        dict1.items(),
        dict2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        if not v_1.shape == v_2.shape:
            print(f"Shape mismatch: {k_1} vs {k_2}")
            return False
        
        if weights:
            return torch.allclose(v_1, v_2)
    
    return True


def compare_models_torch_dictionary(model1_torch, model2_dict):
    """ Compare layers and shape of two models

    Args:
        model_torch (nn.Sequential): torch model
        model_dict (dict): layer names as keys and layer shapes as values

    Returns:
        bool: wether the shapes match
    """
    
    model1_dict = model1_torch.state_dict()
    
    if len(model1_dict) != len(model2_dict):
        print(
            f"Length mismatch: {len(model1_dict)}, {len(model2_dict)}"
        )
        return False
    
    for ((k_1, v_1), (k_2, v_2)) in zip(
        model1_dict.items(),
        model2_dict.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        if not v_1.shape == v_2:
            print(f"Shape mismatch: {k_1} vs {k_2}")
            return False
    
    return True
