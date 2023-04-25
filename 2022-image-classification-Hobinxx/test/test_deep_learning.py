import sys
import pickle
from importing import NotebookFinder
sys.meta_path.append(NotebookFinder())
import assignment_deep_learning_1 as student

import pytest
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, sampler
import torchvision.datasets as datasets
import torchvision.transforms as T

from helper import set_all_seeds, set_deterministic
from tools import compare_models, compare_models_torch_dictionary


def test_train_model():
    RANDOM_SEED = 420

    set_all_seeds(RANDOM_SEED)
    set_deterministic()
    
    transforms = T.Compose([T.Resize((70, 70)),
                                T.RandomCrop((64, 64)),
                                T.ToTensor(),
                                T.Normalize((0.49139968, 0.48215827, 0.44653124), 
                                            (0.24703233, 0.24348505, 0.26158768))])
    dataset = datasets.CIFAR10(root='./CIFAR10',
                                        train=True,
                                        transform=transforms,
                                        download=True)
    dataset = Subset(dataset, list(range(10)))
    loader = DataLoader(dataset, 
                              batch_sampler=sampler.BatchSampler(sampler.SequentialSampler(dataset), 
                                                                 batch_size=3, drop_last=False))
    
    device = torch.device('cpu')
    dtype = torch.float32
    student_model = nn.Sequential(
    # Flatten the 2D image to 1D
        nn.Flatten(),
        nn.Linear(64*64*3, 512, device=device, dtype=dtype),
        nn.ReLU(),
        nn.Linear(512, 512, device=device, dtype=dtype),
        nn.ReLU(),
        nn.Linear(512, 512, device=device, dtype=dtype),
        nn.ReLU(),
        nn.Linear(in_features=512 , out_features=10, device=device, dtype=dtype)
    )
    optimizer = optim.SGD(student_model.parameters(), lr=0.0001)
    loss = nn.CrossEntropyLoss()
    epochs = 10
    _ = student.train_model(student_model, optimizer, loss, epochs, device=device,
                                   loader_train=loader, loader_validation=None)
    
    # Load correct model
    model_correct = torch.load('test/models/model_train.pth')
    assert compare_models(model_correct, student_model, True)
    

# def test_train_model_alt():
#     """ Train whether the student model is different than random init
#     """
#     student_model = torch.load('model_student.pth')
#     model_correct = torch.load('test/models/model_train_initial.pth')
#     assert not compare_models(model_correct, student_model)


def test_model_fcnn():
    with open('test/models/model_fcnn_correct.pkl', 'rb') as f:
        fcnn_correct_dict = pickle.load(f)
    student_model = student.model_fcnn()
    assert compare_models_torch_dictionary(student_model, fcnn_correct_dict), "Model not correct"
    assert type(student.loss_fcnn) is nn.CrossEntropyLoss, "Incorrect loss"


def test_model_alexnet_base():
    with open('test/models/model_alexnet_correct.pkl', 'rb') as f:
        alexnet_correct_dict = pickle.load(f)
    student_model = student.model_alexnet()
    assert compare_models_torch_dictionary(student_model, alexnet_correct_dict), "Model not correct"
    assert type(student.loss_alexnet) is nn.CrossEntropyLoss, "Incorrect loss"


def test_model_alexnet_regularized():
    assert type(student.loss_alexnet_regularized) is nn.CrossEntropyLoss, "Incorrect loss"
    with open('alexnet_regularized.pkl', 'rb') as f:
        alexnet_regularized_student = pickle.load(f)
        
    test_train_deviation = alexnet_regularized_student['train_accuracy_per_epoch'][-1] -\
        alexnet_regularized_student['test_accuracy']
    
    assert len(alexnet_regularized_student['train_accuracy_per_epoch']) == 21, \
    "Model not trained for 20 epochs"
    assert alexnet_regularized_student['valid_accuracy_per_epoch'][-1] > 0.81, \
    "Accuracy too low"
    assert alexnet_regularized_student['accuracy_deviation'][-1] < 4.0, \
    "Train/validation deviation too high"
    assert test_train_deviation < 4.0, "Train/test deviation too high"


# pytest.fail
# FileNotFoundError

pytest.main(["--tb=line"])
