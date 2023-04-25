import os
import sys
from testing.importing import NotebookFinder
sys.meta_path.append(NotebookFinder())
import assignment_detection as student

import pytest

import torch
import torch.nn as nn

from testing.utils import intersection_over_union
from testing.tools import compare_models

def test_yolov3():
    num_classes = 20
    model = student.YOLOv3(num_classes=num_classes)
    img_size = 416
    x = torch.randn((2, 3, img_size, img_size))
    out = model(x)
    
    assert out[0].shape == (2, 3, img_size//32, img_size//32, 5 + num_classes), "Shape of feature_map_1 is wrong"
    assert out[1].shape == (2, 3, img_size//16, img_size//16, 5 + num_classes), "Shape of feature_map_2 is wrong"
    assert out[2].shape == (2, 3, img_size//8, img_size//8, 5 + num_classes), "Shape of feature_map_3 is wrong"
    

def test_iou():
    box1_1 = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    box1_2 = torch.tensor([[0.75, 0.75, 0.5, 0.5]])
    box2_1 = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
    box2_2 = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
    
    iou_1_student = student.iou(box1_1, box1_2)
    iou_1_correct = intersection_over_union(box1_1, box1_2)
    iou_2_student = student.iou(box2_1, box2_2)
    iou_2_correct = intersection_over_union(box2_1, box2_2)
    
    assert torch.allclose(iou_1_correct, iou_1_student), "Calculation of iou is wrong"
    assert iou_2_correct.shape == iou_2_student.shape, "Shape of iou is wrong"
    assert iou_2_correct[0] == iou_2_student[0], "Be careful of the case when there is nan in the iou"
    
    
def test_load_checkpoint():
    path = 'test_model.pth'
    model = nn.Sequential(nn.Linear(10, 100))
    torch.save(model.state_dict(), path)
    model_student = student.load_checkpoint(path, model)
    
    assert compare_models(model, model_student, True), "Loading not correct"
    
    model = nn.Sequential(nn.Linear(10, 100))
    
    assert not compare_models(model, model_student, True), "Loading not correct"


def test_save_checkpoint():
    path = 'test_path.pth'
    model = nn.Sequential(nn.Linear(10, 100))
    student.save_checkpoint(model, path)
    assert os.path.isfile(path), "Saving not implemented"

pytest.main(["--tb=line"])
