from functools import partial
from torch import float64 as f64
from fastai.metrics import (
    fbeta
)


def accuracy_segmentation(inp, target):
    target = target.squeeze(1)
    return (inp.argmax(dim=1) == target).float().mean()


def jaccard_index(inp, target, base_value: int) -> float:
    target = target.squeeze(1)
    inp = inp.argmax(dim=1)
    intersection = ((inp == base_value) & (target == base_value)).unique(return_counts=True)
    union = ((inp == base_value) | (target == base_value)).unique(return_counts=True)
    return (intersection[1][1].to(f64) / union[1][1].to(f64)) * 100.0


def jaccard_index_zero_class(inp, target) -> float:
    return jaccard_index(inp, target, 0)


def jaccard_index_one_class(inp, target) -> float:
    return jaccard_index(inp, target, 1)


def test_showing(inp, target):
    print("Input type: ", type(inp))
    print("Target type: ", type(target))
    try:
        print("Input len: ", len(inp))
        print("Target len: ", len(target))

    except ...:
        print("Type without shape")


def f1_score():
    return partial(fbeta, thresh=0.2, beta=1)
