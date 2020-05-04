from functools import partial
from fastai.metrics import (
    fbeta
)


def accuracy_segmentation(inp, target):
    target = target.squeeze(1)
    return (inp.argmax(dim=1) == target).float().mean()


def f1_score():
    return partial(fbeta, thresh=0.2, beta=1)
