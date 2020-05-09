from random import uniform

from fastai.vision import (
    SegmentationItemList, SegmentationLabelList, open_mask, ImageList, open_image
)
from fastai.vision.transform import (
    dihedral
)


class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)


class SegItemListCustom(SegmentationItemList):
    _label_cls = SegLabelListCustom


class ImageListVertical(ImageList):
    def open(self, fn): return dihedral(open_image(fn), k=5)


class SegmentationItemListTwoSourceFoldersProb:
    def __init__(self, second_folder_probability: float = 0.9, **kwargs):
        self._segILFirst = SegItemListCustom
        self._segILSecond = SegItemListCustom
        if (second_folder_probability > 1.0) or (second_folder_probability < 0.0):
            raise RuntimeError("Not correct probability for second folder getting")
        self._second_folder_probability = second_folder_probability
        self._segILFirstLabeled = None
        self._segILSecondLabeled = None
        self._split_by_folder = False
        self._transform = False
        self._ready = False
        self.c = 0

    def _second_folder(self):
        return uniform(0.0, 1.0) > (1.0 - self._second_folder_probability)

    def _data_ready(self):
        if not self._ready:
            raise RuntimeError("Data loader not ready yet")

    def get(self, i):
        self._data_ready()
        if self._second_folder():
            return self._segILSecondLabeled.get(i)
        return self._segILFirstLabeled.get(i)

    def reconstruct(self, t):
        self._data_ready()
        return self._segILSecond.reconstruct(t)

    def show_xys(self, xs, ys, figsize: tuple = (12, 6), **kwargs):
        self._data_ready()
        self._segILFirstLabeled.show_xys(xs, ys, figsize=figsize, **kwargs)

    def show_xyzs(self, xs, ys, zs, figsize: tuple = None, **kwargs):
        self._data_ready()
        self._segILFirstLabeled.show_xyzs(xs, ys, zs, figsize=figsize, **kwargs)

    def analyze_pred(self, pred):
        return self._segILFirst.analyze_pred(pred)

    def from_folders(self, firstFolder, secondFolder=None, **kwargs):
        self._segILFirst = self._segILFirst.from_folder(firstFolder, **kwargs)
        if secondFolder is None:
            self._segILSecond = self._segILSecond.from_folder(firstFolder, **kwargs)
        else:
            self._segILSecond = self._segILSecond.from_folder(secondFolder, **kwargs)

    def split_by_folder(self, first_train, first_val, second_train, second_val, first_test=None, second_test=None,
                        **kwargs):
        self._segILFirst = self._segILFirst.split_by_folder(train=first_train, valid=first_val)
        self._segILSecond = self._segILSecond.split_by_folder(train=second_train, valid=second_val)
        self._split_by_folder = True

    def label_from_func(self, func, classes, **kwargs):
        if self._split_by_folder:
            self._segILFirstLabeled = self._segILFirst.label_from_func(func=func, classes=classes)
            self._segILSecondLabeled = self._segILSecond.label_from_func(func=func, classes=classes)
        else:
            raise RuntimeError("Not been splited yet")

    def transform(self, tfms, size, tfm_y, **kwargs):
        if self._segILFirstLabeled is None:
            raise RuntimeError("Not been labeled yet")
        if self._segILSecondLabeled is None:
            raise RuntimeError("Not been labeled yet")
        self._segILFirstLabeled = self._segILFirstLabeled.transform(tfms=tfms, size=size, tfm_y=tfm_y, **kwargs)
        self._segILSecondLabeled = self._segILSecondLabeled.transform(tfms=tfms, size=size, tfm_y=tfm_y, **kwargs)
        self._transform = True

    def databunch_and_normalized(self, bs, stats, **kwargs):
        if self._transform:
            self._segILFirstLabeled = self._segILFirstLabeled.databunch(bs=bs).normalize(stats)
            self._segILSecondLabeled = self._segILSecondLabeled.databunch(bs=bs).normalize(stats)
            if self._segILSecondLabeled.c != self._segILFirstLabeled.c:
                raise RuntimeError("Miscasting classes num between folders")
            self.c = self._segILSecondLabeled.c
            self._ready = True
            return self
        else:
            raise RuntimeError("Not been transformed yet")

    def show_batch(self, **kwargs):
        self._data_ready()
        if self._second_folder():
            self._segILSecondLabeled.show_batch(**kwargs)
        else:
            self._segILFirstLabeled.show_batch(**kwargs)
