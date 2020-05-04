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