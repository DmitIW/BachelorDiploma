from functools import partial

from data_utils import set_data_environment

path_to_data = "/home/dmitri/Documents/Datasets/skyFinder"
data_dirs = set_data_environment(path_to_data, subdirs={
    "train_images": "train",
    "train_labels": "train_labels",
    "train_images_p": "train_p",
    "train_p_labels": "train_p_labels",
    "val_images": "val",
    "val_labels": "val_labels",
    "transfer vertical": "td_vertical",
    "transfer horizontal": "td_horizontal"
})
data_dirs.set_translation(
    lambda image_name: f"{image_name.stem}_L.png"
)


def get_label(image, category):
    return data_dirs.get_label(image, category + "_labels")


def get_data_paths(category):
    result = {
        "images": [],
        "labels": []
    }
    for image in data_dirs.get_subdir(category + "_images").iterdir():
        result["images"].append(image)
        result["labels"].append(get_label(image, category))
    print("Dataset size: ", len(result["images"]))
    return result


get_label_train = partial(get_label, category="train")
get_label_val = partial(get_label, category="val")


def get_label_with_context(image_file):
    return get_label(image_file, image_file.parent.stem)


__all__ = [
    data_dirs,
    get_data_paths,
    get_label_train,
    get_label_val,
    get_label_with_context
]
