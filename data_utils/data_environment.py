from pathlib import (
    Path
)


class DataEnv:
    def __init__(self):
        self._data_dirs: dict = {}
        self._image2label_names_mapping: callable = None

    def set_root(self, root_dir: str) -> None:
        self._data_dirs["root"] = Path(root_dir)

    def set_subdir(self, subdir: str, subdir_name: str = None) -> None:
        name = subdir
        if subdir_name is not None:
            name = subdir_name
        self._data_dirs[name] = self._data_dirs["root"] / subdir

    def set_translation(self, map_func: callable):
        self._image2label_names_mapping = map_func

    def get_root(self) -> Path:
        return self._data_dirs["root"]

    def get_subdir(self, subdir_name: str = None) -> Path:
        return self._data_dirs[subdir_name]

    def get_label_name(self, image: Path) -> str:
        return self._image2label_names_mapping(image)

    def get_label(self, image: Path, labels_subdir_name: Path):
        return self._data_dirs[labels_subdir_name] / self.get_label_name(image)


def set_data_environment(data_dir: str, subdirs: dict) -> DataEnv:
    data_env = DataEnv()
    data_env.set_root(data_dir)
    for key, value in subdirs.items():
        data_env.set_subdir(value, key)

    return data_env
