from datetime import datetime
import shutil
import os


def notice(notice_text):
    notice_style = "\x1b[0;31;45m Notice: {} \x1b[0m"
    print(notice_style.format(notice_text))


def time() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def reset_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


class SaveDir:
    def __init__(self, root: str):
        self.__root = root
        self.__log_dir = None
        self.__model_dir = None

    def log_dir(self) -> str:
        if self.__log_dir is None:
            self.__log_dir = self._create_log_dir()
        return self.__log_dir

    def _create_log_dir(self) -> str:
        log_dir = "{}/train_log".format(self.__root)
        reset_dir(log_dir)
        print("{} created".format(log_dir))
        return log_dir

    def model_dir(self) -> str:
        if self.__model_dir is None:
            self.__model_dir = self._create_model_dir()
        return self.__model_dir

    def _create_model_dir(self) -> str:
        model_dir = "{}/model".format(self.__root)
        reset_dir(model_dir)
        print("{} created".format(model_dir))
        return model_dir
