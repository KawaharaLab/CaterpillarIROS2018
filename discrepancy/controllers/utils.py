from datetime import datetime
import os


def notice(notice_text):
    notice_style = "\x1b[0;31;45m Notice: {} \x1b[0m"
    print(notice_style.format(notice_text))


def time() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class SaveDir:
    def __init__(self, root: str):
        self.__root = root
        self.__log_dir = None

    def log_dir(self) -> str:
        if self.__log_dir is None:
            self.__log_dir = self._create_log_dir()
        return self.__log_dir

    def _create_log_dir(self) -> str:
        log_dir = "{}/train_log".format(self.__root)
        os.makedirs(self.__log_dir)
        print("Created {}".format(self.__log_dir))
        return log_dir
