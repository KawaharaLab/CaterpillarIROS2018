from collections import abc


class DataCSVSaver:
    def __init__(self, file_path: str, columns: abc.Iterable):
        # columns: ["col1", "col2", ...]
        self.__file_path = file_path
        self.__columns = columns
        self.__line_placeholder_str = "{}" + ",{}" * (len(self.__columns) - 1) + "\n"

        with open(self.__file_path, 'w') as f:
            header = self.__line_placeholder_str.format(*self.__columns)
            f.write(header)

    def append_data(self, *items):
        assert len(items) == len(self.__columns)
        with open(self.__file_path, 'a') as f:
            line = self.__line_placeholder_str.format(*items)
            f.write(line)

    @property
    def colums(self) -> tuple:
        return tuple(self.__columns)
