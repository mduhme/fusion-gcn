import copy
import os


class FileMetaData:
    """
    Stores file name, subject action label and other properties for each file.
    """
    def __init__(self, fn: str, subject: int, action: int, **properties):
        assert subject >= 0 and action >= 0
        self.file_name = fn
        self.subject = subject
        self.action = action
        self.properties = copy.deepcopy(properties)
        for name, prop in self.properties.items():
            setattr(self, name, prop)

    def __str__(self):
        return os.path.splitext(os.path.basename(self.file_name))[0]
