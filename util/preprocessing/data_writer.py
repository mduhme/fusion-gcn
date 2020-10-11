import numpy as np
import numpy.lib.format


class MemoryMappedArray:
    def __init__(self, out_path: str, dtype: type, shape: tuple):
        self.out_path = out_path
        self.dtype = dtype
        self.shape = shape
        self.data = None

    def create_file(self):
        self.data = np.memmap(self.out_path, self.dtype, "w+", 128, self.shape)

    def close_file(self):
        if self.data is not None:
            MemoryMappedArray._write_header(self.data, self.out_path)
        self.data = None

    def __enter__(self):
        self.create_file()
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_file()

    @staticmethod
    def _write_header(data: np.ndarray, out_path: str):
        header = np.lib.format.header_data_from_array_1_0(data)
        with open(out_path, "r+b") as file:
            np.lib.format.write_array_header_1_0(file, header)
