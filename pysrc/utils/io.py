# %%
import struct
import numpy as np


def save_np_to_file(file_name, array):
    """Assumes array contains float32 values"""
    array = array.astype(np.float32)
    with open(file_name, 'wb') as f:
        f.write(struct.pack("i", int(array.ndim)))
        for s in array.shape:
            print(f"s: {s}")
            f.write(struct.pack("i", int(s)))
        f.write(array.tobytes())


def load_np_from_file(file_name):
    with open(file_name, 'rb') as f:
        ndims, = struct.unpack('i', f.read(4))
        shape = []
        print(f"ndims: {ndims}")
        for s in range(ndims):
            shape.append(struct.unpack('i', f.read(4))[0])

        values = np.fromfile(f, dtype=np.float32, count=np.prod(shape))
        values = np.reshape(values, shape)
        return values


a = np.arange(5) + 0.2
a = np.random.rand(3, 2)
file_name = '/tmp/nptest.bin'
save_np_to_file(file_name, a)
a2 = load_np_from_file(file_name)
print(f"a: {a}")
print(f"a2: {a2}")