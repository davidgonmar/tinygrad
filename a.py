from tinygrad import Tensor
import tinygrad as tg
import numpy as np


inta = np.array(0b11111111111111111111111111111111).astype(np.int32)
print(inta)
intafirsthalf = np.array(0b1111111111111111).astype(np.int16)
intasecondhalf = np.array(0b1111111111111111).astype(np.int16)
print(intafirsthalf)
print(intasecondhalf)

print(type(intafirsthalf))

shape = (2000, 1000)
a = Tensor([inta] * np.prod(shape), dtype=tg.dtypes.int32).reshape(*shape).T

# try bitcast
b = a.bitcast(tg.dtypes.int16)
assert tuple(b.shape) == (shape[0], shape[1] * 2)

# Create bexpected with np.int16 type explicitly
bexpected = np.array([[intasecondhalf, intafirsthalf] * shape[1]] * shape[0], dtype=np.int16)

print(b.numpy())
print(bexpected)
np.testing.assert_array_equal(b.numpy(), bexpected)
