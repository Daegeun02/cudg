## cudg gpuarray type


## memory manager
from .cache import Cache
cache = Cache()

## cudg gpuarray class
from .gpudata  import GPUDATA
from .gpuarray import gpuarray


## basic 4 arithmetic operation
from .function import add, sub, mul
gpuarray.__add__ = add
gpuarray.__sub__ = sub
gpuarray.__mul__ = mul


## universial functions
from .core import *


## linear algebra
from .linalg import linalg
