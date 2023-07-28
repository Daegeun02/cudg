## function source file

## import
from numpy import float32

## memory utilize
from .cache    import Cache

## kernel functions
from .kernels import addsub
from .kernels import mul



## four arithmetic operations
class Function:

    cache = Cache()
    dtype = cache.dtype

    ##====================##
    ## IN1 (operator) IN2 ##
    ##====================##

    def __call__(self, Icoef1, IN1, Icoef2, IN2):
        if (IN1.shape != IN2.shape):
            raise ValueError("array operation must be done with same shape")
        
        gpuarray = self.__class__.dtype

        if ((not isinstance(IN1, gpuarray)) or (not isinstance(IN2, gpuarray))):
            raise ValueError("Not supported for non gpuarray type data")

        ## output must be cache
        if   IN1.iscache:
            OUT = IN1
            self.Iforward(Icoef1, OUT, Icoef2, IN2)
        elif IN2.iscache:
            OUT = IN2
            self.Iforward(Icoef2, OUT, Icoef1, IN1)
        else:
            OUT = self.__class__.cache.request(IN1.shape)
            self.Oforward(OUT, Icoef1, IN1, Icoef2, IN2)
        
        return OUT

    def Iforward(self, Ocoef, OUT, Icoef, IN0):
        raise NotImplementedError("It's not supported yet, sorry")

    def Oforward(self, OUT, Icoef1, IN1, Icoef2, IN2):
        raise NotImplementedError("It's not supported yet, sorry")

    def backward(self, dOUT):
        raise NotImplementedError("It's not supported yet, sorry")

#####################################################################

class Add(Function):

    _Iforward = addsub.Iforward_kernel()
    _Oforward = addsub.Oforward_kernel()

    def Iforward(self, Ocoef, OUT, Icoef, IN0):

        self.__class__._Iforward(
            OUT, Ocoef,
            IN0, Icoef,
            OUT.n_row, OUT.n_col,
            block=OUT._block, grid=OUT._grid
        )

    def Oforward(self, OUT, Icoef1, IN1, Icoef2, IN2):

        self.__class__._Oforward(
            OUT,
            IN1, Icoef1, IN2, Icoef2,
            OUT.n_row, OUT.n_col,
            block=OUT._block, grid=OUT._grid
        )

def add(IN1, IN2):
    return Add()(float32(1), IN1, float32(1), IN2)

#####################################################################

class Sub(Function):

    _Iforward = addsub.Iforward_kernel()
    _Oforward = addsub.Oforward_kernel()

    def Iforward(self, Ocoef, OUT, Icoef, IN0):

        self.__class__._Iforward(
            OUT, Ocoef,
            IN0, Icoef,
            OUT.n_row, OUT.n_col,
            block=OUT._block, grid=OUT._grid
        )

    def Oforward(self, OUT, Icoef1, IN1, Icoef2, IN2):

        self.__class__._Oforward(
            OUT,
            IN1, Icoef1, IN2, Icoef2,
            OUT.n_row, OUT.n_col,
            block=OUT._block, grid=OUT._grid
        )

def sub(IN1, IN2):
    return Sub()(float32(1), IN1, float32(-1), IN2)

#####################################################################

class Mul(Function):

    _Iforward = mul.Iforward_kernel()
    _Oforward = mul.Oforward_kernel()

    def Iforward(self, Ocoef, OUT, Icoef, IN0):

        self.__class__._Iforward(
            OUT,
            IN0,
            OUT.n_row, OUT.n_col,
            block=OUT._block, grid=OUT._grid
        )

    def Oforward(self, OUT, Icoef1, IN1, Icoef2, IN2):

        self.__class__._Oforward(
            OUT,
            IN1, IN2,
            OUT.n_row, OUT.n_col,
            block=OUT._block, grid=OUT._grid
        )

def mul(IN1, IN2):
    return Mul()(None, IN1, None, IN2)
