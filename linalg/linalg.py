from math import sqrt, ceil
from numpy import int32, float32

from .linalgcore import LinearAlgebraCore
from .linalgcore import dot_kernel



_dot = dot_kernel()

def dot( rtrn, ary1, ary2, stream=None ):
    shp1 = ary1.shape
    shp2 = ary2.shape
    shpr = rtrn.shape

    if ( ( shp1[1] != shp2[0] ) or ( shpr != (shp1[0],shp2[1]) ) ):
        raise ValueError( "size doesn't fit" )

    tx = ary1.tx
    it = ary1.it
    
    _block = (tx, 1, 1)
    _grid  = (shp1[0],shp2[1])

    _dot(
        rtrn.gpudata, it,
        ary1.gpudata, ary1.n_row, ary1.n_col,
        ary2.gpudata, ary2.n_row, ary2.n_col,
        block=_block, grid=_grid
    )


class LinearAlgebra:
    core = LinearAlgebraCore()

    def __init__(self):
        pass

    def dot(self, result, ary1, ary2, stream=None):
        shp1 = ary1.shape
        shp2 = ary2.shape
        shpr = result.shape
        if ((shp1[1] != shp2[0]) or (shpr != (shp1[0],shp2[1]))):
            raise ValueError("size doesn't fit")
        ## optimal kernel size
        tx = int(sqrt(shp1[1]))
        _block = (tx, 1, 1)
        _grid  = (shp1[0],shp2[1])

        iters = int32(ceil(shp1[1] / tx))

        self.__class__.core.dot(
            result, iters,
            ary1, ary1.n_row, ary1.n_col,
            ary2, ary2.n_row, ary2.n_col,
            block=_block, grid=_grid, stream=stream
        )

    def norm(self, ary, axis=None, result=None, stream=None):
        shp = ary.shape

        self.__class__.core.norm(
        )

    """
    def vstack(self, *arys, result=None):
        if (result == cudg.array):
            shpr = result.shape
        rows = 0
        for ary in arys:
            shpi = ary.shape
            if (shpr[1] != shpi[1]):
                raise ValueError("NO")
            rows += shpi[0]
        if (shpr[0] != rows):
            raise ValueError("NO")

        self.__class__.core.vstack(
            result
        )
    """
