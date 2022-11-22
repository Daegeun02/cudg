from numpy  import float32, inf

from .kernels.ufunc import *


## compile kernel functions
_clip = clip_kernel()
_sum  = sum_kernel()
_copy = copy_kernel()



def clip(ary, lwbd, upbd, stream=None):
    if (upbd == inf):
        upbd = 1e38
    if (lwbd == -inf):
        lwbd = -1e38

    _clip(
        ary, ary.n_row, ary.n_col,
        float32(upbd), float32(lwbd),
        block=ary._block, grid=ary._grid, stream=stream
    )


def sum(result, ary):
    _block = (ary.shape[0],1,1)
    _grid  = (1,1)

    _sum(
        result,
        ary, ary.n_col,
        block= _block, grid=_grid
    )

    return result.to_host()


def copy(_to, _from, stream=None):
    shpt = _to.shape
    shpf = _from.shape

    if (shpf != shpt):
        raise ValueError("can not copy array between different size")

    _copy(
        _to, _to.n_row, _to.n_col,
        _from,
        block=_to._block, grid=_to._grid, stream=stream
    )
