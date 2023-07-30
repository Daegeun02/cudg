from ..VRAManager import VRAM, cuda

from numpy import ndarray
from numpy import empty
from numpy import int32, float32

from math import sqrt, ceil

from functools import reduce



class gpuarray:


    def __init__( self, shape: tuple, gpudata: ndarray ):
        '''
        GPU Array Type
        '''
        self.shape    = shape
        self.ary_size = reduce( lambda x, y: x * y, shape )

        self.n_row = int32( shape[0] )
        self.n_col = int32( shape[1] )

        self.gpudata = VRAM._request( self.ary_size )

        self.to_gpu( gpudata )

        self.kernel_size()

    
    def __del__( self ):

        VRAM._return( self.gpudata )

    
    def to_gpu( self, gpudata: ndarray ):

        if isinstance( gpudata, ndarray ):
            if ( gpudata.itemsize == 4 ):
                pass
            elif ( gpudata.itemsize == 8 ):
                gpudata = gpudata.astype( float32 )
            else:
                raise ValueError( 'somethings strange' )
        else:
            raise ValueError( 'not numpy.ndarray' )

        cuda.memcpy_htod( self.gpudata, gpudata )

    
    def to_host( self ):
        gpudata = empty( ( self.ary_size ), dtype=float32 )

        cuda.memcpy_dtoh( gpudata, self.gpudata )

        return gpudata.reshape( self.shape )


    def kernel_size( self ):

        bx = int( self.n_row / 32 ) + 1
        by = int( self.n_col / 32 ) + 1

        self._block = (32,32,1)
        self._grid  = (bx,by,1)

        self.tx = int( sqrt( self.shape[1] ) )
        self.it = int32( ceil( self.shape[1] / self.tx ) )