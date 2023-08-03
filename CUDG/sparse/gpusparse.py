from VRAManager import VRAM, cuda

from numpy import ndarray
from numpy import array, empty
from numpy import int32, float32



class gpusparse:


    def __init__( self, shape:tuple, gpudata: ndarray ):
        '''
        GPU Sparse Array Type
        '''
        self.shape = shape

        self.n_row = int32( shape[0] )
        self.n_col = int32( shape[1] )

        _sparse = self._be_sparse( gpudata )

        self.ary_size = _sparse.shape[1]

        self.idxdata = VRAM._request( self.ary_size )
        self.gpudata = VRAM._request( self.ary_size )


    def __del__( self ):
        pass


    def _be_sparse( self, gpudata:ndarray ):
        n_row = self.shape[0]
        n_col = self.shape[1]
        tol = 10e-38
        idx = []
        val = []

        for row in n_row:
            for col in n_col:
                val_rc = gpudata[row,col]
                if ( abs( val_rc < tol ) ):
                    idx.append( row * n_col + col )
                    val.append( val_rc )
        
        return array( [idx, val], dtype=float32 )

    
    def _to_gpu( self, gpudata: ndarray ):

        pass

    
    def to_gpu( self, gpudata: ndarray ):

        _sparse = self._be_sparse( gpudata )

        if ( self.ary_size == _sparse.shape[1] ):

            cuda.memcpy_htod( self.idxdata, _sparse[0,:].astype(int32)   )
            cuda.memcpy_htod( self.gpudata, _sparse[1,:].astype(float32) )