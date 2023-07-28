import pycuda.autoinit
import pycuda.driver as cuda

from numpy import int32, float32, ndarray, empty

from functools import reduce



class GPUDATA:

    num_of_ary = 0
    
    def __init__(self, shape, iscache=False):
        ## it's important
        self.ary_size = reduce(lambda x, y: x * y, shape)
        self.buf_size = self.ary_size * 4    ## 4 bytes
        self.iscache = iscache

        ## shape data
        self.n_row = int32(shape[0])
        self.n_col = int32(shape[1])

        ## initialize
        self.gpudata = cuda.mem_alloc(self.buf_size)

        ## for monitoring memory usage
        self.__class__.num_of_ary += 1
        # print("allocation" + f"{self.__class__.num_of_ary}")
        
        self.kernel_size()

    def __del__(self):
        try:
            self.gpudata.free()

            ## for monitoring memory usage
            # print("free" + f"{self.__class__.num_of_ary}")
            self.__class__.num_of_ary -= 1
        except:
            print("memory link might be happened")

    def to_gpu(self, gpudata):
        ## only 4 bytes data type allowed
        if isinstance(gpudata, ndarray):
            if (gpudata.itemsize == 4):
                pass
            elif (gpudata.itemsize == 8):
                gpudata = gpudata.astype(float32)
            else:
                raise ValueError()
        else:
            raise ValueError("only for np.ndarray")

        ## copy memory to GPU
        cuda.memcpy_htod(self.gpudata, gpudata)

    def to_host(self):
        gpudata = empty((self.ary_size), dtype=float32)
        cuda.memcpy_dtoh(gpudata, self.gpudata)

        return gpudata

    ## block, grid size
    def kernel_size(self):

        bx = int(self.n_row / 32) + 1
        by = int(self.n_col / 32) + 1

        self._block = (32,32,1)
        self._grid  = (bx,by,1)