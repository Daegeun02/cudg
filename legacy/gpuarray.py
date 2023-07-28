from .gpudata import GPUDATA



class gpuarray(GPUDATA):

    def __init__(self, shape, gpudata=None, iscache=False):
        ## basic information
        self.shape = shape

        ## get memory space
        super().__init__(shape, iscache=iscache)

        ## to support Deep Learning
        self.grad = None

        ## fill gpudata
        if (type(gpudata) == None.__class__):
            pass
        elif isinstance(gpudata, gpuarray):
            self.gpudata = gpudata
        else:
            self.to_gpu(gpudata)

    def to_host(self):
        gpudata = super().to_host()

        return gpudata.reshape(self.shape)



if __name__ == "__main__":
    a = gpuarray((10,10))

    print(a.shape)
    print(a.n_row)
    print(a.n_col)
    print(a.gpudata)
    print(a._block)
    print(a._grid)
