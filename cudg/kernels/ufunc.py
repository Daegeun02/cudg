from pycuda.compiler import SourceModule



def clip_kernel():
        
    ## block=(16,16,1), grid=(bx,by)
    kernel_code = \
    """
    #define tx (threadIdx.x)
    #define ty (threadIdx.y)
    #define bx (blockIdx.x)
    #define by (blockIdx.y)

    __global__ void clip(
        float* ary, int n_row, int n_col,
        float upbd, float lwbd
    ) {
        const int n_ = 16;

        int _row = tx + bx * n_;
        int _col = ty + by * n_;

        if ((_row < n_row) && (_col < n_col)) {
            int idx = _row * n_col + _col;

            if (ary[idx] > upbd) {
                ary[idx] = upbd;
            }
            if (ary[idx] < lwbd) {
                ary[idx] = lwbd;
            }
            __syncthreads();
        }
    }
    """
    kernel = SourceModule(kernel_code)

    return kernel.get_function("clip")

def sum_kernel():
    ## block=(n_row,1,1), grid=(1,1)
    kernel_code = \
    """
    #define _row (threadIdx.x)
    #define n_row (blockDim.x)

    __global__ void sum(
        float* result,
        float* ary, int n_col
    ) {
        __shared__ float sM[1024];

        sM[_row] = 0;

        if (n_col > 1) {
            if (_row < n_row) {
                int idx;
                for (int i = 0; i < n_col; i++) {
                    idx = _row * n_col + i;
                    sM[_row] += ary[idx];
                }
            }
        }
        else if (n_col == 1) {
            if (_row < n_row) {
                sM[_row] += ary[_row];
            }
        }
        __syncthreads();

        if (_row == 0) {
            result[0] = 0;
            for (int i = 0; i < n_row; i++) {
                result[0] += sM[i];
            }
        }
        __syncthreads();
    }
    """
    kernel = SourceModule(kernel_code)

    return kernel.get_function("sum")

def copy_kernel():
    kernel_code = \
    """
    #define tx (threadIdx.x)
    #define ty (threadIdx.y)
    #define bx (blockIdx.x)
    #define by (blockIdx.y)

    __global__ void copy(
        float* result, int n_row, int n_col,
        float* ary
    ) {
        const int n_ = 16;

        int _row = tx + bx * n_;
        int _col = ty + by * n_;

        if ((_row < n_row) && (_col < n_col)) {
            int idx = _row * n_col + _col;

            result[idx] = ary[idx];
        }

        __syncthreads();
    }
    """
    kernel = SourceModule(kernel_code)

    return kernel.get_function("copy")
