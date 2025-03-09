## square function
## kernel function

from pycuda.compiler import SourceModule



def Iforward_kernel():
    kernel_code = \
    """
    #define tx (threadIdx.x)
    #define ty (threadIdx.y)
    #define bx (blockIdx.x)
    #define by (blockIdx.y)

    __global__ void Iforward(
        float* INOUT,
        int n_row, int n_col
   ) {
        const int n_ = 32;
        
        int _row = tx + bx * n_;
        int _col = ty + by * n_;

        if ((_row < n_row) && (_col < n_col)) {
            int idx = _row * n_col + _col;

            INOUT[idx] = INOUT[idx] * INOUT[idx];
        }

        __syncthreads();
   }
    """
    kernel = SourceModule(kernel_code)

    return kernel.get_function("Iforward")

def Oforward_kernel():
    kernel_code = \
    """
    #define tx (threadIdx.x)
    #define ty (threadIdx.y)
    #define bx (blockIdx.x)
    #define by (blockIdx.y)

    __global__ void Oforward(
        float* OUT,
        float* IN0,
        int n_row, int n_col
   ) {
        const int n_ = 32;
        
        int _row = tx + bx * n_;
        int _col = ty + by * n_;

        if ((_row < n_row) && (_col < n_col)) {
            int idx = _row * n_col + _col;

            OUT[idx] = IN0[idx] * IN0[idx];
        }

        __syncthreads();
   }
    """
    kernel = SourceModule(kernel_code)

    return kernel.get_function("Oforward")

def backward_kernel():
    kernel_code = \
    """
    #define tx (threadIdx.x)
    #define ty (threadIdx.y)
    #define bx (blockIdx.x)
    #define by (blockIdx.y)

    __global__ void backward(
        float* dIN, 
        float* dOUT, float* IN0,
        int n_row, int n_col
   ) {
        const int n_ = 32;
        
        int _row = tx + bx * n_;
        int _col = ty + by * n_;

        if ((_row < n_row) && (_col < n_col)) {
            int idx = _row * n_col + _col;

            dIN[idx] = 2;
            dIN[idx] *= IN0[idx];
            dIN[idx] *= dOUT[idx];
        }

        __syncthreads();
   }
    """
    kernel = SourceModule(kernel_code)

    return kernel.get_function("backward")