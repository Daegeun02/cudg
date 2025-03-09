## exponential function
## kernel function

from pycuda.compiler import SourceModule



def forward_kernel():
    kernel_code = \
    """
    #include <math.h>

    #define tx (threadIdx.x)
    #define ty (threadIdx.y)
    #define bx (blockIdx.x)
    #define by (blockIdx.y)
    
    __global__ void forward(
        float* y, 
        float* x,
        int n_row, int n_col,
    ) {
        const int n_ = 32;
        
        int _row = tx + bx * n_;
        int _col = ty + by * n_;

        if ((_row < n_row) && (_col < n_col)) {
            int idx = _row * n_col + _col;

            y[idx] = exp(x[idx]);
        }

        __syncthreads();
    }
    """
    kernel = SourceModule(kernel_code)

    return kernel.get_function("forward")

def backward_kernel():
    kernel_code = \
    """
    #include <math.h>

    #define tx (threadIdx.x)
    #define ty (threadIdx.y)
    #define bx (blockIdx.x)
    #define by (blockIdx.y)
    
    __global__ void backward(
        float* dx, 
        float* dy, float* x,
        int n_row, int n_col,
   ) {
        const int n_ = 32;
        
        int _row = tx + bx * n_;
        int _col = ty + by * n_;

        if ((_row < n_row) && (_col < n_col)) {
            int idx = _row * n_col + _col;

            dx[idx] = 1;
            dx[idx] *= exp(x[idx]);
            dx[idx] *= dy[idx];
        }

        __syncthreads();
   }
    """
    kernel = SourceModule(kernel_code)

    return kernel.get_function("backward")