from pycuda.compiler import SourceModule

def dot_kernel():
    ## block=(tx,1,1), grid=(row,col)
    kernel_code = \
    """
    #define tx (threadIdx.x)
    #define bx (blockIdx.x)
    #define by (blockIdx.y)
    #define N_ (blockDim.x)

    __global__ void dot(
        float* rtrn, int iters,
        float* ary1, int n_row1, int n_col1,
        float* ary2, int n_row2, int n_col2
    ) {
        __shared__ float sM[1024];
        sM[tx] = 0;

        int idxi;
        int idx1;
        int idx2;
        int idxr;

        for ( int i = 0; i < iters; i++ ) {
            idxi = tx * iters + i;

            if ( idxi < n_row2 ) {
                idx1 = bx * n_col1 + idxi;
                idx2 = by + n_col2 * idxi;

                sM[tx] += ( ary1[idx1] * ary2[idx2] );
            }
        }
        __syncthreads();

        if ( tx == 0 ) {
            idxr = bx * n_col2 + by;
            rtrn[idxr] = 0;
            for ( int i = 0; i < N_; i++ ) {
                rtrn[idxr] += sM[i];
            }
        }
        __syncthreads();
    }
    """
    kernel = SourceModule( kernel_code )
    
    return kernel.get_function( "dot" )


class LinearAlgebraCore:

    def __init__(self):
        self.dot = self.dot_kernel()
        # self.norm = self.norm_kernel()

    def dot_kernel(self):
        ## block=(tx,1,1), grid=(row,col)
        kernel_code = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)
        #define by (blockIdx.y)
        #define N_ (blockDim.x)

        __global__ void dot(
            float* result, int iteration,
            float* ary1, int n_row1, int n_col1,
            float* ary2, int n_row2, int n_col2
        ) {
            __shared__ float sM[1024];
            sM[tx] = 0;

            int idxi;
            int idx1;
            int idx2;
            int idxr;

            for (int i = 0; i < iteration; i++) {
                idxi = tx * iteration + i;

                if (idxi < n_row2) {
                    idx1 = bx * n_col1 + idxi;
                    idx2 = by + n_col2 * idxi;

                    sM[tx] += (ary1[idx1] * ary2[idx2]);
                }
            }
            __syncthreads();

            if (tx == 0) {
                idxr = bx * n_col2 + by;
                result[idxr] = 0;
                for (int i = 0; i < N_; i++) {
                    result[idxr] += sM[i];
                }
            }
            __syncthreads();
        }
        """
        kernel = SourceModule(kernel_code)
        
        return kernel.get_function("dot")

    def norm_kernel(self):
        ## block=(n_row,1,1), grid=(1,1,1)
        kernel_code = \
        """
        __device__ float _sqrt(float value) {
            float s = 0;
            float t = 0;

            s = value / 2;

            for (;s != t;) {
                t = s;
                s = ((value/t) + t) / 2;
            }

            return s;
        }

        __global__ void _norm(
            float* result, 
        ) {
        }
        """
        kernel = SourceModule(kernel_code)

        return kernel.get_function("_norm")
