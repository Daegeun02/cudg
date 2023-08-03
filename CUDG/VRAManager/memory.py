import pycuda.driver as cuda



class Memory:

    type = cuda.DeviceAllocation


    def _request( self, size: int ) -> cuda.DeviceAllocation:
        '''
        for each number has 4 byte 
        because GPU uses 32bit
        '''
        return cuda.mem_alloc( size * 4 )

    
    def _filter( self, gpudata ):
        pass


    def _return( self, mem: cuda.DeviceAllocation ) -> None:

        try:
            mem.free()
        except:
            print( 'Memory Link might be happened' )
            print( 'Recommand to restart kernel' )


VRAM = Memory()