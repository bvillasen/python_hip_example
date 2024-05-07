import numpy as np
import ctypes
from ctypes import c_int32, c_double, c_void_p
import HIP.hip_tools as hip 
import HIP.roctx_tools as roctx
c_double_p = ctypes.POINTER(c_double)


# Initialize HIP functions
hip.init()

# Initialize ROCTX functions
roctx.init()
roctx.sync_device = True

# Initialize HIP device
device_id = 0
hip.set_device( device_id )

N = 256 * 256 * 256
n_iter = 10
a = 1.5

# Set host arrays
rt_init_cpu = roctx.roctx_start('init_cpu')
h_x = np.random.rand(N).astype(np.float64)
h_y = np.random.rand(N).astype(np.float64)
cpu_result = a * h_x + h_y
roctx.roctx_stop(rt_init_cpu)

# Set device arrays
rt_init_gpu = roctx.roctx_start('init_gpu')
size = N * h_x.itemsize
d_x = hip.allocate_device( size )
d_y = hip.allocate_device( size )
gpu_result = np.empty(N, dtype=np.float64)
roctx.roctx_stop(rt_init_gpu)

# Copy input arrays from host to device
rt_copy_in = roctx.roctx_start('copy_input')
hip.copy_host_to_device( d_x, h_x.ctypes.data_as(c_void_p), size )
hip.copy_host_to_device( d_y, h_y.ctypes.data_as(c_void_p), size )
roctx.roctx_stop(rt_copy_in)

# Execute daxpy
rt_daxpy = roctx.roctx_start('gpu_daxpy')
hip.daxpy( c_int32(N), c_double(a), c_void_p(d_x), c_void_p(d_y) )
roctx.roctx_stop(rt_daxpy)

# Copy output back to host
rt_copy_out = roctx.roctx_start('copy_output')
hip.copy_device_to_host(  gpu_result.ctypes.data_as(c_void_p), d_y, size )
roctx.roctx_stop(rt_copy_out)

# Validate result
rt_validate = roctx.roctx_start('validate')
diff =  gpu_result - cpu_result 
diff[cpu_result>0] /= cpu_result[cpu_result>0]
diff = np.abs( diff )
print( f'diff. min: {diff.min()}  max: {diff.max()}')
if diff.max() >1e10: print('Validation Failed') 
else: print('Validation Passed')
roctx.roctx_stop(rt_validate)

# Clean
rt_clean = roctx.roctx_start('clean')
hip.free_device( d_x )
hip.free_device( d_y )
roctx.roctx_stop(rt_clean)

print('Finished!')





