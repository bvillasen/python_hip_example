import sys, os
import HIP.hip_tools as hip

roctx_profile = True
sync_device = False

# Roctracer functions
start_roctracer = None
stop_roctracer = None
roctxr_start = None
roctxr_stop = None

def init():
    from ctypes import cdll, c_int, c_char_p
    global start_roctracer, stop_roctracer
    global roctxr_start, roctxr_stop
    
    hip_library_name = f'./HIP/libHIPcode.so'
    # print(f'Loading HIP library: {hip_library_name}')
    
    libhip = cdll.LoadLibrary( hip_library_name )
    
    start_roctracer = libhip.start_roctracer
    stop_roctracer = libhip.stop_roctracer
    
    roctxr_push = libhip.roctxr_push
    roctxr_push.argtypes = [c_char_p]
    
    roctxr_start = libhip.roctxr_start
    roctxr_start.argtypes = [ c_char_p ]
    roctxr_start.resypes = c_int 
    
    roctxr_stop  = libhip.roctxr_stop
    roctxr_stop.argtypes = [ c_int ]


encode = lambda s : s.encode('utf-8')


def roctx_start( name ):
  if not roctx_profile: return
  id = roctxr_start( encode(name) )
  return id
  
torch_device = None
def roctx_stop( id, sync_device=False ):
  if not roctx_profile: return
  if sync_device: hip.sync_device()
  roctxr_stop( id )