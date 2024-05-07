
#include <roctx.h>
#include <roctracer_ext.h>


extern "C" {

void start_roctracer(){
  roctracer_start();
}

void stop_roctracer(){
  roctracer_stop();
}

int roctxr_start( char *c){
  int id = roctxRangeStart(c);
  return id;
}

void roctxr_stop( int id){
  roctxRangeStop(id);
}

void roctxr_push( char *c){
  roctxRangePush(c);
}

void roctxr_pop(){
  roctxRangePop();
}


}