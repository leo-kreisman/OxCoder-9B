// SPDX-License-Identifier: AGPL-3.0-only
// Shared mmap tensor pages are wrapped directly; no second weight copy.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> pipeline;
static NSMutableDictionary<NSValue*,id<MTLBuffer>>* weights;
static id<MTLBuffer> input,output;
static NSString* lastError;
extern "C" const char* kq_metal_error(){return lastError?lastError.UTF8String:"Unknown Metal error";}
extern "C" int kq_metal_init(const char* source){
 @autoreleasepool {
  device=MTLCreateSystemDefaultDevice();if(!device){lastError=@"No Metal device";return -1;}
  NSError* error=nil;MTLCompileOptions* options=[MTLCompileOptions new];options.fastMathEnabled=NO;
  id<MTLLibrary> library=[device newLibraryWithSource:[NSString stringWithUTF8String:source] options:options error:&error];
  if(!library){lastError=error.localizedDescription;return -1;}
  pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"q4_gemv"] error:&error];
  if(!pipeline||pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128){lastError=@"Unsupported Metal pipeline/thread width";return -1;}
  queue=[device newCommandQueue];weights=[NSMutableDictionary new];
  input=[device newBufferWithLength:12288*4 options:MTLResourceStorageModeShared];
  output=[device newBufferWithLength:248320*4 options:MTLResourceStorageModeShared];
  if(!queue||!input||!output){lastError=@"Metal allocation failed";return -1;}return 0;
 }
}
extern "C" int kq_metal_matvec(const void* w,uint64_t bytes,int R,int C,int type,const float* x,float* y){
 @autoreleasepool {
  if(!pipeline||R>248320||C>12288||R<1||C<1){lastError=@"Invalid Metal matrix";return -1;}
  NSValue* key=[NSValue valueWithPointer:w];id<MTLBuffer> buffer=weights[key];
  if(!buffer){
   if(device.currentAllocatedSize+bytes>7400000000ULL){lastError=@"Metal admission budget exceeded";return -1;}
   buffer=[device newBufferWithBytesNoCopy:(void*)w length:bytes options:MTLResourceStorageModeShared deallocator:nil];
   if(!buffer){lastError=@"Cannot wrap mapped tensor; check page alignment and maxBufferLength";return -1;}
   weights[key]=buffer;
  }
  memcpy(input.contents,x,(size_t)C*4);
  id<MTLCommandBuffer> command=[queue commandBuffer];id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
  int dims[4]={R,C,type,0};
  [encoder setComputePipelineState:pipeline];[encoder setBuffer:buffer offset:0 atIndex:0];
  [encoder setBuffer:input offset:0 atIndex:1];[encoder setBuffer:output offset:0 atIndex:2];
  [encoder setBytes:dims length:sizeof(dims) atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake((R+3)/4,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
  [encoder endEncoding];[command commit];[command waitUntilCompleted];
  if(command.status!=MTLCommandBufferStatusCompleted){lastError=command.error.localizedDescription;return -1;}
  memcpy(y,output.contents,(size_t)R*4);return 0;
 }
}
extern "C" uint64_t kq_metal_allocated(){return device?device.currentAllocatedSize:0;}
