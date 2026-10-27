// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_fp16.h>
__device__ float hf(const unsigned char* p){return __half2float(*reinterpret_cast<const __half*>(p));}
__device__ float qw(const unsigned char* p,int type,int j){
 if(type==12){
  int k=j/32,sub=j%32;const unsigned char* s=p+4;
  int sc=k<4?s[k]&63:(s[k+4]&15)|((s[k-4]>>6)<<4);
  int mn=k<4?s[k+4]&63:(s[k+4]>>4)|((s[k]>>6)<<4);
  int v=(p[16+(k/2)*32+sub]>>((k%2)*4))&15;
  return (hf(p)*sc)*v-hf(p+2)*mn;
 }
 int h=j/128,k=(j%128)/32,l=j%32;
 int low=(p[h*64+(k%2)*32+l]>>((k/2)*4))&15;
 int high=(p[128+h*32+l]>>(2*k))&3;
 int scale=((const signed char*)p)[192+h*8+k*2+l/16];
 return (hf(p+208)*scale)*((low|(high<<4))-32);
}
extern "C" __global__ void q4_gemv(const unsigned char* w,const float* x,float* y,int R,int C,int type){
 int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(row>=R)return;
 int G=C/256,B=type==12?144:210;float sum=0;
 for(int g=0;g<G;g++){
  const unsigned char* p=w+(((long long)(row/8)*G+g)*8+row%8)*B;
  for(int j=lane;j<256;j+=32)sum+=qw(p,type,j)*x[g*256+j];
 }
 for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);
 if(lane==0)y[row]=sum;
}
extern "C" __global__ void q4_embed(const unsigned char* w,float* y,int row,int C,int type){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=C)return;
 int G=C/256,B=type==12?144:210;
 y[i]=qw(w+(((long long)(row/8)*G+i/256)*8+row%8)*B,type,i%256);
}
extern "C" __global__ void q4_copy(const float* x,const float* unused,float* y,int width){
 int i=blockIdx.x*blockDim.x+threadIdx.x;y[i]=x[i];
}
extern "C" __global__ void q4_silu(const float* a,const float* b,const float* unused,float* y,int hd,int nk,int rep,int width){
 int i=blockIdx.x*blockDim.x+threadIdx.x;float x=a[i];y[i]=(x/(1.f+expf(-x)))*b[i];
}
extern "C" __global__ void q4_rms(float* h,const float* residual,const float* w,float* xn,float* xp,int D,int add){
 __shared__ double sums[256];int t=threadIdx.x;double sum=0;
 for(int i=t;i<D;i+=256){float v=h[i]+(add?residual[i]:0.f);h[i]=v;sum+=(double)v*v;}
 sums[t]=sum;__syncthreads();
 for(int n=128;n;n>>=1){if(t<n)sums[t]+=sums[t+n];__syncthreads();}
 float s=1.f/sqrtf((float)(sums[0]/D)+1e-6f);
 for(int i=t;i<D;i+=256){float v=h[i]*s*w[i];xn[i]=v;xp[i]=v;}
}
