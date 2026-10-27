// SPDX-License-Identifier: AGPL-3.0-only
#include <metal_stdlib>
using namespace metal;
inline float qh(device const uchar* p){return float(as_type<half>(ushort(uint(p[0])|(uint(p[1])<<8))));}
inline float qw(device const uchar* p,int type,int j){
 if(type==12){
  int k=j/32,l=j%32;device const uchar* s=p+4;
  int sc=k<4?s[k]&63:(s[k+4]&15)|((s[k-4]>>6)<<4);
  int mn=k<4?s[k+4]&63:(s[k+4]>>4)|((s[k]>>6)<<4);
  return (qh(p)*sc)*((p[16+(k/2)*32+l]>>((k%2)*4))&15)-qh(p+2)*mn;
 }
 int h=j/128,k=(j%128)/32,l=j%32;
 int low=(p[h*64+(k%2)*32+l]>>((k/2)*4))&15;
 int high=(p[128+h*32+l]>>(2*k))&3;
 int scale=int(as_type<char>(p[192+h*8+k*2+l/16]));
 return (qh(p+208)*scale)*((low|(high<<4))-32);
}
kernel void q4_gemv(device const uchar* weights [[buffer(0)]],device const float* x [[buffer(1)]],
 device float* y [[buffer(2)]],constant int3& shape [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],
 uint3 group [[threadgroup_position_in_grid]]){
 int R=shape.x,C=shape.y,type=shape.z,row=int(group.x)*4+int(tid)/32,lane=int(tid)%32;
 if(row>=R)return;
 int G=C/256,B=type==12?144:210;float sum=0;
 for(int g=0;g<G;g++){
  device const uchar* p=weights+(((ulong(row/8)*G+g)*8+row%8)*B);
  for(int j=lane;j<256;j+=32)sum+=qw(p,type,j)*x[g*256+j];
 }
 sum=simd_sum(sum);if(lane==0)y[row]=sum;
}
