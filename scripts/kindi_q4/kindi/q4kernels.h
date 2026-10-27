/* SPDX-License-Identifier: AGPL-3.0-only
 * Q4_K/Q6_K block interpretation follows the documented GGML format.
 * Lossless 8-row block interleave; no activation or weight requantization.
 */
static int q4_fd=-1, q4_stream=0;
void kc_storage(int fd,int stream){q4_fd=fd;q4_stream=stream;}
static int q4_block_bytes(int type){return type==12?144:210;}
static const uint8_t* q4_block(const Mat* a,int row,int group){
  return a->W+(((size_t)(row/8)*(a->C/256)+group)*8+(row%8))*q4_block_bytes(a->type);
}
static uint16_t q4_u16(const uint8_t* p){uint16_t v;memcpy(&v,p,2);return v;}
static void q4_decode(const uint8_t* b,int type,float* y){
  if(type==12){
    const float d=h2f(q4_u16(b)), dm=h2f(q4_u16(b+2));
    const uint8_t* s=b+4;
    for(int k=0;k<8;k++){
      const int sc=k<4?s[k]&63:(s[k+4]&15)|((s[k-4]>>6)<<4);
      const int mn=k<4?s[k+4]&63:(s[k+4]>>4)|((s[k]>>6)<<4);
      const float ds=d*sc, ms=dm*mn;
      const uint8_t* q=b+16+(k/2)*32;
      for(int j=0;j<32;j++)y[k*32+j]=ds*((q[j]>>((k%2)*4))&15)-ms;
    }
  }else{
    const float d=h2f(q4_u16(b+208));
    const int8_t* s=(const int8_t*)(b+192);
    for(int h=0;h<2;h++)for(int k=0;k<4;k++)for(int j=0;j<32;j++){
      const int low=(b[h*64+(k%2)*32+j]>>((k/2)*4))&15;
      const int high=(b[128+h*32+j]>>(2*k))&3;
      const float ds=d*s[h*8+k*2+j/16];
      y[h*128+k*32+j]=ds*((low|(high<<4))-32);
    }
  }
}
static void q4_row(const Mat* a,int row,float* y){
  for(int g=0;g<a->C/256;g++)q4_decode(q4_block(a,row,g),a->type,y+g*256);
}
static float q4_dot(const Mat* a,int row,const float* x){
#if defined(__AVX2__)
  __m256 acc=_mm256_setzero_ps();
  const __m256i mask15=_mm256_set1_epi32(15),mask3=_mm256_set1_epi32(3);
  for(int g=0;g<a->C/256;g++){
    const uint8_t* b=q4_block(a,row,g); const float* in=x+256*g;
    if(a->type==12){
      const float d=h2f(q4_u16(b)),dm=h2f(q4_u16(b+2));const uint8_t* s=b+4;
      for(int k=0;k<8;k++){
        int sc=k<4?s[k]&63:(s[k+4]&15)|((s[k-4]>>6)<<4);
        int mn=k<4?s[k+4]&63:(s[k+4]>>4)|((s[k]>>6)<<4);
        __m256 ds=_mm256_set1_ps(d*sc),ms=_mm256_set1_ps(dm*mn);
        const uint8_t* q=b+16+(k/2)*32;
        for(int j=0;j<32;j+=8){
          __m256i v=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(q+j)));
          if(k&1)v=_mm256_srli_epi32(v,4);
          v=_mm256_and_si256(v,mask15);
          __m256 w=_mm256_sub_ps(_mm256_mul_ps(ds,_mm256_cvtepi32_ps(v)),ms);
          acc=_mm256_add_ps(acc,_mm256_mul_ps(w,_mm256_loadu_ps(in+k*32+j)));
        }
      }
    }else{
      const float d=h2f(q4_u16(b+208));const int8_t* s=(const int8_t*)(b+192);
      for(int h=0;h<2;h++)for(int k=0;k<4;k++)for(int j=0;j<32;j+=8){
        __m256i low=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(b+h*64+(k%2)*32+j)));
        if(k>=2)low=_mm256_srli_epi32(low,4);low=_mm256_and_si256(low,mask15);
        __m256i high=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(b+128+h*32+j)));
        high=_mm256_and_si256(_mm256_srlv_epi32(high,_mm256_set1_epi32(2*k)),mask3);
        __m256i q=_mm256_sub_epi32(_mm256_or_si256(low,_mm256_slli_epi32(high,4)),_mm256_set1_epi32(32));
        __m256 w=_mm256_mul_ps(_mm256_set1_ps(d*s[h*8+k*2+j/16]),_mm256_cvtepi32_ps(q));
        acc=_mm256_add_ps(acc,_mm256_mul_ps(w,_mm256_loadu_ps(in+h*128+k*32+j)));
      }
    }
  }
  float lane[8];_mm256_storeu_ps(lane,acc);float total=0;for(int i=0;i<8;i++)total+=lane[i];return total;
#else
  float sum=0;
  for(int g=0;g<a->C/256;g++){
    float w[256]; q4_decode(q4_block(a,row,g),a->type,w);
    float part=0;
    #pragma omp simd reduction(+:part)
    for(int j=0;j<256;j++)part+=w[j]*x[g*256+j];
    sum+=part;
  }
  return sum;
#endif
}
static void q4_drop(const Mat* a){
  if(!q4_stream || !a->W || !a->bytes)return;
  /* Each tensor starts and ends on its own page. Only this model pack is affected. */
  madvise((void*)a->W,a->bytes,MADV_DONTNEED);
#if defined(__linux__)
  posix_fadvise(q4_fd,(off_t)a->offset,(off_t)a->bytes,POSIX_FADV_DONTNEED);
#endif
}
static void q4_drop_layer(const Layer* l){
  q4_drop(&l->qkv);q4_drop(&l->gate);q4_drop(&l->ssm_out);
  q4_drop(&l->q);q4_drop(&l->k);q4_drop(&l->v);q4_drop(&l->o);
  q4_drop(&l->ffg);q4_drop(&l->ffu);q4_drop(&l->ffd);
}
