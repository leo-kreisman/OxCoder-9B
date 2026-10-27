/* Q4_K/Q6_K adaptation of Kindi's Qwen3.5 CPU forward pass.
 * Original CPU engine Copyright (c) 2026 Amir Husain.
 * SPDX-License-Identifier: AGPL-3.0-only
 * No Hadamard folds: these are the original non-ternary Qwen weight blocks.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#ifndef _OPENMP
static int omp_get_num_threads(void){return 1;}
static int omp_get_thread_num(void){return 0;}
#include <time.h>
static double omp_get_wtime(void){struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#endif

typedef struct { const uint8_t* W; const float* S; int R, C, type; uint64_t offset, bytes; } Mat;   /* W [R/8][C/256][8][quant-block-byte]; S unused for Q4/Q6. */

typedef struct {
  int is_attn;
  const float *attn_norm, *post_norm;
  /* GDN */
  Mat qkv, gate, ssm_out;
  const float *beta_w, *alpha_w, *dt_b, *ssm_a, *conv_w, *ssm_norm;
  float *M, *conv;                       /* state [NV][128][128], conv [3][CONV_CH] */
  /* attention */
  Mat q, k, v, o;
  const float *q_norm, *k_norm;
  uint16_t *Kc, *Vc;                     /* [ctx][NHKV][HD] fp16 */
  /* FFN */
  Mat ffg, ffu, ffd;
} Layer;

typedef struct {
  int D, NL, NFF, NH, NHKV, HD, NK, HK, NV, CONV_CH, HB, rope_half, ctx, vocab;
  Layer* L;
  Mat emb, head;
  const float *out_norm, *sD, *sV, *sA, *sF, *cos_t, *sin_t;
  /* scratch */
  float *h, *xn, *xp, *y, *qkv, *zv, *cs, *qk, *gout, *outn, *m, *qn, *kr, *vv, *qr, *kro, *sc, *att, *ffg, *ffu,
        *T, *logits, *gb, *gg;
  double* red;                           /* per-thread partial sums */
} Model;

/* ------------------------------------------------------------------------------------------- helpers */
/* stage timers (thread 0, between synchronised points): 0 embed 1 norm+fold 2 gemv 3 gdn-other 4 attn-other 5 ffn-other 6 head */
double kc_times[8];
static double t_last; static int t_cur;
#define TS(cat) do { if(omp_get_thread_num() == 0){ const double _t = omp_get_wtime(); kc_times[t_cur] += _t - t_last; t_last = _t; t_cur = (cat); } } while(0)
static inline void span(long long n, int* a, int* b){
  const int nt = omp_get_num_threads(), id = omp_get_thread_num();
  *a = (int)(n * id / nt); *b = (int)(n * (id + 1) / nt);
}
static inline float h2f(uint16_t h){
#if defined(__x86_64__)
return _cvtsh_ss(h);
#else
_Float16 v; memcpy(&v,&h,2); return (float)v;
#endif
}
static inline uint16_t f2h(float f){
#if defined(__x86_64__)
return _cvtss_sh(f, 0);
#else
_Float16 v=(_Float16)f; uint16_t h;memcpy(&h,&v,2);return h;
#endif
}
static inline float silu(float x){ return x / (1.0f + expf(-x)); }
static inline float sigm(float x){ return 1.0f / (1.0f + expf(-x)); }

static void fwht_block(float* x, int hb){
  for(int h = 1; h < hb; h <<= 1)
    for(int i = 0; i < hb; i += 2 * h)
      for(int j = i; j < i + h; ++j){ const float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
  const float s = 1.0f / sqrtf((float)hb);
  for(int j = 0; j < hb; ++j) x[j] *= s;
}

/* out = fold(v) = FWHT_blocks(sg * v); in parallel over blocks (must be called by all threads; ends in a barrier) */
static void fold(const Model* m, const float* v, const float* sg, float* out, int n){
  (void)m; (void)sg;
  #pragma omp for schedule(static)
  for(int i=0;i<n;i++) out[i]=v[i]; /* Original Qwen has no Bonsai Hadamard transform. */
}

#include "q4kernels.h"
static int kc_failed=0;
#ifdef KINDI_METAL
extern int kq_metal_matvec(const void*,uint64_t,int,int,int,const float*,float*);
#endif
static void gemv(Model* m, int n, const Mat* const* A, float* const* y, const float* x){
  (void)m;
#ifdef KINDI_METAL
  for(int i=0;i<n;i++){
    if(kq_metal_matvec(A[i]->W,A[i]->bytes,A[i]->R,A[i]->C,A[i]->type,x,y[i])){
      kc_failed=1;memset(y[i],0,A[i]->R*sizeof(float));
    }
  }
  return;
#endif
  int total=0;for(int i=0;i<n;i++)total+=A[i]->R;
  int lo,hi;span(total,&lo,&hi);int base=0;
  for(int i=0;i<n;i++){
    int a=lo>base?lo-base:0,b=hi-base<A[i]->R?hi-base:A[i]->R;
    for(int r=a;r<b;r++)y[i][r]=q4_dot(A[i],r,x);
    base+=A[i]->R;
  }
  #pragma omp barrier
}
static void gemv1(Model* m,const Mat* A,float* y,const float* x){const Mat* aa[]={A};float* yy[]={y};gemv(m,1,aa,yy,x);}

/* ------------------------------------------------------------------------------------------- norms */
/* residual add (if add) + RMS norm + fold of m->h into xn / xp; all threads; ends in a barrier */
static void rmsfold(Model* m, const float* w, int add){
  const int D = m->D;
  const int prev = t_cur; TS(1);
  #pragma omp single
  {
    double acc = 0.0;
    for(int e = 0; e < D; ++e){
      const float xf = add ? m->h[e] + m->y[e] : m->h[e];
      m->h[e] = xf;
      acc += (double)xf * (double)xf;
    }
    const float s = 1.0f / sqrtf((float)(acc / D) + 1e-6f);
    for(int e = 0; e < D; ++e) m->xn[e] = (m->h[e] * s) * w[e];
  }
  fold(m, m->xn, m->sD, m->xp, D);
  TS(prev);
}

static void rms_rows(const float* x, const float* w, float* out, int rows, int n){
  for(int r = 0; r < rows; ++r){
    const float* xr = x + (size_t)r * n;
    double acc = 0.0;
    for(int e = 0; e < n; ++e) acc += (double)xr[e] * (double)xr[e];
    const float sc = 1.0f / sqrtf((float)(acc / n) + 1e-6f);
    for(int e = 0; e < n; ++e) out[(size_t)r * n + e] = (xr[e] * sc) * w[e];
  }
}

/* ------------------------------------------------------------------------------------------- blocks */
static void gdn(Model* m, Layer* L, int add){
  TS(3);
  const int D = m->D, NV = m->NV, NK = m->NK, HK = m->HK, CC = m->CONV_CH;
  rmsfold(m, L->attn_norm, add);
  { const Mat* a[2] = {&L->qkv, &L->gate}; float* yy[2] = {m->qkv, m->zv}; gemv(m, 2, a, yy, m->xp); }
  /* gates (per v head) and the causal conv (per channel) */
  #pragma omp for schedule(static) nowait
  for(int h = 0; h < NV; ++h){
    float ab = 0.f, aa = 0.f;
    for(int i = 0; i < D; ++i){ ab += L->beta_w[(size_t)h * D + i] * m->xn[i]; aa += L->alpha_w[(size_t)h * D + i] * m->xn[i]; }
    m->gb[h] = sigm(ab);
    const float a = aa + L->dt_b[h];
    const float sp = (a > 20.0f) ? a : log1pf(expf(fminf(a, 20.0f)));
    m->gg[h] = sp * L->ssm_a[h];
  }
  #pragma omp for schedule(static)
  for(int c = 0; c < CC; ++c){
    float* cv = L->conv;
    const float c0 = cv[c], c1 = cv[CC + c], c2 = cv[2 * CC + c], nw = m->qkv[c];
    const float o = c0 * L->conv_w[4 * c] + c1 * L->conv_w[4 * c + 1] + c2 * L->conv_w[4 * c + 2] + nw * L->conv_w[4 * c + 3];
    cv[c] = c1; cv[CC + c] = c2; cv[2 * CC + c] = nw;
    m->cs[c] = silu(o);
  }
  /* L2 norm of the q and k heads */
  #pragma omp for schedule(static)
  for(int r = 0; r < 2 * NK; ++r){
    const float* xr = m->cs + (size_t)r * HK;
    double acc = 0.0;
    for(int e = 0; e < HK; ++e) acc += (double)xr[e] * (double)xr[e];
    const float sc = 1.0f / fmaxf(sqrtf((float)acc), 1e-6f);
    for(int e = 0; e < HK; ++e) m->qk[(size_t)r * HK + e] = xr[e] * sc;
  }
  /* delta rule per v head, then the per-head RMS norm */
  const float scale = 1.0f / sqrtf((float)HK);
  #pragma omp for schedule(static)
  for(int h = 0; h < NV; ++h){
    const int kh = h % NK;
    const float* ks = m->qk + (size_t)(NK + kh) * HK;
    const float* qs = m->qk + (size_t)kh * HK;
    const float* v = m->cs + 2 * NK * HK + (size_t)h * HK;
    const float e = expf(m->gg[h]);
    float* Mh = L->M + (size_t)h * HK * HK;
    float sk[128], o[128], dd[128];
    for(int j = 0; j < HK; ++j) sk[j] = 0.f;
    for(int i = 0; i < HK; ++i){ const float ki = ks[i]; const float* mr = Mh + i * HK; for(int j = 0; j < HK; ++j) sk[j] += (mr[j] * e) * ki; }
    for(int j = 0; j < HK; ++j){ dd[j] = (v[j] - sk[j]) * m->gb[h]; o[j] = 0.f; }
    for(int i = 0; i < HK; ++i){
      const float ki = ks[i], qi = qs[i];
      float* mr = Mh + i * HK;
      for(int j = 0; j < HK; ++j){ const float mm = mr[j] * e + ki * dd[j]; mr[j] = mm; o[j] += mm * qi; }
    }
    for(int j = 0; j < HK; ++j) m->gout[(size_t)h * HK + j] = o[j] * scale;
    rms_rows(m->gout + (size_t)h * HK, L->ssm_norm, m->outn + (size_t)h * HK, 1, HK);
  }
  /* gated output in the grouped v-head order, then the fold */
  const int rep = NV / NK;
  #pragma omp for schedule(static)
  for(int e = 0; e < NV * HK; ++e){
    int src = e;
    /* Original Qwen GGUF projection rows already use native v-head order. */
    m->m[e] = silu(m->zv[src]) * m->outn[src];
  }
  fold(m, m->m, m->sV, m->xp, NV * HK);
  gemv1(m, &L->ssm_out, m->y, m->xp);
}

static void attn(Model* m, Layer* L, int pos, int add){
  TS(4);
  const int NH = m->NH, NKV = m->NHKV, HD = m->HD, rh = m->rope_half, G = NH / NKV, n = pos + 1;
  rmsfold(m, L->attn_norm, add);
  { const Mat* a[3] = {&L->q, &L->k, &L->v}; float* yy[3] = {m->qn, m->kr, m->vv}; gemv(m, 3, a, yy, m->xp); }
  const float* cs = m->cos_t + (size_t)pos * rh;
  const float* sn = m->sin_t + (size_t)pos * rh;
  #pragma omp for schedule(static)
  for(int r = 0; r < NH + NKV; ++r){
    float tmp[512];
    float* dst;
    if(r < NH){ rms_rows(m->qn + (size_t)r * 2 * HD, L->q_norm, tmp, 1, HD); dst = m->qr + (size_t)r * HD; }
    else { rms_rows(m->kr + (size_t)(r - NH) * HD, L->k_norm, tmp, 1, HD); dst = m->kro + (size_t)(r - NH) * HD; }
    for(int j = 0; j < rh; ++j){
      const float a = tmp[j], b = tmp[j + rh];
      dst[j] = a * cs[j] - b * sn[j];
      dst[j + rh] = a * sn[j] + b * cs[j];
    }
    for(int j = 2 * rh; j < HD; ++j) dst[j] = tmp[j];
    if(r >= NH){
      const int kv = r - NH;
      uint16_t* K = L->Kc + ((size_t)pos * NKV + kv) * HD;
      uint16_t* V = L->Vc + ((size_t)pos * NKV + kv) * HD;
      for(int j = 0; j < HD; ++j){ K[j] = f2h(dst[j]); V[j] = f2h(m->vv[(size_t)kv * HD + j]); }
    }
  }
  /* scores sc[head][t] */
  #pragma omp for schedule(static)
  for(int t = 0; t < n; ++t){
    for(int kv = 0; kv < NKV; ++kv){
      const uint16_t* K = L->Kc + ((size_t)t * NKV + kv) * HD;
      float kf[512];
      for(int j = 0; j < HD; ++j) kf[j] = h2f(K[j]);
      for(int g = 0; g < G; ++g){
        const float* q = m->qr + (size_t)(kv * G + g) * HD;
        float a = 0.f;
        for(int j = 0; j < HD; ++j) a += kf[j] * q[j];
        m->sc[(size_t)(kv * G + g) * m->ctx + t] = a * 0.0625f;
      }
    }
  }
  /* softmax per head */
  #pragma omp for schedule(static)
  for(int hh = 0; hh < NH; ++hh){
    float* s = m->sc + (size_t)hh * m->ctx;
    float mx = -1e30f;
    for(int t = 0; t < n; ++t) mx = fmaxf(mx, s[t]);
    float tot = 0.f;
    for(int t = 0; t < n; ++t) tot += expf(s[t] - mx);
    for(int t = 0; t < n; ++t) s[t] = expf(s[t] - mx) / tot;
  }
  /* weighted V, gated; parallel over (head, 32-dim slice) */
  #pragma omp for schedule(static)
  for(int u = 0; u < NH * (HD / 32); ++u){
    const int hh = u / (HD / 32), d0 = (u % (HD / 32)) * 32, kv = hh / G;
    const float* p = m->sc + (size_t)hh * m->ctx;
    float acc[32];
    for(int j = 0; j < 32; ++j) acc[j] = 0.f;
    for(int t = 0; t < n; ++t){
      const uint16_t* V = L->Vc + ((size_t)t * NKV + kv) * HD + d0;
      const float pt = p[t];
      for(int j = 0; j < 32; ++j) acc[j] += pt * h2f(V[j]);
    }
    for(int j = 0; j < 32; ++j) m->att[(size_t)hh * HD + d0 + j] = acc[j] * sigm(m->qn[(size_t)hh * 2 * HD + HD + d0 + j]);
  }
  fold(m, m->att, m->sA, m->xp, NH * HD);
  gemv1(m, &L->o, m->y, m->xp);
}

static void ffn(Model* m, Layer* L){
  TS(5);
  rmsfold(m, L->post_norm, 1);
  { const Mat* a[2] = {&L->ffg, &L->ffu}; float* yy[2] = {m->ffg, m->ffu}; gemv(m, 2, a, yy, m->xp); }
  #pragma omp for schedule(static)
  for(int e = 0; e < m->NFF; ++e) m->m[e] = silu(m->ffg[e]) * m->ffu[e];
  fold(m, m->m, m->sF, m->xp, m->NFF);
  gemv1(m, &L->ffd, m->y, m->xp);
}

static void embed(Model* m,int tok){
  #pragma omp single
  { q4_row(&m->emb,tok,m->h); }
}

/* one token; head != 0: also the output head (logits in m->logits); returns the greedy token or -1 */
int kc_step(Model* m, int tok, int pos, int head){
  int best = -1; kc_failed=0;
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  {
    embed(m, tok);
    for(int il = 0; il < m->NL; ++il){
      Layer* L = m->L + il;
      if(L->is_attn) attn(m, L, pos, il > 0); else gdn(m, L, il > 0);
      ffn(m, L);
      #pragma omp single
      { q4_drop_layer(L); }
    }
    if(head){
      TS(6);
      rmsfold(m, m->out_norm, 1);
      gemv1(m, &m->head, m->logits, m->xp);
      #pragma omp single
      { q4_drop(&m->head); q4_drop(&m->emb); }
      #pragma omp single
      {
        int bi = 0; float bv = m->logits[0];
        for(int i = 1; i < m->head.R; ++i) if(m->logits[i] > bv){ bv = m->logits[i]; bi = i; }
        best = bi;
      }
    } else {
      #pragma omp single
      { for(int e = 0; e < m->D; ++e) m->h[e] += m->y[e]; }   /* keep h = the residual after the last block */
    }
  }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return kc_failed ? -2 : best;
}

void kc_matvec(const Mat* a,const float* x,float* y){
 #pragma omp parallel for schedule(static)
 for(int r=0;r<a->R;r++)y[r]=q4_dot(a,r,x);
}
void kc_dequant(const Mat* a,int row,float* y){q4_row(a,row,y);}
void kc_init(void){}
int kc_sizeof_model(void){ return (int)sizeof(Model); }
int kc_sizeof_layer(void){ return (int)sizeof(Layer); }
