/* spec.cu - the GDN steps of a T-token pass (decode T = 1, speculative verify T <= 8) with the state write-back
   switchable, and the commit of the accepted tokens. Same per-token arithmetic as k_gdn_pre_T / k_gdn_deltan_T
   (chunk.cu), which keep k_gdn_pre / k_gdn_deltan's. A verify pass (wb = 0) leaves the recurrent state (M, conv)
   as it was; once the accepted count n is known, the commit kernels advance the state by the first n tokens from
   the inputs the verify pass kept (per layer), so the state is exactly that of n one-token steps. */

extern "C" __global__ void k_sgdn_pre_T(const float* __restrict__ xn, int ldxn, const unsigned short* __restrict__ bw,
    const unsigned short* __restrict__ aw, const float* __restrict__ dtb, const float* __restrict__ ssma,
    float* __restrict__ beta, float* __restrict__ g, int ldg, int n,
    const float* __restrict__ qkv, int ldqkv, float* __restrict__ conv, const float* __restrict__ cw,
    float* __restrict__ cs, int ldcs, float* __restrict__ qk, int ldqk, int C, int NV, int NK, int HK, int T, int wb){
  const int b = blockIdx.x, t = threadIdx.x;
  if(b < NV){
    const int tok = blockIdx.y;
    if(tok >= T) return;
    __shared__ float sb[256];
    __shared__ float sa[256];
    const float* x = xn + (long long)tok * ldxn;
    float ab = 0.f, aa = 0.f;
    for(int i = t; i < n; i += 256){
      const float xv = x[i];
      ab += __uint_as_float((unsigned int)bw[(long long)b * n + i] << 16) * xv;
      aa += __uint_as_float((unsigned int)aw[(long long)b * n + i] << 16) * xv;
    }
    sb[t] = ab; sa[t] = aa;
    __syncthreads();
    for(int st = 128; st > 0; st >>= 1){
      if(t < st){ sb[t] += sb[t + st]; sa[t] += sa[t + st]; }
      __syncthreads();
    }
    if(t == 0){
      beta[(long long)tok * ldg + b] = 1.0f / (1.0f + expf(-sb[0]));
      const float a = sa[0] + dtb[b];
      const float sp = (a > 20.0f) ? a : log1pf(expf(fminf(a, 20.0f)));
      g[(long long)tok * ldg + b] = sp * ssma[b];
    }
    return;
  }
  if(blockIdx.y) return;
  const bool qkh = b < NV + 2 * NK;
  int c;
  bool act;
  if(qkh){ act = t < HK; c = (b - NV) * HK + t; }
  else { c = 2 * NK * HK + (b - NV - 2 * NK) * 256 + t; act = c < C; if(!act) return; }
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, w0 = 0.f, w1 = 0.f, w2 = 0.f, w3 = 0.f;
  if(act){
    c0 = conv[c]; c1 = conv[C + c]; c2 = conv[2 * C + c];
    w0 = cw[4 * c]; w1 = cw[4 * c + 1]; w2 = cw[4 * c + 2]; w3 = cw[4 * c + 3];
  }
  __shared__ float row[128];
  __shared__ float scl;
  for(int tok = 0; tok < T; ++tok){
    float y = 0.f;
    if(act){
      const float nw = qkv[(long long)tok * ldqkv + c];
      const float o = c0 * w0 + c1 * w1 + c2 * w2 + nw * w3;
      c0 = c1; c1 = c2; c2 = nw;
      y = o / (1.0f + expf(-o));
      cs[(long long)tok * ldcs + c] = y;
    }
    if(!qkh) continue;
    __syncthreads();
    if(act) row[t] = y;
    __syncthreads();
    if(t < 32){
      double acc = 0.0;
      for(int e = t; e < HK; e += 32){ const double v = (double)row[e]; acc += v * v; }
      acc = wsum_d(acc);
      if(t == 0) scl = 1.0f / fmaxf(sqrtf((float)acc), 1e-6f);
    }
    __syncthreads();
    if(act) qk[(long long)tok * ldqk + (b - NV) * HK + t] = row[t] * scl;
  }
  if(act && wb){ conv[c] = c0; conv[C + c] = c1; conv[2 * C + c] = c2; }
}

extern "C" __global__ void k_sconv_C(float* __restrict__ conv, const float* __restrict__ qkv, int ldqkv, int C,
                                     const int* __restrict__ Tp){
  /* commit: the conv shift register after the first *Tp tokens (grid ceil(C / 256), block 256) */
  const int c = blockIdx.x * 256 + threadIdx.x;
  if(c >= C) return;
  const int T = *Tp;
  float c0 = conv[c], c1 = conv[C + c], c2 = conv[2 * C + c];
  for(int tok = 0; tok < T; ++tok){ c0 = c1; c1 = c2; c2 = qkv[(long long)tok * ldqkv + c]; }
  conv[c] = c0; conv[C + c] = c1; conv[2 * C + c] = c2;
}

__device__ __forceinline__ void sgdn_deltan(float* __restrict__ M, const float* __restrict__ q,
    const float* __restrict__ k, int ldqk, const float* __restrict__ v, int ldv, const float* __restrict__ g,
    const float* __restrict__ beta, int ldg, const float* __restrict__ nw, float* __restrict__ out, int ldo,
    float scale, int n_k_heads, int T, int wb){
  const int h = blockIdx.x, j = threadIdx.x;
  __shared__ float ks[128];
  __shared__ float qs[128];
  __shared__ float os[128];
  __shared__ float scl;
  const int kh = h % n_k_heads;
  float* Mh = M + (long long)h * 128 * 128;
  float m[128];
#pragma unroll
  for(int i = 0; i < 128; ++i) m[i] = Mh[i * 128 + j];
  for(int tok = 0; tok < T; ++tok){
    __syncthreads();
    ks[j] = k[(long long)tok * ldqk + kh * 128 + j];
    qs[j] = q[(long long)tok * ldqk + kh * 128 + j];
    __syncthreads();
    const float e = expf(g[(long long)tok * ldg + h]);
    float sk = 0.f;
#pragma unroll
    for(int i = 0; i < 128; ++i) sk += (m[i] * e) * ks[i];
    const float dd = (v[(long long)tok * ldv + h * 128 + j] - sk) * beta[(long long)tok * ldg + h];
    float o = 0.f;
#pragma unroll
    for(int i = 0; i < 128; ++i){
      const float mm = m[i] * e + ks[i] * dd;
      m[i] = mm;
      o += mm * qs[i];
    }
    os[j] = o * scale;
    __syncthreads();
    if(j < 32){
      double acc = 0.0;
      for(int x = j; x < 128; x += 32){ const double vv = (double)os[x]; acc += vv * vv; }
      acc = wsum_d(acc);
      if(j == 0) scl = 1.0f / sqrtf((float)(acc / 128) + 1e-6f);
    }
    __syncthreads();
    out[(long long)tok * ldo + h * 128 + j] = (os[j] * scl) * nw[j];
  }
  if(wb){
#pragma unroll
    for(int i = 0; i < 128; ++i) Mh[i * 128 + j] = m[i];
  }
}

extern "C" __global__ void __launch_bounds__(128) k_sgdn_deltan_T(float* __restrict__ M, const float* __restrict__ q,
    const float* __restrict__ k, int ldqk, const float* __restrict__ v, int ldv, const float* __restrict__ g,
    const float* __restrict__ beta, int ldg, const float* __restrict__ nw, float* __restrict__ out, int ldo,
    float scale, int n_k_heads, int T, int wb){
  sgdn_deltan(M, q, k, ldqk, v, ldv, g, beta, ldg, nw, out, ldo, scale, n_k_heads, T, wb);
}

extern "C" __global__ void __launch_bounds__(128) k_sgdn_deltan_C(float* __restrict__ M, const float* __restrict__ q,
    const float* __restrict__ k, int ldqk, const float* __restrict__ v, int ldv, const float* __restrict__ g,
    const float* __restrict__ beta, int ldg, const float* __restrict__ nw, float* __restrict__ out, int ldo,
    float scale, int n_k_heads, const int* __restrict__ Tp){
  /* commit: the state after the first *Tp tokens - the same code as the pass (its outputs go to scratch), so the
     state rounds exactly as in one-token steps */
  sgdn_deltan(M, q, k, ldqk, v, ldv, g, beta, ldg, nw, out, ldo, scale, n_k_heads, *Tp, 1);
}

/* ------------------------------------------------------------------ verify attention: several queries per K read */
/* k_attn_sq8m: k_attn_sq8t for QN <= 4 consecutive queries of one pass (query k sees n0 + k positions) - each K row
   is read and decoded once for all of them; per query the arithmetic is k_attn_sq8t's, in the same order, so the
   scores are bit-identical. Query k's scores go to s + k * ss (rows of n0 + k). grid (NHKV, ceil((n0 + QN - 1) / 256)). */
extern "C" __global__ void __launch_bounds__(256) k_attn_sq8m(const uint4* __restrict__ Kq,
    const unsigned short* __restrict__ Kd, const signed char* __restrict__ qq, const float* __restrict__ qd, int qs,
    float* __restrict__ s, long long ss, int n0, int QN, int n_kv, int G){
  const int nmax = n0 + QN - 1;
  if((int)blockIdx.y * 256 >= nmax) return;
  __shared__ float4 q4[4 * 8 * 64];
  __shared__ float dq[4 * 8 * 8];
  const int kv = blockIdx.x, tid = threadIdx.x, t = blockIdx.y * 256 + tid;
  float* qf = (float*)q4;
  for(int k = 0; k < QN; ++k){
    for(int i = tid; i < G * 256; i += 256) qf[k * 2048 + i] = (float)qq[(long long)k * qs + kv * G * 256 + i];
    for(int i = tid; i < G * 8; i += 256) dq[k * 64 + i] = qd[(long long)k * (qs / 32) + kv * G * 8 + i];
  }
  __syncthreads();
  if(t >= nmax) return;
  const uint4* row = Kq + ((long long)t * n_kv + kv) * 16;
  const unsigned short* drow = Kd + ((long long)t * n_kv + kv) * 8;
  float acc[4][8];
#pragma unroll
  for(int k = 0; k < 4; ++k)
#pragma unroll
    for(int j = 0; j < 8; ++j) acc[k][j] = 0.f;
#pragma unroll 2
  for(int b = 0; b < 8; ++b){
    float part[4][8];
#pragma unroll
    for(int k = 0; k < 4; ++k)
#pragma unroll
      for(int j = 0; j < 8; ++j) part[k][j] = 0.f;
    const float dk = h2f(KLDG(&drow[b]));
#pragma unroll
    for(int hl = 0; hl < 2; ++hl){
      const uint4 w = KLDG(&row[b * 2 + hl]);
      const unsigned int ww[4] = {w.x, w.y, w.z, w.w};
      float kf[16];
#pragma unroll
      for(int i = 0; i < 4; ++i)
#pragma unroll
        for(int k = 0; k < 4; ++k) kf[i * 4 + k] = sb2f(ww[i] >> (8 * k));
#pragma unroll
      for(int k = 0; k < 4; ++k){
        if(k < QN){
#pragma unroll
          for(int j = 0; j < 8; ++j){
            if(j < G){
              float p = part[k][j];
#pragma unroll
              for(int e = 0; e < 4; ++e){
                const float4 q = q4[k * 512 + j * 64 + b * 8 + hl * 4 + e];
                p += kf[e * 4] * q.x; p += kf[e * 4 + 1] * q.y; p += kf[e * 4 + 2] * q.z; p += kf[e * 4 + 3] * q.w;
              }
              part[k][j] = p;
            }
          }
        }
      }
    }
#pragma unroll
    for(int k = 0; k < 4; ++k)
#pragma unroll
      for(int j = 0; j < 8; ++j) if(k < QN && j < G) acc[k][j] += part[k][j] * (dq[k * 64 + j * 8 + b] * dk);
  }
#pragma unroll
  for(int k = 0; k < 4; ++k){
    const int n = n0 + k;
    if(k < QN && t < n){
#pragma unroll
      for(int j = 0; j < 8; ++j) if(j < G) s[k * ss + (long long)(kv * G + j) * n + t] = acc[k][j] * 0.0625f;
    }
  }
}
