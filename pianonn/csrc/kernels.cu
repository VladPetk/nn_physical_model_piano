// Fused kernels for the two costliest parts of a render (docs/speed.md):
//
// * the string bank: the damped-sinusoid oscillators of pianonn.oscbank, forward and analytic backward, over all
//   active notes and the whole window in one launch each. Nothing of size [notes, oscillators, samples] is ever
//   stored: the forward keeps only the output, the backward recomputes E sin(phi) and E cos(phi). The phase is
//   formed in float64 once per oscillator and 128-sample tile, then in float32 within the tile.
// * the sympathetic bank's resonators: a complex one-pole with a time-varying decay, run sequentially per resonator
//   (forward), and its adjoint recurrence backwards in time (backward).
//
// Same mathematics as the PyTorch reference (pianonn.oscbank.OscBank, pianonn.dsp.linear_recurrence); the tests in
// tests/test_cuda.py compare the two.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>

#define RESTRIKE_RAMP 0.002f
#define TWO_PI_F 6.283185307179586f
#define PI_F 3.141592653589793f

constexpr int T = 128;  // samples per tile (forward, backward A); oscillators per block (backward B)
constexpr int NW = T / 32;
constexpr int MAXS = 8;  // frames one tile's samples can reach, per curve (else direct atomics)

struct Bank {
  // oscillators [P, Q] (set 0 at [0, Q0), set 1 at [Q0, Q))
  const float* freq; const float* alpha; const float* amp; const float* adamp;
  const int* g0; const float* w0; const int* g1; const float* w1;  // group curves (G > 0)
  // notes [P]
  const double* onset; const float* tc; const float* c_onset; const float* rs; const float* rsd; int R;
  const float* C; const long long* row; int F; int hist; int hop;  // damper integral rows [rows, F] at hop
  const float* m; int G; int CM; int chop;  // log gains [P, G, CM] at the control hop
  const int* nv; int NCH; int chunk;  // active oscillators [P, NCH, 2] per activity chunk and set
  int Q; int Q0; int start; int L; double sr;
  int ncs; int nms;  // frames a tile's samples reach: damper integral, control curves
};

struct State {
  float tau, x, ramp, Draw, D, S, cw, mw;
  int ci0, mi0;
};

__device__ __forceinline__ State sample_state(const Bank& b, int p, int l) {
  State st;
  long long s_abs = (long long)b.start + l;
  double tau64 = (double)s_abs / b.sr - b.onset[p];
  st.tau = (float)(tau64 > 0.0 ? tau64 : 0.0);
  float tc = b.tc[p];
  st.x = st.tau / tc;
  st.ramp = 0.5f - 0.5f * cospif(fminf(st.x, 1.f));
  double pos = (double)(s_abs + b.hist) / b.hop;
  int i0 = (int)floor(pos);
  i0 = min(max(i0, 0), b.F - 2);
  st.ci0 = i0;
  st.cw = fminf(fmaxf((float)(pos - i0), 0.f), 1.f);
  const float* Cr = b.C + (size_t)b.row[p] * b.F;
  float c = Cr[i0] * (1.f - st.cw) + Cr[i0 + 1] * st.cw;
  st.Draw = c - b.c_onset[p];
  st.D = fmaxf(st.Draw, 0.f);
  float S = 0.f;
  for (int r = 0; r < b.R; r++) S += fminf(fmaxf((st.tau - b.rsd[p * b.R + r]) / RESTRIKE_RAMP, 0.f), 1.f);
  st.S = S;
  if (b.G > 0) {
    double pm = (double)s_abs / b.chop;
    int j0 = (int)floor(pm);
    j0 = min(max(j0, 0), b.CM - 2);
    st.mi0 = j0;
    st.mw = fminf(fmaxf((float)(pm - j0), 0.f), 1.f);
  } else {
    st.mi0 = 0; st.mw = 0.f;
  }
  return st;
}

__device__ __forceinline__ float group_gain_log(const Bank& b, int p, int k, const State& st) {
  const float* mr = b.m + ((size_t)p * b.G + k) * b.CM;
  return mr[st.mi0] * (1.f - st.mw) + mr[st.mi0 + 1] * st.mw;
}

__device__ __forceinline__ void phase_sincos(float base, float f, float dt, float* s, float* c) {
  float cyc = fmaf(f, dt, base);
  cyc -= floorf(cyc);
  float sn, cs;
  __sincosf(fmaf(TWO_PI_F, cyc, -PI_F), &sn, &cs);  // argument in [-pi, pi): the intrinsic's accurate range
  *s = -sn;
  *c = -cs;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
  return v;
}

// shared-memory layout of the per-oscillator parameters of one note
struct OscShared {
  float *f, *al, *a, *ad, *base, *w0, *w1;
  int *g0, *g1;
};

__device__ __forceinline__ OscShared osc_shared(float* sh, int Q) {
  OscShared o;
  o.f = sh; o.al = o.f + Q; o.a = o.al + Q; o.ad = o.a + Q; o.base = o.ad + Q; o.w0 = o.base + Q; o.w1 = o.w0 + Q;
  o.g0 = (int*)(o.w1 + Q); o.g1 = o.g0 + Q;
  return o;
}

__device__ __forceinline__ void stage_oscs(const Bank& b, const OscShared& o, int p, int nv0, int nv1, double rel0) {
  for (int j = threadIdx.x; j < nv0 + nv1; j += blockDim.x) {
    int q = j < nv0 ? j : b.Q0 + (j - nv0);
    size_t i = (size_t)p * b.Q + q;
    float f = b.freq[i];
    o.f[q] = f; o.al[q] = b.alpha[i]; o.a[q] = b.amp[i]; o.ad[q] = b.adamp[i];
    double cyc = (double)f * rel0;
    o.base[q] = (float)(cyc - floor(cyc));
    if (b.G > 0) { o.g0[q] = b.g0[i]; o.w0[q] = b.w0[i]; o.g1[q] = b.g1[i]; o.w1[q] = b.w1[i]; }
  }
}

// ------------------------------------------------------------------------------------------------- string forward
__global__ void osc_forward_kernel(Bank b, float* __restrict__ y) {
  extern __shared__ float sh[];
  const int p = blockIdx.y, l0 = blockIdx.x * T, tid = threadIdx.x, l = l0 + tid;
  const int c = l0 / b.chunk;
  const int nv0 = b.nv[((size_t)p * b.NCH + c) * 2], nv1 = b.nv[((size_t)p * b.NCH + c) * 2 + 1];
  if (nv0 == 0) return;
  const double on = b.onset[p];
  if ((double)(b.start + l0 + T - 1) / b.sr <= on) return;  // before the strike: the ramp is 0 throughout
  OscShared o = osc_shared(sh, b.Q);
  float* gain = (float*)(o.g1 + b.Q);  // [G][T + 1]
  stage_oscs(b, o, p, nv0, nv1, (double)(b.start + l0) / b.sr - on - 0.5 * (double)b.tc[p]);
  State st = sample_state(b, p, min(l, b.L - 1));
  for (int k = 0; k < b.G; k++) gain[k * (T + 1) + tid] = __expf(group_gain_log(b, p, k, st));
  __syncthreads();
  const float dt = (float)((double)tid / b.sr);
  const float eb = -b.rs[p] * st.S;
  float acc = 0.f;
  for (int set = 0; set < 2; set++) {
    const int qb = set ? b.Q0 : 0, qe = set ? b.Q0 + nv1 : nv0;
    for (int q = qb; q < qe; q++) {
      float sn, cs;
      phase_sincos(o.base[q], o.f[q], dt, &sn, &cs);
      float E = __expf(eb - (o.al[q] * st.tau + o.ad[q] * st.D));
      float g = 1.f;
      if (b.G > 0) g = o.w0[q] * gain[o.g0[q] * (T + 1) + tid] + o.w1[q] * gain[o.g1[q] * (T + 1) + tid];
      acc = fmaf(o.a[q] * g, E * sn, acc);
    }
  }
  if (l < b.L) y[(size_t)p * b.L + l] = st.ramp * acc;
}

// ------------------------------------------------- string backward A: per sample, reduced over the oscillators
struct GradsA {
  float* d_tc; float* d_conset; float* d_rs; float* dC; float* dm;
};

__global__ void osc_backward_a_kernel(Bank b, const float* __restrict__ gy, GradsA g) {
  extern __shared__ float sh[];
  const int p = blockIdx.y, l0 = blockIdx.x * T, tid = threadIdx.x, l = l0 + tid;
  const int c = l0 / b.chunk;
  const int nv0 = b.nv[((size_t)p * b.NCH + c) * 2], nv1 = b.nv[((size_t)p * b.NCH + c) * 2 + 1];
  if (nv0 == 0) return;
  const double on = b.onset[p];
  if ((double)(b.start + l0 + T - 1) / b.sr <= on) return;
  OscShared o = osc_shared(sh, b.Q);
  float* gain = (float*)(o.g1 + b.Q);       // [G][T + 1]
  float* Y = gain + b.G * (T + 1);          // [G][T + 1]: each group's sum of a E sin(phi) at this sample
  float* red = Y + b.G * (T + 1);           // [NW][3 + ncs + nms G]
  stage_oscs(b, o, p, nv0, nv1, (double)(b.start + l0) / b.sr - on - 0.5 * (double)b.tc[p]);
  const bool in = l < b.L;
  State st = sample_state(b, p, min(l, b.L - 1));
  for (int k = 0; k < b.G; k++) {
    gain[k * (T + 1) + tid] = __expf(group_gain_log(b, p, k, st));
    Y[k * (T + 1) + tid] = 0.f;
  }
  __syncthreads();
  const float dt = (float)((double)tid / b.sr);
  const float eb = -b.rs[p] * st.S;
  float ys = 0.f, yd = 0.f, yc = 0.f;
  int cg0 = -1, cg1 = -1;
  float ca0 = 0.f, ca1 = 0.f;
  for (int set = 0; set < 2; set++) {
    const int qb = set ? b.Q0 : 0, qe = set ? b.Q0 + nv1 : nv0;
    for (int q = qb; q < qe; q++) {
      float sn, cs;
      phase_sincos(o.base[q], o.f[q], dt, &sn, &cs);
      float E = __expf(eb - (o.al[q] * st.tau + o.ad[q] * st.D));
      float a = o.a[q];
      float aMs = a * E * sn;
      float gq = 1.f;
      if (b.G > 0) {
        int k0 = o.g0[q];
        gq = o.w0[q] * gain[k0 * (T + 1) + tid];
        if (k0 != cg0) { if (cg0 >= 0) Y[cg0 * (T + 1) + tid] += ca0; cg0 = k0; ca0 = 0.f; }
        ca0 = fmaf(o.w0[q], aMs, ca0);
        float w1 = o.w1[q];
        if (w1 != 0.f) {
          int k1 = o.g1[q];
          gq = fmaf(w1, gain[k1 * (T + 1) + tid], gq);
          if (k1 != cg1) { if (cg1 >= 0) Y[cg1 * (T + 1) + tid] += ca1; cg1 = k1; ca1 = 0.f; }
          ca1 = fmaf(w1, aMs, ca1);
        }
      }
      ys = fmaf(gq, aMs, ys);
      yd = fmaf(gq * o.ad[q], aMs, yd);
      yc = fmaf(gq * a * o.f[q], E * cs, yc);
    }
  }
  if (cg0 >= 0) Y[cg0 * (T + 1) + tid] += ca0;
  if (cg1 >= 0) Y[cg1 * (T + 1) + tid] += ca1;

  const float gyv = in ? gy[(size_t)p * b.L + l] : 0.f;
  const float Gs = gyv * st.ramp;
  const float tc = b.tc[p];
  const float dramp = st.x < 1.f ? -0.5f * PI_F * sinpif(fminf(st.x, 1.f)) * st.tau / (tc * tc) : 0.f;
  const bool slots_c = b.ncs <= MAXS, slots_m = b.nms <= MAXS;
  const int NCS = slots_c ? b.ncs : 0, NMS = slots_m ? b.nms : 0;
  const int NV = 3 + NCS + NMS * b.G;
  const int lane = tid & 31, warp = tid >> 5;
  const float dcn = st.Draw > 0.f ? -Gs * yd : 0.f;     // d c_note at this sample
  float v[3];
  v[0] = gyv * dramp * ys - PI_F * Gs * yc;            // d tc
  v[1] = -dcn;                                          // d c_onset
  v[2] = -Gs * st.S * ys;                               // d restrike nats
  for (int k = 0; k < 3; k++) {
    float s = warp_sum(v[k]);
    if (lane == 0) red[warp * NV + k] = s;
  }
  // the damper integral's frames: this sample's two, re the tile's first frame
  const int iA = min(max((int)(((long long)b.start + l0 + b.hist) / b.hop), 0), b.F - 2);
  if (slots_c) {
    const int sl = st.ci0 - iA;
    for (int k = 0; k < NCS; k++) {
      float u = (k == sl ? dcn * (1.f - st.cw) : 0.f) + (k == sl + 1 ? dcn * st.cw : 0.f);
      float s = warp_sum(u);
      if (lane == 0) red[warp * NV + 3 + k] = s;
    }
  } else if (dcn != 0.f) {
    float* row = g.dC + (size_t)b.row[p] * b.F;
    atomicAdd(row + st.ci0, dcn * (1.f - st.cw));
    atomicAdd(row + st.ci0 + 1, dcn * st.cw);
  }
  // the curves' control frames
  int iB = 0;
  if (b.G > 0) {
    iB = min(max((int)((long long)(b.start + l0) / b.chop), 0), b.CM - 2);
    const int sm = st.mi0 - iB;
    for (int k = 0; k < b.G; k++) {
      const float d = Gs * Y[k * (T + 1) + tid] * gain[k * (T + 1) + tid];  // d log gain
      if (slots_m) {
        for (int j = 0; j < NMS; j++) {
          float u = (j == sm ? d * (1.f - st.mw) : 0.f) + (j == sm + 1 ? d * st.mw : 0.f);
          float s = warp_sum(u);
          if (lane == 0) red[warp * NV + 3 + NCS + NMS * k + j] = s;
        }
      } else if (d != 0.f) {
        float* mr = g.dm + ((size_t)p * b.G + k) * b.CM;
        atomicAdd(mr + st.mi0, d * (1.f - st.mw));
        atomicAdd(mr + st.mi0 + 1, d * st.mw);
      }
    }
  }
  __syncthreads();
  for (int k = tid; k < NV; k += T) {
    float s = 0.f;
    for (int w = 0; w < NW; w++) s += red[w * NV + k];
    if (s == 0.f) continue;
    if (k == 0) atomicAdd(g.d_tc + p, s);
    else if (k == 1) atomicAdd(g.d_conset + p, s);
    else if (k == 2) atomicAdd(g.d_rs + p, s);
    else if (k < 3 + NCS) {
      int f = iA + (k - 3);
      if (f < b.F) atomicAdd(g.dC + (size_t)b.row[p] * b.F + f, s);
    } else {
      int kk = (k - 3 - NCS) / NMS, j = (k - 3 - NCS) % NMS;
      int f = iB + j;
      if (f < b.CM) atomicAdd(g.dm + ((size_t)p * b.G + kk) * b.CM + f, s);
    }
  }
}

// ---------------------------------------------- string backward B: per oscillator, reduced over the samples
struct GradsB {
  float* d_freq; float* d_alpha; float* d_amp; float* d_adamp;
};

__global__ void osc_backward_b_kernel(Bank b, const float* __restrict__ gy, GradsB g) {
  extern __shared__ float sh[];
  const int c = blockIdx.x, p = blockIdx.y, qt = blockIdx.z, tid = threadIdx.x;
  const int nv0 = b.nv[((size_t)p * b.NCH + c) * 2], nv1 = b.nv[((size_t)p * b.NCH + c) * 2 + 1];
  const int n_act = nv0 + nv1;
  if (nv0 == 0 || qt * T >= n_act) return;
  const int j = qt * T + tid;
  const bool active = j < n_act;
  const int q = j < nv0 ? j : b.Q0 + (j - nv0);
  const size_t oi = (size_t)p * b.Q + (active ? q : 0);
  const float f = b.freq[oi], al = b.alpha[oi], a = b.amp[oi], ad = b.adamp[oi];
  int k0 = 0, k1 = 0;
  float w0 = 1.f, w1 = 0.f;
  if (b.G > 0) { k0 = b.g0[oi]; k1 = b.g1[oi]; w0 = b.w0[oi]; w1 = b.w1[oi]; }
  float* s_tau = sh; float* s_D = s_tau + T; float* s_eb = s_D + T; float* s_G = s_eb + T; float* s_rel = s_G + T;
  float* s_dt = s_rel + T; float* gain = s_dt + T;  // [G][T + 1]
  s_dt[tid] = (float)((double)tid / b.sr);
  const double on = b.onset[p];
  const float tc = b.tc[p], rs = b.rs[p];
  const int lb = c * b.chunk, le = min(lb + b.chunk, b.L);
  float Sa = 0.f, St = 0.f, Sd = 0.f, Sf = 0.f;
  for (int sub = lb; sub < le; sub += T) {
    if ((double)(b.start + sub + T - 1) / b.sr <= on) continue;
    __syncthreads();
    const int l = sub + tid;
    State st = sample_state(b, p, min(l, b.L - 1));
    s_tau[tid] = st.tau; s_D[tid] = st.D; s_eb[tid] = -rs * st.S;
    s_G[tid] = l < le ? gy[(size_t)p * b.L + l] * st.ramp : 0.f;
    s_rel[tid] = st.tau - 0.5f * tc;
    for (int k = 0; k < b.G; k++) gain[k * (T + 1) + tid] = __expf(group_gain_log(b, p, k, st));
    __syncthreads();
    if (!active) continue;
    double cyc0 = (double)f * ((double)(b.start + sub) / b.sr - on - 0.5 * (double)tc);
    const float base = (float)(cyc0 - floor(cyc0));
    for (int i = 0; i < T; i++) {
      const float Gs = s_G[i];
      if (Gs == 0.f) continue;  // the same i for every thread: no divergence
      float sn, cs;
      phase_sincos(base, f, s_dt[i], &sn, &cs);
      const float tau = s_tau[i], D = s_D[i];
      const float E = __expf(s_eb[i] - (al * tau + ad * D));
      float gq = 1.f;
      if (b.G > 0) gq = w0 * gain[k0 * (T + 1) + i] + w1 * gain[k1 * (T + 1) + i];
      const float wg = Gs * gq * E;
      const float ws = wg * sn;
      Sa += ws;
      St = fmaf(tau, ws, St);
      Sd = fmaf(D, ws, Sd);
      Sf = fmaf(s_rel[i], wg * cs, Sf);
    }
  }
  if (active) {
    atomicAdd(g.d_amp + oi, Sa);
    atomicAdd(g.d_alpha + oi, -a * St);
    atomicAdd(g.d_adamp + oi, -a * Sd);
    atomicAdd(g.d_freq + oi, TWO_PI_F * a * Sf);
  }
}

// ------------------------------------------------------------------------------------------------- host side
static Bank make_bank(torch::Tensor freq, torch::Tensor alpha, torch::Tensor amp, torch::Tensor adamp,
                      torch::Tensor g0, torch::Tensor w0, torch::Tensor g1, torch::Tensor w1,
                      torch::Tensor onset, torch::Tensor tc, torch::Tensor c_onset, torch::Tensor rs,
                      torch::Tensor rsd, torch::Tensor C, torch::Tensor row, torch::Tensor m, torch::Tensor nv,
                      int64_t hist, int64_t hop, int64_t chop, int64_t chunk, int64_t Q0, int64_t start, int64_t L,
                      double sr) {
  Bank b;
  b.freq = freq.data_ptr<float>(); b.alpha = alpha.data_ptr<float>(); b.amp = amp.data_ptr<float>();
  b.adamp = adamp.data_ptr<float>();
  b.G = m.numel() ? (int)m.size(1) : 0;
  b.CM = m.numel() ? (int)m.size(2) : 0;
  b.m = m.numel() ? m.data_ptr<float>() : nullptr;
  b.g0 = b.G ? g0.data_ptr<int>() : nullptr; b.w0 = b.G ? w0.data_ptr<float>() : nullptr;
  b.g1 = b.G ? g1.data_ptr<int>() : nullptr; b.w1 = b.G ? w1.data_ptr<float>() : nullptr;
  b.onset = onset.data_ptr<double>(); b.tc = tc.data_ptr<float>(); b.c_onset = c_onset.data_ptr<float>();
  b.rs = rs.data_ptr<float>(); b.rsd = rsd.data_ptr<float>(); b.R = (int)rsd.size(1);
  b.C = C.data_ptr<float>(); b.row = row.data_ptr<int64_t>(); b.F = (int)C.size(1);
  b.hist = (int)hist; b.hop = (int)hop; b.chop = (int)chop;
  b.nv = nv.data_ptr<int>(); b.NCH = (int)nv.size(1); b.chunk = (int)chunk;
  b.Q = (int)freq.size(1); b.Q0 = (int)Q0; b.start = (int)start; b.L = (int)L; b.sr = sr;
  b.ncs = (T - 1) / b.hop + 3;  // a tile's samples lie in <= (T - 1) / hop + 2 frames, each reaching the next too
  b.nms = b.chop > 0 ? (T - 1) / b.chop + 3 : 0;
  return b;
}

static void set_smem(const void* kernel, size_t bytes) {
  if (bytes > 48 * 1024)
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
}

#define BANK_ARGS torch::Tensor freq, torch::Tensor alpha, torch::Tensor amp, torch::Tensor adamp, torch::Tensor g0, \
  torch::Tensor w0, torch::Tensor g1, torch::Tensor w1, torch::Tensor onset, torch::Tensor tc, torch::Tensor c_onset, \
  torch::Tensor rs, torch::Tensor rsd, torch::Tensor C, torch::Tensor row, torch::Tensor m, torch::Tensor nv, \
  int64_t hist, int64_t hop, int64_t chop, int64_t chunk, int64_t Q0, int64_t start, int64_t L, double sr
#define BANK_PASS freq, alpha, amp, adamp, g0, w0, g1, w1, onset, tc, c_onset, rs, rsd, C, row, m, nv, hist, hop, chop, \
  chunk, Q0, start, L, sr

torch::Tensor osc_forward(BANK_ARGS) {
  Bank b = make_bank(BANK_PASS);
  const int P = (int)freq.size(0);
  auto y = torch::zeros({P, L}, freq.options());
  if (P == 0 || L == 0) return y;
  size_t smem = (size_t)b.Q * 9 * 4 + (size_t)b.G * (T + 1) * 4;
  set_smem((const void*)osc_forward_kernel, smem);
  dim3 grid((unsigned)((L + T - 1) / T), (unsigned)P);
  osc_forward_kernel<<<grid, T, smem, at::cuda::getCurrentCUDAStream()>>>(b, y.data_ptr<float>());
  return y;
}

std::vector<torch::Tensor> osc_backward(torch::Tensor gy, BANK_ARGS) {
  Bank b = make_bank(BANK_PASS);
  const int P = (int)freq.size(0);
  auto d_freq = torch::zeros_like(freq), d_alpha = torch::zeros_like(freq), d_amp = torch::zeros_like(freq),
       d_adamp = torch::zeros_like(freq);
  auto d_tc = torch::zeros_like(tc), d_conset = torch::zeros_like(c_onset), d_rs = torch::zeros_like(rs);
  auto dC = torch::zeros_like(C);
  auto dm = torch::zeros_like(m);
  if (P > 0 && L > 0) {
    auto stream = at::cuda::getCurrentCUDAStream();
    GradsA ga{d_tc.data_ptr<float>(), d_conset.data_ptr<float>(), d_rs.data_ptr<float>(), dC.data_ptr<float>(),
              b.G ? dm.data_ptr<float>() : nullptr};
    size_t smem_a = (size_t)b.Q * 9 * 4 + (size_t)b.G * (T + 1) * 4 * 2 + (size_t)NW * (3 + MAXS + MAXS * b.G) * 4;
    set_smem((const void*)osc_backward_a_kernel, smem_a);
    dim3 grid_a((unsigned)((L + T - 1) / T), (unsigned)P);
    osc_backward_a_kernel<<<grid_a, T, smem_a, stream>>>(b, gy.data_ptr<float>(), ga);
    GradsB gb{d_freq.data_ptr<float>(), d_alpha.data_ptr<float>(), d_amp.data_ptr<float>(), d_adamp.data_ptr<float>()};
    size_t smem_b = (size_t)T * 6 * 4 + (size_t)b.G * (T + 1) * 4;
    set_smem((const void*)osc_backward_b_kernel, smem_b);
    dim3 grid_b((unsigned)b.NCH, (unsigned)P, (unsigned)((b.Q + T - 1) / T));
    osc_backward_b_kernel<<<grid_b, T, smem_b, stream>>>(b, gy.data_ptr<float>(), gb);
  }
  return {d_freq, d_alpha, d_amp, d_adamp, d_tc, d_conset, d_rs, dC, dm};
}

// ------------------------------------------------------------------------------- sympathetic resonators
// drive [B, L, K], es [B, L, K] (damper engagement per sample), per resonator [B, K, S]: alpha, adamp, freq, gin,
// z0 (complex start state as re, im). Writes z [B, L, K, S] (re, im).
__global__ void reson_forward_kernel(const float* __restrict__ drive, const float* __restrict__ es,
                                     const float* __restrict__ alpha, const float* __restrict__ adamp,
                                     const float* __restrict__ freq, const float* __restrict__ gin,
                                     const float* __restrict__ z0r, const float* __restrict__ z0i,
                                     float* __restrict__ zr_out, float* __restrict__ zi_out,
                                     int B, int K, int S, int L, float sr, float max_decay) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= B * K * S) return;
  const int bb = r / (K * S), k = (r / S) % K, KS = K * S, ks = r % KS;
  const float al = alpha[r], ad = adamp[r], g = gin[r];
  float sw, cw;
  sincosf(TWO_PI_F / sr * freq[r], &sw, &cw);
  float zr = z0r[r], zi = z0i[r];
  const float* dr = drive + (size_t)bb * L * K + k;
  const float* e = es + (size_t)bb * L * K + k;
  float* outr = zr_out + (size_t)bb * L * KS + ks;
  float* outi = zi_out + (size_t)bb * L * KS + ks;
  for (int t = 0; t < L; t++) {
    const float dec = fminf(al + e[(size_t)t * K] * ad, max_decay);
    const float rr = expf(-dec / sr);
    const float nr = rr * (cw * zr - sw * zi) + dr[(size_t)t * K] * g;
    const float ni = rr * (sw * zr + cw * zi);
    zr = nr; zi = ni;
    outr[(size_t)t * KS] = zr;
    outi[(size_t)t * KS] = zi;
  }
}

// gy [B, L] (d out, out = sum of z.re over the resonators), gzr, gzi [B, K, S] (d final state). Writes per resonator
// d alpha, d adamp, d freq, d gin, d z0 and per resonator and sample d drive, d es (summed over S by the caller).
__global__ void reson_backward_kernel(const float* __restrict__ gy, const float* __restrict__ gzr,
                                      const float* __restrict__ gzi, const float* __restrict__ drive,
                                      const float* __restrict__ es, const float* __restrict__ alpha,
                                      const float* __restrict__ adamp, const float* __restrict__ freq,
                                      const float* __restrict__ gin, const float* __restrict__ z0r,
                                      const float* __restrict__ z0i, const float* __restrict__ zr_s,
                                      const float* __restrict__ zi_s, float* __restrict__ d_alpha,
                                      float* __restrict__ d_adamp, float* __restrict__ d_freq, float* __restrict__ d_gin,
                                      float* __restrict__ d_z0r, float* __restrict__ d_z0i,
                                      float* __restrict__ d_drive, float* __restrict__ d_es,
                                      int B, int K, int S, int L, float sr, float max_decay) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= B * K * S) return;
  const int bb = r / (K * S), k = (r / S) % K, KS = K * S, ks = r % KS;
  const float al = alpha[r], ad = adamp[r], g = gin[r];
  float sw, cw;
  sincosf(TWO_PI_F / sr * freq[r], &sw, &cw);
  const float* dr = drive + (size_t)bb * L * K + k;
  const float* e = es + (size_t)bb * L * K + k;
  const float* zr = zr_s + (size_t)bb * L * KS + ks;
  const float* zi = zi_s + (size_t)bb * L * KS + ks;
  float* ddr = d_drive + (size_t)bb * L * KS + ks;
  float* des = d_es + (size_t)bb * L * KS + ks;
  const float* gyb = gy + (size_t)bb * L;
  float mr = gzr[r], mi = gzi[r];  // d / d z_t, accumulated backwards
  float dal = 0.f, dad = 0.f, dom = 0.f, dg = 0.f;
  for (int t = L - 1; t >= 0; t--) {
    mr += gyb[t];
    // z_t = rr R z_{t-1} + x_t
    const float pr = t > 0 ? zr[(size_t)(t - 1) * KS] : z0r[r];
    const float pi = t > 0 ? zi[(size_t)(t - 1) * KS] : z0i[r];
    const float ev = e[(size_t)t * K];
    const float dec0 = al + ev * ad;
    const float dec = fminf(dec0, max_decay);
    const float rr = expf(-dec / sr);
    const float Rr = cw * pr - sw * pi, Ri = sw * pr + cw * pi;  // R z_{t-1}
    const float d_rr = mr * Rr + mi * Ri;
    const float d_dec = dec0 < max_decay ? -d_rr * rr / sr : 0.f;
    dal += d_dec;
    dad += ev * d_dec;
    des[(size_t)t * KS] = ad * d_dec;
    // d omega: R' z = (-sw pr - cw pi, cw pr - sw pi)
    dom += rr * (mr * (-sw * pr - cw * pi) + mi * (cw * pr - sw * pi));
    const float dx = mr;
    dg += dx * dr[(size_t)t * K];
    ddr[(size_t)t * KS] = dx * g;
    // to z_{t-1}: (rr R)^T mu
    const float nr = rr * (cw * mr + sw * mi);
    const float ni = rr * (-sw * mr + cw * mi);
    mr = nr; mi = ni;
  }
  d_alpha[r] = dal; d_adamp[r] = dad; d_freq[r] = dom * TWO_PI_F / sr; d_gin[r] = dg;
  d_z0r[r] = mr; d_z0i[r] = mi;
}

std::vector<torch::Tensor> reson_forward(torch::Tensor drive, torch::Tensor es, torch::Tensor alpha, torch::Tensor adamp,
                                         torch::Tensor freq, torch::Tensor gin, torch::Tensor z0r, torch::Tensor z0i,
                                         double sr, double max_decay) {
  const int B = (int)alpha.size(0), K = (int)alpha.size(1), S = (int)alpha.size(2), L = (int)drive.size(1);
  auto zr = torch::empty({B, L, K, S}, drive.options()), zi = torch::empty({B, L, K, S}, drive.options());
  const int n = B * K * S, th = 64;
  if (n > 0 && L > 0)
    reson_forward_kernel<<<(n + th - 1) / th, th, 0, at::cuda::getCurrentCUDAStream()>>>(
        drive.data_ptr<float>(), es.data_ptr<float>(), alpha.data_ptr<float>(), adamp.data_ptr<float>(),
        freq.data_ptr<float>(), gin.data_ptr<float>(), z0r.data_ptr<float>(), z0i.data_ptr<float>(),
        zr.data_ptr<float>(), zi.data_ptr<float>(), B, K, S, L, (float)sr, (float)max_decay);
  return {zr, zi};
}

std::vector<torch::Tensor> reson_backward(torch::Tensor gy, torch::Tensor gzr, torch::Tensor gzi, torch::Tensor drive,
                                          torch::Tensor es, torch::Tensor alpha, torch::Tensor adamp, torch::Tensor freq,
                                          torch::Tensor gin, torch::Tensor z0r, torch::Tensor z0i, torch::Tensor zr,
                                          torch::Tensor zi, double sr, double max_decay) {
  const int B = (int)alpha.size(0), K = (int)alpha.size(1), S = (int)alpha.size(2), L = (int)drive.size(1);
  auto d_alpha = torch::zeros_like(alpha), d_adamp = torch::zeros_like(alpha), d_freq = torch::zeros_like(alpha),
       d_gin = torch::zeros_like(alpha), d_z0r = torch::zeros_like(alpha), d_z0i = torch::zeros_like(alpha);
  auto d_drive = torch::zeros({B, L, K, S}, drive.options()), d_es = torch::zeros({B, L, K, S}, drive.options());
  const int n = B * K * S, th = 64;
  if (n > 0 && L > 0)
    reson_backward_kernel<<<(n + th - 1) / th, th, 0, at::cuda::getCurrentCUDAStream()>>>(
        gy.data_ptr<float>(), gzr.data_ptr<float>(), gzi.data_ptr<float>(), drive.data_ptr<float>(),
        es.data_ptr<float>(), alpha.data_ptr<float>(), adamp.data_ptr<float>(), freq.data_ptr<float>(),
        gin.data_ptr<float>(), z0r.data_ptr<float>(), z0i.data_ptr<float>(), zr.data_ptr<float>(),
        zi.data_ptr<float>(), d_alpha.data_ptr<float>(), d_adamp.data_ptr<float>(), d_freq.data_ptr<float>(),
        d_gin.data_ptr<float>(), d_z0r.data_ptr<float>(), d_z0i.data_ptr<float>(), d_drive.data_ptr<float>(),
        d_es.data_ptr<float>(), B, K, S, L, (float)sr, (float)max_decay);
  return {d_alpha, d_adamp, d_freq, d_gin, d_z0r, d_z0i, d_drive, d_es};
}
