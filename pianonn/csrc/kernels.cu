// Fused kernels for the two costliest parts of a render (docs/speed.md):
//
// * the string bank: the damped-sinusoid oscillators of pianonn.oscbank, forward and analytic backward, over all
//   active notes and the whole window in one launch each. Nothing of size [notes, oscillators, samples] is ever
//   stored: the forward keeps only the output, the backward recomputes E sin(phi) and E cos(phi). The phase is
//   formed in float64 once per oscillator and 128-sample tile, then in float32 within the tile.
// * the sympathetic bank's resonators: a complex one-pole with a time-varying decay, run sequentially per resonator
//   (forward), and its adjoint recurrence backwards in time (backward).
// * the coupled strings' bus bank: the string bank with several outputs per oscillator (each with a sine and a cosine
//   amplitude) and the pitch glide (pianonn.oscbank.bus_bank).
// * the coupled strings' longitudinal force per note: the square of the bridge slope through a DC blocker and the free
//   longitudinal modes (NeuralPhysicalPiano._longitudinal), sequential per note, nothing stored for the backward.
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

// ---------------------------------------------------------------------------------- the coupled strings' bus bank
// pianonn.oscbank.bus_bank (docs/physics_revamp.md): each oscillator feeds NB buses, each with its own sine and cosine
// amplitude (amp [P, Q, NB, 2]); the modes (set 0) glide, f (1 + e exp(-beta tau)), the knock's resonances (set 1) do
// not. Outputs: the first NM buses (the vertical and in-plane radiation per microphone) summed per example, mix [B, NM,
// L]; the rest (the two longitudinal sums) per note, lon [P, NB - NM, L]; optionally the mean of the first NM / 2
// buses (the vertical bridge force) per key row, keys [rows, L], for the sympathetic bank. A group index < 0: no group
// gain (the knock's resonances). The amplitudes are read from global memory (the same address for every thread of a
// block: a broadcast), not staged.

struct BusExtra {
  const float* amp;
  const float* ge; const float* gb;  // glide per note: e, beta
  const long long* bidx;             // example per note
  int NM; int keys;
};

__device__ __forceinline__ float glide_g(float beta, float tau) {
  return beta > 1e-6f ? -expm1f(-beta * tau) / beta : tau;  // (1 - exp(-beta tau)) / beta
}

__device__ __forceinline__ float bus_gain(const Bank& b, const OscShared& o, const float* gain, int q, int i) {
  if (b.G == 0) return 1.f;
  const int k0 = o.g0[q];
  if (k0 < 0) return 1.f;
  return o.w0[q] * gain[k0 * (T + 1) + i] + o.w1[q] * gain[o.g1[q] * (T + 1) + i];
}

template <int NB>
__global__ void bus_forward_kernel(Bank b, BusExtra x, float* __restrict__ mix, float* __restrict__ lon,
                                   float* __restrict__ keys) {
  extern __shared__ float sh[];
  const int p = blockIdx.y, l0 = blockIdx.x * T, tid = threadIdx.x, l = l0 + tid;
  const int c = l0 / b.chunk;
  const int nv0 = b.nv[((size_t)p * b.NCH + c) * 2], nv1 = b.nv[((size_t)p * b.NCH + c) * 2 + 1];
  if (nv0 == 0) return;
  const double on = b.onset[p];
  if ((double)(b.start + l0 + T - 1) / b.sr <= on) return;
  OscShared o = osc_shared(sh, b.Q);
  float* gain = (float*)(o.g1 + b.Q);  // [G][T + 1]
  stage_oscs(b, o, p, nv0, nv1, (double)(b.start + l0) / b.sr - on - 0.5 * (double)b.tc[p]);
  State st = sample_state(b, p, min(l, b.L - 1));
  for (int k = 0; k < b.G; k++) gain[k * (T + 1) + tid] = __expf(group_gain_log(b, p, k, st));
  __syncthreads();
  const float dt = (float)((double)tid / b.sr);
  const float eg = x.ge[p] * glide_g(x.gb[p], st.tau);
  const float eb = -b.rs[p] * st.S;
  float acc[NB];
#pragma unroll
  for (int j = 0; j < NB; j++) acc[j] = 0.f;
  for (int set = 0; set < 2; set++) {
    const int qb = set ? b.Q0 : 0, qe = set ? b.Q0 + nv1 : nv0;
    const float dts = set ? dt : dt + eg;
    for (int q = qb; q < qe; q++) {
      float sn, cs;
      phase_sincos(o.base[q], o.f[q], dts, &sn, &cs);
      const float ge = bus_gain(b, o, gain, q, tid) * __expf(eb - (o.al[q] * st.tau + o.ad[q] * st.D));
      const float s = ge * sn, cc = ge * cs;
      const float* __restrict__ a = x.amp + ((size_t)p * b.Q + q) * (2 * NB);
#pragma unroll
      for (int j = 0; j < NB; j++) acc[j] = fmaf(__ldg(a + 2 * j), s, fmaf(__ldg(a + 2 * j + 1), cc, acc[j]));
    }
  }
  if (l >= b.L) return;
  const float r = st.ramp;
  const size_t L = b.L;
  const long long e = x.bidx[p];
#pragma unroll
  for (int j = 0; j < NB; j++) {
    if (j < x.NM) atomicAdd(mix + ((size_t)e * x.NM + j) * L + l, r * acc[j]);
    else lon[((size_t)p * (NB - x.NM) + (j - x.NM)) * L + l] = r * acc[j];
  }
  if (x.keys) {
    float mono = 0.f;
    for (int j = 0; j < x.NM / 2; j++) mono += acc[j];
    atomicAdd(keys + (size_t)b.row[p] * L + l, r * mono / (x.NM / 2));
  }
}

// the gradient reaching each bus of note p at sample l (0 past the window's end)
template <int NB>
__device__ __forceinline__ void bus_grads(const Bank& b, const BusExtra& x, const float* gmix, const float* glon,
                                          const float* gkeys, int p, int l, float* G) {
  const size_t L = b.L;
  const bool in = l < b.L;
  const long long e = x.bidx[p];
  const float gk = (in && x.keys) ? gkeys[(size_t)b.row[p] * L + l] / (x.NM / 2) : 0.f;
#pragma unroll
  for (int j = 0; j < NB; j++) {
    if (!in) G[j] = 0.f;
    else if (j < x.NM) G[j] = gmix[((size_t)e * x.NM + j) * L + l] + (j < x.NM / 2 ? gk : 0.f);
    else G[j] = glon[((size_t)p * (NB - x.NM) + (j - x.NM)) * L + l];
  }
}

// backward A: per sample, reduced over the oscillators (the note's scalars, the damper rows, the curves, the glide)
template <int NB>
__global__ void bus_backward_a_kernel(Bank b, BusExtra x, const float* __restrict__ gmix, const float* __restrict__ glon,
                                      const float* __restrict__ gkeys, GradsA g, float* __restrict__ d_ge,
                                      float* __restrict__ d_gb) {
  extern __shared__ float sh[];
  const int p = blockIdx.y, l0 = blockIdx.x * T, tid = threadIdx.x, l = l0 + tid;
  const int c = l0 / b.chunk;
  const int nv0 = b.nv[((size_t)p * b.NCH + c) * 2], nv1 = b.nv[((size_t)p * b.NCH + c) * 2 + 1];
  if (nv0 == 0) return;
  const double on = b.onset[p];
  if ((double)(b.start + l0 + T - 1) / b.sr <= on) return;
  OscShared o = osc_shared(sh, b.Q);
  float* gain = (float*)(o.g1 + b.Q);       // [G][T + 1]
  float* Y = gain + b.G * (T + 1);          // [G][T + 1]: each group's U at this sample
  float* red = Y + b.G * (T + 1);           // [NW][5 + ncs + nms G]
  stage_oscs(b, o, p, nv0, nv1, (double)(b.start + l0) / b.sr - on - 0.5 * (double)b.tc[p]);
  State st = sample_state(b, p, min(l, b.L - 1));
  for (int k = 0; k < b.G; k++) {
    gain[k * (T + 1) + tid] = __expf(group_gain_log(b, p, k, st));
    Y[k * (T + 1) + tid] = 0.f;
  }
  __syncthreads();
  float Gy[NB];
  bus_grads<NB>(b, x, gmix, glon, gkeys, p, l, Gy);
  const float dt = (float)((double)tid / b.sr);
  const float e = x.ge[p], beta = x.gb[p];
  const float gf = glide_g(beta, st.tau);
  const float eb = -b.rs[p] * st.S;
  // U = sum_j Gy_j d y_j / d (gain E) (the output's weight), V = its derivative in the phase
  float ys = 0.f, yd = 0.f, yv = 0.f, yv0 = 0.f;
  int cg0 = -1, cg1 = -1;
  float ca0 = 0.f, ca1 = 0.f;
  for (int set = 0; set < 2; set++) {
    const int qb = set ? b.Q0 : 0, qe = set ? b.Q0 + nv1 : nv0;
    const float dts = set ? dt : fmaf(e, gf, dt);
    float yvs = 0.f;
    for (int q = qb; q < qe; q++) {
      float sn, cs;
      phase_sincos(o.base[q], o.f[q], dts, &sn, &cs);
      const float E = __expf(eb - (o.al[q] * st.tau + o.ad[q] * st.D));
      const float* __restrict__ a = x.amp + ((size_t)p * b.Q + q) * (2 * NB);
      float us = 0.f, vs = 0.f;
#pragma unroll
      for (int j = 0; j < NB; j++) {
        const float as = __ldg(a + 2 * j), ac = __ldg(a + 2 * j + 1);
        us = fmaf(Gy[j], fmaf(as, sn, ac * cs), us);
        vs = fmaf(Gy[j], fmaf(as, cs, -ac * sn), vs);
      }
      const float U = E * us, V = E * vs;
      float gq = 1.f;
      if (b.G > 0 && o.g0[q] >= 0) {
        const int k0 = o.g0[q];
        gq = o.w0[q] * gain[k0 * (T + 1) + tid];
        if (k0 != cg0) { if (cg0 >= 0) Y[cg0 * (T + 1) + tid] += ca0; cg0 = k0; ca0 = 0.f; }
        ca0 = fmaf(o.w0[q], U, ca0);
        const float w1 = o.w1[q];
        if (w1 != 0.f) {
          const int k1 = o.g1[q];
          gq = fmaf(w1, gain[k1 * (T + 1) + tid], gq);
          if (k1 != cg1) { if (cg1 >= 0) Y[cg1 * (T + 1) + tid] += ca1; cg1 = k1; ca1 = 0.f; }
          ca1 = fmaf(w1, U, ca1);
        }
      }
      ys = fmaf(gq, U, ys);
      yd = fmaf(gq * o.ad[q], U, yd);
      yvs = fmaf(gq * o.f[q], V, yvs);
    }
    yv += yvs;
    if (set == 0) yv0 = yvs;
  }
  if (cg0 >= 0) Y[cg0 * (T + 1) + tid] += ca0;
  if (cg1 >= 0) Y[cg1 * (T + 1) + tid] += ca1;

  const float r = st.ramp;
  const float tc = b.tc[p];
  const float dramp = st.x < 1.f ? -0.5f * PI_F * sinpif(fminf(st.x, 1.f)) * st.tau / (tc * tc) : 0.f;
  const bool slots_c = b.ncs <= MAXS, slots_m = b.nms <= MAXS;
  const int NCS = slots_c ? b.ncs : 0, NMS = slots_m ? b.nms : 0;
  const int NV = 5 + NCS + NMS * b.G;
  const int lane = tid & 31, warp = tid >> 5;
  const float dcn = st.Draw > 0.f ? -r * yd : 0.f;  // d c_note at this sample
  const float dg = beta > 1e-6f ? (st.tau * __expf(-beta * st.tau) - gf) / beta : -0.5f * st.tau * st.tau;
  float v[5];
  v[0] = dramp * ys - PI_F * r * yv;    // d tc
  v[1] = -dcn;                           // d c_onset
  v[2] = -r * st.S * ys;                 // d restrike nats
  v[3] = TWO_PI_F * r * yv0 * gf;        // d glide e
  v[4] = TWO_PI_F * r * yv0 * e * dg;    // d glide beta
  for (int k = 0; k < 5; k++) {
    float s = warp_sum(v[k]);
    if (lane == 0) red[warp * NV + k] = s;
  }
  const int iA = min(max((int)(((long long)b.start + l0 + b.hist) / b.hop), 0), b.F - 2);
  if (slots_c) {
    const int sl = st.ci0 - iA;
    for (int k = 0; k < NCS; k++) {
      float u = (k == sl ? dcn * (1.f - st.cw) : 0.f) + (k == sl + 1 ? dcn * st.cw : 0.f);
      float s = warp_sum(u);
      if (lane == 0) red[warp * NV + 5 + k] = s;
    }
  } else if (dcn != 0.f) {
    float* row = g.dC + (size_t)b.row[p] * b.F;
    atomicAdd(row + st.ci0, dcn * (1.f - st.cw));
    atomicAdd(row + st.ci0 + 1, dcn * st.cw);
  }
  int iB = 0;
  if (b.G > 0) {
    iB = min(max((int)((long long)(b.start + l0) / b.chop), 0), b.CM - 2);
    const int sm = st.mi0 - iB;
    for (int k = 0; k < b.G; k++) {
      const float d = r * Y[k * (T + 1) + tid] * gain[k * (T + 1) + tid];  // d log gain
      if (slots_m) {
        for (int j = 0; j < NMS; j++) {
          float u = (j == sm ? d * (1.f - st.mw) : 0.f) + (j == sm + 1 ? d * st.mw : 0.f);
          float s = warp_sum(u);
          if (lane == 0) red[warp * NV + 5 + NCS + NMS * k + j] = s;
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
    else if (k == 3) atomicAdd(d_ge + p, s);
    else if (k == 4) atomicAdd(d_gb + p, s);
    else if (k < 5 + NCS) {
      int f = iA + (k - 5);
      if (f < b.F) atomicAdd(g.dC + (size_t)b.row[p] * b.F + f, s);
    } else {
      int kk = (k - 5 - NCS) / NMS, j = (k - 5 - NCS) % NMS;
      int f = iB + j;
      if (f < b.CM) atomicAdd(g.dm + ((size_t)p * b.G + kk) * b.CM + f, s);
    }
  }
}

// backward B: per oscillator, reduced over one activity chunk's samples (amplitudes, decays, frequency)
template <int NB>
__global__ void bus_backward_b_kernel(Bank b, BusExtra x, const float* __restrict__ gmix, const float* __restrict__ glon,
                                      const float* __restrict__ gkeys, GradsB g) {
  extern __shared__ float sh[];
  const int c = blockIdx.x, p = blockIdx.y, qt = blockIdx.z, tid = threadIdx.x;
  const int nv0 = b.nv[((size_t)p * b.NCH + c) * 2], nv1 = b.nv[((size_t)p * b.NCH + c) * 2 + 1];
  const int n_act = nv0 + nv1;
  if (nv0 == 0 || qt * T >= n_act) return;
  const int jq = qt * T + tid;
  const bool active = jq < n_act;
  const int q = jq < nv0 ? jq : b.Q0 + (jq - nv0);
  const bool modes = q < b.Q0;
  const size_t oi = (size_t)p * b.Q + (active ? q : 0);
  const float f = b.freq[oi], al = b.alpha[oi], ad = b.adamp[oi];
  float A[2 * NB];
#pragma unroll
  for (int j = 0; j < 2 * NB; j++) A[j] = x.amp[oi * (2 * NB) + j];
  int k0 = -1, k1 = 0;
  float w0 = 1.f, w1 = 0.f;
  if (b.G > 0) { k0 = b.g0[oi]; k1 = b.g1[oi]; w0 = b.w0[oi]; w1 = b.w1[oi]; }
  float* s_tau = sh; float* s_D = s_tau + T; float* s_eb = s_D + T; float* s_rel = s_eb + T; float* s_dt = s_rel + T;
  float* s_eg = s_dt + T; float* s_any = s_eg + T; float* s_G = s_any + T;  // [NB][T]
  float* gain = s_G + NB * T;  // [G][T + 1]
  s_dt[tid] = (float)((double)tid / b.sr);
  const double on = b.onset[p];
  const float tc = b.tc[p], rs = b.rs[p], e = x.ge[p], beta = x.gb[p];
  const int lb = c * b.chunk, le = min(lb + b.chunk, b.L);
  float dA[2 * NB];
#pragma unroll
  for (int j = 0; j < 2 * NB; j++) dA[j] = 0.f;
  float St = 0.f, Sd = 0.f, Sf = 0.f;
  for (int sub = lb; sub < le; sub += T) {
    if ((double)(b.start + sub + T - 1) / b.sr <= on) continue;
    __syncthreads();
    const int l = sub + tid;
    State st = sample_state(b, p, min(l, b.L - 1));
    s_tau[tid] = st.tau; s_D[tid] = st.D; s_eb[tid] = -rs * st.S;
    s_rel[tid] = st.tau - 0.5f * tc;
    s_eg[tid] = e * glide_g(beta, st.tau);
    float Gy[NB];
    bus_grads<NB>(b, x, gmix, glon, gkeys, p, l < le ? l : b.L, Gy);
    bool any = false;
#pragma unroll
    for (int j = 0; j < NB; j++) { s_G[j * T + tid] = Gy[j] * st.ramp; any |= Gy[j] * st.ramp != 0.f; }
    s_any[tid] = any ? 1.f : 0.f;
    for (int k = 0; k < b.G; k++) gain[k * (T + 1) + tid] = __expf(group_gain_log(b, p, k, st));
    __syncthreads();
    if (!active) continue;
    double cyc0 = (double)f * ((double)(b.start + sub) / b.sr - on - 0.5 * (double)tc);
    const float base = (float)(cyc0 - floor(cyc0));
    for (int i = 0; i < T; i++) {
      if (s_any[i] == 0.f) continue;  // the same i for every thread: no divergence
      const float eg = modes ? s_eg[i] : 0.f;
      float sn, cs;
      phase_sincos(base, f, s_dt[i] + eg, &sn, &cs);
      float gq = 1.f;
      if (k0 >= 0) gq = w0 * gain[k0 * (T + 1) + i] + w1 * gain[k1 * (T + 1) + i];
      const float E = gq * __expf(s_eb[i] - (al * s_tau[i] + ad * s_D[i]));
      const float sE = E * sn, cE = E * cs;
      float U = 0.f, V = 0.f;
#pragma unroll
      for (int j = 0; j < NB; j++) {
        const float Gj = s_G[j * T + i];
        dA[2 * j] = fmaf(Gj, sE, dA[2 * j]);
        dA[2 * j + 1] = fmaf(Gj, cE, dA[2 * j + 1]);
        U = fmaf(Gj, fmaf(A[2 * j], sE, A[2 * j + 1] * cE), U);
        V = fmaf(Gj, fmaf(A[2 * j], cE, -A[2 * j + 1] * sE), V);
      }
      St = fmaf(s_tau[i], U, St);
      Sd = fmaf(s_D[i], U, Sd);
      Sf = fmaf(s_rel[i] + eg, V, Sf);
    }
  }
  if (active) {
#pragma unroll
    for (int j = 0; j < 2 * NB; j++) atomicAdd(g.d_amp + oi * (2 * NB) + j, dA[j]);
    atomicAdd(g.d_alpha + oi, -St);
    atomicAdd(g.d_adamp + oi, -Sd);
    atomicAdd(g.d_freq + oi, TWO_PI_F * Sf);
  }
}

static BusExtra make_extra(torch::Tensor amp, torch::Tensor ge, torch::Tensor gb, torch::Tensor bidx, int64_t NM,
                           bool keys) {
  BusExtra x;
  x.amp = amp.data_ptr<float>(); x.ge = ge.data_ptr<float>(); x.gb = gb.data_ptr<float>();
  x.bidx = bidx.data_ptr<int64_t>(); x.NM = (int)NM; x.keys = keys ? 1 : 0;
  return x;
}

#define BUS_DISPATCH(NBV, ...) \
  if ((NBV) == 6) { constexpr int NB = 6; __VA_ARGS__; } \
  else if ((NBV) == 4) { constexpr int NB = 4; __VA_ARGS__; } \
  else TORCH_CHECK(false, "bus bank: 4 or 6 buses (1 or 2 channels)");

std::vector<torch::Tensor> bus_forward(BANK_ARGS, torch::Tensor ge, torch::Tensor gb, torch::Tensor bidx, int64_t NM,
                                       int64_t n_examples, int64_t n_rows, bool keys) {
  Bank b = make_bank(BANK_PASS);
  BusExtra x = make_extra(amp, ge, gb, bidx, NM, keys);
  const int P = (int)freq.size(0), NBV = (int)amp.size(2);
  auto mix = torch::zeros({n_examples, NM, L}, freq.options());
  auto lon = torch::zeros({P, NBV - NM, L}, freq.options());
  auto kout = torch::zeros({keys ? n_rows : 0, L}, freq.options());
  if (P == 0 || L == 0) return {mix, lon, kout};
  size_t smem = (size_t)b.Q * 9 * 4 + (size_t)b.G * (T + 1) * 4;
  dim3 grid((unsigned)((L + T - 1) / T), (unsigned)P);
  auto stream = at::cuda::getCurrentCUDAStream();
  BUS_DISPATCH(NBV,
    set_smem((const void*)bus_forward_kernel<NB>, smem);
    bus_forward_kernel<NB><<<grid, T, smem, stream>>>(b, x, mix.data_ptr<float>(), lon.data_ptr<float>(),
                                                      keys ? kout.data_ptr<float>() : nullptr))
  return {mix, lon, kout};
}

std::vector<torch::Tensor> bus_backward(torch::Tensor gmix, torch::Tensor glon, torch::Tensor gkeys, BANK_ARGS,
                                        torch::Tensor ge, torch::Tensor gb, torch::Tensor bidx, int64_t NM, bool keys) {
  Bank b = make_bank(BANK_PASS);
  BusExtra x = make_extra(amp, ge, gb, bidx, NM, keys);
  const int P = (int)freq.size(0), NBV = (int)amp.size(2);
  auto d_freq = torch::zeros_like(freq), d_alpha = torch::zeros_like(freq), d_adamp = torch::zeros_like(freq);
  auto d_amp = torch::zeros_like(amp);
  auto d_tc = torch::zeros_like(tc), d_conset = torch::zeros_like(c_onset), d_rs = torch::zeros_like(rs);
  auto d_ge = torch::zeros_like(ge), d_gb = torch::zeros_like(gb);
  auto dC = torch::zeros_like(C);
  auto dm = torch::zeros_like(m);
  if (P > 0 && L > 0) {
    auto stream = at::cuda::getCurrentCUDAStream();
    GradsA ga{d_tc.data_ptr<float>(), d_conset.data_ptr<float>(), d_rs.data_ptr<float>(), dC.data_ptr<float>(),
              b.G ? dm.data_ptr<float>() : nullptr};
    GradsB gbr{d_freq.data_ptr<float>(), d_alpha.data_ptr<float>(), d_amp.data_ptr<float>(), d_adamp.data_ptr<float>()};
    const float* pk = keys ? gkeys.data_ptr<float>() : nullptr;
    size_t smem_a = (size_t)b.Q * 9 * 4 + (size_t)b.G * (T + 1) * 4 * 2 + (size_t)NW * (5 + MAXS + MAXS * b.G) * 4;
    size_t smem_b = (size_t)T * (7 + NBV) * 4 + (size_t)b.G * (T + 1) * 4;
    dim3 grid_a((unsigned)((L + T - 1) / T), (unsigned)P);
    dim3 grid_b((unsigned)b.NCH, (unsigned)P, (unsigned)((b.Q + T - 1) / T));
    BUS_DISPATCH(NBV,
      set_smem((const void*)bus_backward_a_kernel<NB>, smem_a);
      bus_backward_a_kernel<NB><<<grid_a, T, smem_a, stream>>>(b, x, gmix.data_ptr<float>(), glon.data_ptr<float>(), pk,
                                                               ga, d_ge.data_ptr<float>(), d_gb.data_ptr<float>());
      set_smem((const void*)bus_backward_b_kernel<NB>, smem_b);
      bus_backward_b_kernel<NB><<<grid_b, T, smem_b, stream>>>(b, x, gmix.data_ptr<float>(), glon.data_ptr<float>(), pk,
                                                               gbr))
  }
  return {d_freq, d_alpha, d_amp, d_adamp, d_tc, d_conset, d_rs, d_ge, d_gb, dC, dm};
}

// --------------------------------------------------------------------------------- the longitudinal force per note
// synth.NeuralPhysicalPiano._longitudinal: from the two longitudinal sums So, Se (lon [P, 2, L]), F_even = So^2 + Se^2,
// F_odd = -2 So Se; out = hp(F_even + F_odd) + sum_j c_j Re z_j with z_j,t = a_j z_j,t-1 + (j even ? F_odd : F_even),
// a_j = exp(-alpha / sr + i omega_j), hp the DC blocker y_t = r y_t-1 + x_t - x_t-1.
//
// The recurrences are linear, so time is cut into segments of LSEG samples, each run by its own thread from a zero
// state (pass 1); one thread per note then carries the true state across the segments' boundaries (pass 2), and each
// segment adds what that state contributes, a^(k+1) z_in (pass 3). The backward does the same for the adjoint
// recurrences (backwards in time, for the input's gradient) and for the forward-mode tangents w = dz / da (the decay's
// and frequencies' gradients): w_t = wl_t + a^(k+1) w_in + (k+1) a^k z_in. A warp's 32 segments advance together
// through tiles of 32 samples staged in shared memory, so that loads and stores are coalesced. Nothing of size [P, L]
// is stored besides the output; the final state is returned without gradient.

constexpr int LJ = 4;      // longitudinal modes at most
constexpr int LSEG = 512;  // samples per segment
constexpr int LW = 2;      // warps per block

struct cpx { float r, i; };
__device__ __forceinline__ cpx cmul(cpx a, cpx b) { return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r}; }
__device__ __forceinline__ cpx cadd(cpx a, cpx b) { return {a.r + b.r, a.i + b.i}; }
__device__ __forceinline__ cpx cfma(cpx a, cpx b, cpx c) {  // a b + c
  return {fmaf(a.r, b.r, fmaf(-a.i, b.i, c.r)), fmaf(a.r, b.i, fmaf(a.i, b.r, c.i))};
}

struct LongArgs {
  const float* lon; const float* c; const float* alpha; const float* omega; const float* hp0; const float* z0;
  int P, J, L, S; float sr, r;
};

struct Seg { int p, b, e; bool ok; };
__device__ __forceinline__ Seg seg_of(const LongArgs& A, int id) {
  Seg g;
  g.ok = id < A.P * A.S;
  const int i = g.ok ? id : 0;
  g.p = i / A.S;
  g.b = (i % A.S) * LSEG;
  g.e = g.ok ? min(g.b + LSEG, A.L) : g.b;
  return g;
}

// row k of a tile is lane k's segment at offset q: dst[k][i] = src[p_k row + off + b_k + q + i]
__device__ __forceinline__ void tile_load(float (*dst)[33], const float* src, size_t stride, size_t off, const Seg& g,
                                          int q, int lane) {
  for (int k = 0; k < 32; k++) {
    const int p = __shfl_sync(0xffffffff, g.p, k), b = __shfl_sync(0xffffffff, g.b, k);
    const int e = __shfl_sync(0xffffffff, g.e, k);
    const int t = b + q + lane;
    dst[k][lane] = t < e ? src[(size_t)p * stride + off + t] : 0.f;
  }
}

__device__ __forceinline__ void tile_store(float* dst, float (*src)[33], size_t stride, size_t off, const Seg& g, int q,
                                           int lane, bool add) {
  for (int k = 0; k < 32; k++) {
    const int p = __shfl_sync(0xffffffff, g.p, k), b = __shfl_sync(0xffffffff, g.b, k);
    const int e = __shfl_sync(0xffffffff, g.e, k);
    const int t = b + q + lane;
    if (t < e) {
      float* d = dst + (size_t)p * stride + off + t;
      *d = add ? *d + src[k][lane] : src[k][lane];
    }
  }
}

__device__ __forceinline__ void long_coef(const LongArgs& A, int p, cpx* a, float* c) {
  const float rho = expf(-A.alpha[p] / A.sr);
#pragma unroll
  for (int j = 0; j < LJ; j++) {
    if (j < A.J) {
      float s, co;
      sincosf(A.omega[p * A.J + j], &s, &co);
      a[j] = {rho * co, rho * s}; c[j] = A.c[p * A.J + j];
    } else {
      a[j] = {0.f, 0.f}; c[j] = 0.f;
    }
  }
}

// a^n in double (the carries: n up to LSEG)
__device__ __forceinline__ cpx cpow_n(const LongArgs& A, int p, int j, int n) {
  const double lr = -(double)A.alpha[p] / A.sr * n, ang = (double)A.omega[p * A.J + j] * n;
  const double m = exp(lr);
  return {(float)(m * cos(ang)), (float)(m * sin(ang))};
}

__device__ __forceinline__ float force_at(const LongArgs& A, int p, int t) {
  const size_t base = (size_t)p * 2 * A.L + t;
  const float So = A.lon[base], Se = A.lon[base + A.L];
  return fmaf(So, So, Se * Se) - 2.f * So * Se;
}

// forward pass 1: each segment from a zero state; its outputs, and its end state (y, z_j) at ends[p, s, 1 + 2J]
__global__ void long_fwd1(LongArgs A, float* __restrict__ out, float* __restrict__ ends) {
  __shared__ float sO[LW][32][33], sE[LW][32][33], sY[LW][32][33];
  const int id = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  const Seg g = seg_of(A, id);
  cpx a[LJ], z[LJ];
  float c[LJ];
  long_coef(A, g.p, a, c);
#pragma unroll
  for (int j = 0; j < LJ; j++) z[j] = {0.f, 0.f};
  float xp = 0.f, y = 0.f;
  if (g.ok) xp = g.b == 0 ? A.hp0[g.p * 2] : force_at(A, g.p, g.b - 1);
  for (int q = 0; q < LSEG; q += 32) {
    tile_load(sO[wp], A.lon, 2 * (size_t)A.L, 0, g, q, lane);
    tile_load(sE[wp], A.lon, 2 * (size_t)A.L, A.L, g, q, lane);
    __syncwarp();
    const int n = min(32, g.e - g.b - q);
    for (int i = 0; i < n; i++) {
      const float So = sO[wp][lane][i], Se = sE[wp][lane][i];
      const float fe = fmaf(So, So, Se * Se), fo = -2.f * So * Se, x = fe + fo;
      y = fmaf(A.r, y, x - xp);
      xp = x;
      float o = y;
#pragma unroll
      for (int j = 0; j < LJ; j++) {
        if (j < A.J) {
          z[j] = cfma(a[j], z[j], {(j & 1) ? fe : fo, 0.f});
          o = fmaf(c[j], z[j].r, o);
        }
      }
      sY[wp][lane][i] = o;
    }
    __syncwarp();
    tile_store(out, sY[wp], A.L, 0, g, q, lane, false);
    __syncwarp();
  }
  if (!g.ok) return;
  float* en = ends + (size_t)id * (1 + 2 * LJ);
  en[0] = y;
#pragma unroll
  for (int j = 0; j < LJ; j++) { en[1 + 2 * j] = z[j].r; en[2 + 2 * j] = z[j].i; }
}

// pass 2: per note, the true state entering each segment, ins[p, s, 1 + 2J] (y, z_j), and the final state
__global__ void long_fwd2(LongArgs A, const float* __restrict__ ends, float* __restrict__ ins, float* __restrict__ hp1,
                          float* __restrict__ z1) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= A.P) return;
  float y = A.hp0[p * 2 + 1];
  cpx z[LJ];
#pragma unroll
  for (int j = 0; j < LJ; j++) z[j] = j < A.J ? cpx{A.z0[(p * A.J + j) * 2], A.z0[(p * A.J + j) * 2 + 1]} : cpx{0.f, 0.f};
  for (int s = 0; s < A.S; s++) {
    const size_t id = (size_t)p * A.S + s;
    float* in = ins + id * (1 + 2 * LJ);
    const float* en = ends + id * (1 + 2 * LJ);
    in[0] = y;
#pragma unroll
    for (int j = 0; j < LJ; j++) { in[1 + 2 * j] = z[j].r; in[2 + 2 * j] = z[j].i; }
    const int n = min(LSEG, A.L - s * LSEG);
    y = en[0] + powf(A.r, (float)n) * y;
#pragma unroll
    for (int j = 0; j < LJ; j++)
      if (j < A.J) z[j] = cadd({en[1 + 2 * j], en[2 + 2 * j]}, cmul(cpow_n(A, p, j, n), z[j]));
  }
  hp1[p * 2] = force_at(A, p, A.L - 1); hp1[p * 2 + 1] = y;
#pragma unroll
  for (int j = 0; j < LJ; j++)
    if (j < A.J) { z1[(p * A.J + j) * 2] = z[j].r; z1[(p * A.J + j) * 2 + 1] = z[j].i; }
}

// pass 3: each segment adds its entering state's ring, r^(k+1) y_in + sum_j c_j Re a_j^(k+1) z_in,j
__global__ void long_fwd3(LongArgs A, const float* __restrict__ ins, float* __restrict__ out) {
  __shared__ float sY[LW][32][33];
  const int id = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  const Seg g = seg_of(A, id);
  cpx a[LJ], z[LJ];
  float c[LJ];
  long_coef(A, g.p, a, c);
  const float* in = ins + (size_t)(g.ok ? id : 0) * (1 + 2 * LJ);
  float y = in[0];
#pragma unroll
  for (int j = 0; j < LJ; j++) z[j] = {in[1 + 2 * j], in[2 + 2 * j]};
  for (int q = 0; q < LSEG; q += 32) {
    const int n = min(32, g.e - g.b - q);
    for (int i = 0; i < n; i++) {
      y *= A.r;
      float o = y;
#pragma unroll
      for (int j = 0; j < LJ; j++) {
        if (j < A.J) {
          z[j] = cmul(a[j], z[j]);
          o = fmaf(c[j], z[j].r, o);
        }
      }
      sY[wp][lane][i] = o;
    }
    __syncwarp();
    tile_store(out, sY[wp], A.L, 0, g, q, lane, true);
    __syncwarp();
  }
}

// backward pass 1, per segment from zero: forward in time the tangent sums (A1 = sum g Re zl, SW = sum g wl, G0 = sum
// g a^(k+1), G1 = sum g (k+1) a^k) and the end states zl, wl; backward in time the adjoints mu_j, nu (the DC blocker)
// from zero, the input's gradient they give, and their values at the segment's first sample.
// rec[p, s, :]: A1 [J], SW, G0, G1, ZL, WL, MU [2J each], NU [1]
constexpr int LREC = LJ + 12 * LJ + 1;

__global__ void long_bwd1(LongArgs A, const float* __restrict__ gout, float* __restrict__ d_lon, float* __restrict__ rec) {
  __shared__ float sO[LW][32][33], sE[LW][32][33], sG[LW][32][33];
  const int id = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  const Seg g = seg_of(A, id);
  cpx a[LJ];
  float c[LJ];
  long_coef(A, g.p, a, c);
  float* R = rec + (size_t)(g.ok ? id : 0) * LREC;
  {
    cpx z[LJ], w[LJ], SW[LJ], G0[LJ], G1[LJ], pw[LJ], kp[LJ];
    float A1[LJ];
#pragma unroll
    for (int j = 0; j < LJ; j++) {
      z[j] = w[j] = SW[j] = G0[j] = G1[j] = {0.f, 0.f};
      pw[j] = a[j]; kp[j] = {1.f, 0.f};  // a^(k+1), (k+1) a^k at k = 0
      A1[j] = 0.f;
    }
    for (int q = 0; q < LSEG; q += 32) {
      tile_load(sO[wp], A.lon, 2 * (size_t)A.L, 0, g, q, lane);
      tile_load(sE[wp], A.lon, 2 * (size_t)A.L, A.L, g, q, lane);
      tile_load(sG[wp], gout, A.L, 0, g, q, lane);
      __syncwarp();
      const int n = min(32, g.e - g.b - q);
      for (int i = 0; i < n; i++) {
        const float So = sO[wp][lane][i], Se = sE[wp][lane][i], gt = sG[wp][lane][i];
        const float fe = fmaf(So, So, Se * Se), fo = -2.f * So * Se;
#pragma unroll
        for (int j = 0; j < LJ; j++) {
          if (j < A.J) {
            w[j] = cfma(a[j], w[j], z[j]);
            z[j] = cfma(a[j], z[j], {(j & 1) ? fe : fo, 0.f});
            A1[j] = fmaf(gt, z[j].r, A1[j]);
            SW[j] = {fmaf(gt, w[j].r, SW[j].r), fmaf(gt, w[j].i, SW[j].i)};
            G0[j] = {fmaf(gt, pw[j].r, G0[j].r), fmaf(gt, pw[j].i, G0[j].i)};
            G1[j] = {fmaf(gt, kp[j].r, G1[j].r), fmaf(gt, kp[j].i, G1[j].i)};
            kp[j] = cfma(a[j], kp[j], pw[j]);  // (k+2) a^(k+1) = a (k+1) a^k + a^(k+1)
            pw[j] = cmul(a[j], pw[j]);
          }
        }
      }
      __syncwarp();
    }
    if (g.ok) {
#pragma unroll
      for (int j = 0; j < LJ; j++) {
        R[j] = A1[j];
        float* q = R + LJ + 2 * j;
        q[0] = SW[j].r; q[1] = SW[j].i;
        q[2 * LJ] = G0[j].r; q[2 * LJ + 1] = G0[j].i;
        q[4 * LJ] = G1[j].r; q[4 * LJ + 1] = G1[j].i;
        q[6 * LJ] = z[j].r; q[6 * LJ + 1] = z[j].i;
        q[8 * LJ] = w[j].r; q[8 * LJ + 1] = w[j].i;
      }
    }
  }
  cpx mu[LJ];
#pragma unroll
  for (int j = 0; j < LJ; j++) mu[j] = {0.f, 0.f};
  float nu_next = 0.f;
  for (int q = ((LSEG - 1) / 32) * 32; q >= 0; q -= 32) {
    tile_load(sO[wp], A.lon, 2 * (size_t)A.L, 0, g, q, lane);
    tile_load(sE[wp], A.lon, 2 * (size_t)A.L, A.L, g, q, lane);
    tile_load(sG[wp], gout, A.L, 0, g, q, lane);
    __syncwarp();
    const int n = min(32, g.e - g.b - q);
    for (int i = n - 1; i >= 0; i--) {
      const float So = sO[wp][lane][i], Se = sE[wp][lane][i], gt = sG[wp][lane][i];
      const float nu = fmaf(A.r, nu_next, gt);
      const float dx = nu - nu_next;
      nu_next = nu;
      float dfe = dx, dfo = dx;
#pragma unroll
      for (int j = 0; j < LJ; j++) {
        if (j < A.J) {
          mu[j] = cfma(a[j], mu[j], {gt, 0.f});
          if (j & 1) dfe = fmaf(c[j], mu[j].r, dfe);
          else dfo = fmaf(c[j], mu[j].r, dfo);
        }
      }
      sO[wp][lane][i] = 2.f * (So * dfe - Se * dfo);
      sE[wp][lane][i] = 2.f * (Se * dfe - So * dfo);
    }
    __syncwarp();
    tile_store(d_lon, sO[wp], 2 * (size_t)A.L, 0, g, q, lane, false);
    tile_store(d_lon, sE[wp], 2 * (size_t)A.L, A.L, g, q, lane, false);
    __syncwarp();
  }
  if (!g.ok) return;
#pragma unroll
  for (int j = 0; j < LJ; j++) { R[LJ + 10 * LJ + 2 * j] = mu[j].r; R[LJ + 10 * LJ + 2 * j + 1] = mu[j].i; }
  R[LREC - 1] = nu_next;
}

// backward pass 2, per note: the true entering states (z_in, w_in) forward over the segments, the parameters'
// gradients; the true adjoints after each segment (mu_next, nu_next) backward over them, nxt[p, s, 2J + 1]
__global__ void long_bwd2(LongArgs A, const float* __restrict__ rec, float* __restrict__ nxt, float* __restrict__ d_c,
                          float* __restrict__ d_alpha, float* __restrict__ d_omega) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= A.P) return;
  cpx a[LJ];
  float c[LJ];
  long_coef(A, p, a, c);
  cpx z[LJ], w[LJ], Sw[LJ];
  float Sc[LJ];
#pragma unroll
  for (int j = 0; j < LJ; j++) {
    z[j] = j < A.J ? cpx{A.z0[(p * A.J + j) * 2], A.z0[(p * A.J + j) * 2 + 1]} : cpx{0.f, 0.f};
    w[j] = Sw[j] = {0.f, 0.f};
    Sc[j] = 0.f;
  }
  for (int s = 0; s < A.S; s++) {
    const float* R = rec + ((size_t)p * A.S + s) * LREC;
    const int n = min(LSEG, A.L - s * LSEG);
#pragma unroll
    for (int j = 0; j < LJ; j++) {
      if (j < A.J) {
        const float* q = R + LJ + 2 * j;
        const cpx SW = {q[0], q[1]}, G0 = {q[2 * LJ], q[2 * LJ + 1]}, G1 = {q[4 * LJ], q[4 * LJ + 1]};
        const cpx ZL = {q[6 * LJ], q[6 * LJ + 1]}, WL = {q[8 * LJ], q[8 * LJ + 1]};
        Sc[j] += R[j] + cmul(z[j], G0).r;
        Sw[j] = cadd(Sw[j], cadd(SW, cadd(cmul(w[j], G0), cmul(z[j], G1))));
        const cpx an = cpow_n(A, p, j, n), an1 = cpow_n(A, p, j, n - 1);
        const cpx nz = cadd(ZL, cmul(an, z[j]));
        w[j] = cadd(WL, cadd(cmul(an, w[j]), cmul({(float)n * an1.r, (float)n * an1.i}, z[j])));
        z[j] = nz;
      }
    }
  }
  float da = 0.f;
#pragma unroll
  for (int j = 0; j < LJ; j++) {
    if (j < A.J) {
      const cpx aw = cmul(a[j], Sw[j]);  // dz = w da; da / d alpha = -a / sr, da / d omega = i a
      da += -c[j] * aw.r / A.sr;
      d_omega[p * A.J + j] = -c[j] * aw.i;
      d_c[p * A.J + j] = Sc[j];
    }
  }
  d_alpha[p] = da;
  cpx mu[LJ];
#pragma unroll
  for (int j = 0; j < LJ; j++) mu[j] = {0.f, 0.f};
  float nu = 0.f;
  for (int s = A.S - 1; s >= 0; s--) {
    const size_t id = (size_t)p * A.S + s;
    float* X = nxt + id * (2 * LJ + 1);
    const float* R = rec + id * LREC;
#pragma unroll
    for (int j = 0; j < LJ; j++) { X[2 * j] = mu[j].r; X[2 * j + 1] = mu[j].i; }
    X[2 * LJ] = nu;
    const int n = min(LSEG, A.L - s * LSEG);
#pragma unroll
    for (int j = 0; j < LJ; j++)
      if (j < A.J) mu[j] = cadd({R[LJ + 10 * LJ + 2 * j], R[LJ + 10 * LJ + 2 * j + 1]}, cmul(cpow_n(A, p, j, n), mu[j]));
    nu = R[LREC - 1] + powf(A.r, (float)n) * nu;
  }
}

// backward pass 3: each segment adds the input gradient of the adjoints entering from its end
__global__ void long_bwd3(LongArgs A, const float* __restrict__ nxt, float* __restrict__ d_lon) {
  __shared__ float sO[LW][32][33], sE[LW][32][33];
  const int id = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  const Seg g = seg_of(A, id);
  cpx a[LJ], mu[LJ];
  float c[LJ];
  long_coef(A, g.p, a, c);
  const float* X = nxt + (size_t)(g.ok ? id : 0) * (2 * LJ + 1);
#pragma unroll
  for (int j = 0; j < LJ; j++) mu[j] = {X[2 * j], X[2 * j + 1]};
  float prev = X[2 * LJ];  // r^(e - t) nu_next at t + 1
  for (int q = ((LSEG - 1) / 32) * 32; q >= 0; q -= 32) {
    tile_load(sO[wp], A.lon, 2 * (size_t)A.L, 0, g, q, lane);
    tile_load(sE[wp], A.lon, 2 * (size_t)A.L, A.L, g, q, lane);
    __syncwarp();
    const int n = min(32, g.e - g.b - q);
    for (int i = n - 1; i >= 0; i--) {
      const float So = sO[wp][lane][i], Se = sE[wp][lane][i];
      const float cur = A.r * prev;
      const float dx = cur - prev;
      prev = cur;
      float dfe = dx, dfo = dx;
#pragma unroll
      for (int j = 0; j < LJ; j++) {
        if (j < A.J) {
          mu[j] = cmul(a[j], mu[j]);
          if (j & 1) dfe = fmaf(c[j], mu[j].r, dfe);
          else dfo = fmaf(c[j], mu[j].r, dfo);
        }
      }
      sO[wp][lane][i] = 2.f * (So * dfe - Se * dfo);
      sE[wp][lane][i] = 2.f * (Se * dfe - So * dfo);
    }
    __syncwarp();
    tile_store(d_lon, sO[wp], 2 * (size_t)A.L, 0, g, q, lane, true);
    tile_store(d_lon, sE[wp], 2 * (size_t)A.L, A.L, g, q, lane, true);
    __syncwarp();
  }
}

static LongArgs long_args(torch::Tensor lon, torch::Tensor c, torch::Tensor alpha, torch::Tensor omega,
                          torch::Tensor hp0, torch::Tensor z0, double sr, double r_hp) {
  LongArgs A;
  A.lon = lon.data_ptr<float>(); A.c = c.data_ptr<float>(); A.alpha = alpha.data_ptr<float>();
  A.omega = omega.data_ptr<float>(); A.hp0 = hp0.numel() ? hp0.data_ptr<float>() : nullptr;
  A.z0 = z0.data_ptr<float>();
  A.P = (int)lon.size(0); A.L = (int)lon.size(2); A.J = (int)c.size(1); A.S = (A.L + LSEG - 1) / LSEG;
  A.sr = (float)sr; A.r = (float)r_hp;
  TORCH_CHECK(A.J <= LJ, "longitudinal modes: at most 4");
  return A;
}

std::vector<torch::Tensor> long_forward(torch::Tensor lon, torch::Tensor c, torch::Tensor alpha, torch::Tensor omega,
                                        torch::Tensor hp0, torch::Tensor z0, double sr, double r_hp) {
  LongArgs A = long_args(lon, c, alpha, omega, hp0, z0, sr, r_hp);
  auto out = torch::zeros({A.P, A.L}, lon.options());
  auto hp1 = hp0.clone(), z1 = z0.clone();
  if (A.P == 0 || A.L == 0) return {out, hp1, z1};
  auto ends = torch::empty({A.P, A.S, 1 + 2 * LJ}, lon.options()), ins = torch::empty_like(ends);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n = A.P * A.S, th = 32 * LW;
  long_fwd1<<<(n + th - 1) / th, th, 0, stream>>>(A, out.data_ptr<float>(), ends.data_ptr<float>());
  long_fwd2<<<(A.P + 127) / 128, 128, 0, stream>>>(A, ends.data_ptr<float>(), ins.data_ptr<float>(),
                                                   hp1.data_ptr<float>(), z1.data_ptr<float>());
  long_fwd3<<<(n + th - 1) / th, th, 0, stream>>>(A, ins.data_ptr<float>(), out.data_ptr<float>());
  return {out, hp1, z1};
}

std::vector<torch::Tensor> long_backward(torch::Tensor gout, torch::Tensor lon, torch::Tensor c, torch::Tensor alpha,
                                         torch::Tensor omega, torch::Tensor z0, double sr, double r_hp) {
  LongArgs A = long_args(lon, c, alpha, omega, torch::zeros({0}, lon.options()), z0, sr, r_hp);
  auto d_lon = torch::zeros_like(lon), d_c = torch::zeros_like(c), d_alpha = torch::zeros_like(alpha),
       d_omega = torch::zeros_like(omega);
  if (A.P == 0 || A.L == 0) return {d_lon, d_c, d_alpha, d_omega};
  auto rec = torch::empty({A.P, A.S, LREC}, lon.options());
  auto nxt = torch::empty({A.P, A.S, 2 * LJ + 1}, lon.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int n = A.P * A.S, th = 32 * LW;
  long_bwd1<<<(n + th - 1) / th, th, 0, stream>>>(A, gout.data_ptr<float>(), d_lon.data_ptr<float>(),
                                                  rec.data_ptr<float>());
  long_bwd2<<<(A.P + 127) / 128, 128, 0, stream>>>(A, rec.data_ptr<float>(), nxt.data_ptr<float>(),
                                                   d_c.data_ptr<float>(), d_alpha.data_ptr<float>(),
                                                   d_omega.data_ptr<float>());
  long_bwd3<<<(n + th - 1) / th, th, 0, stream>>>(A, nxt.data_ptr<float>(), d_lon.data_ptr<float>());
  return {d_lon, d_c, d_alpha, d_omega};
}
