#include <torch/extension.h>
#include <vector>

#define BANK_ARGS torch::Tensor freq, torch::Tensor alpha, torch::Tensor amp, torch::Tensor adamp, torch::Tensor g0, \
  torch::Tensor w0, torch::Tensor g1, torch::Tensor w1, torch::Tensor onset, torch::Tensor tc, torch::Tensor c_onset, \
  torch::Tensor rs, torch::Tensor rsd, torch::Tensor C, torch::Tensor row, torch::Tensor m, torch::Tensor nv, \
  int64_t hist, int64_t hop, int64_t chop, int64_t chunk, int64_t Q0, int64_t start, int64_t L, double sr

torch::Tensor osc_forward(BANK_ARGS);
std::vector<torch::Tensor> osc_backward(torch::Tensor gy, BANK_ARGS);
std::vector<torch::Tensor> bus_forward(BANK_ARGS, torch::Tensor ge, torch::Tensor gb, torch::Tensor bidx, int64_t NM,
                                       int64_t n_examples, int64_t n_rows, bool keys);
std::vector<torch::Tensor> bus_backward(torch::Tensor gmix, torch::Tensor glon, torch::Tensor gkeys, BANK_ARGS,
                                        torch::Tensor ge, torch::Tensor gb, torch::Tensor bidx, int64_t NM, bool keys);
std::vector<torch::Tensor> long_forward(torch::Tensor lon, torch::Tensor c, torch::Tensor alpha, torch::Tensor omega,
                                        torch::Tensor hp0, torch::Tensor z0, double sr, double r_hp);
std::vector<torch::Tensor> long_backward(torch::Tensor gout, torch::Tensor lon, torch::Tensor c, torch::Tensor alpha,
                                         torch::Tensor omega, torch::Tensor z0, double sr, double r_hp);
std::vector<torch::Tensor> reson_forward(torch::Tensor drive, torch::Tensor es, torch::Tensor alpha, torch::Tensor adamp,
                                         torch::Tensor freq, torch::Tensor gin, torch::Tensor z0r, torch::Tensor z0i,
                                         double sr, double max_decay);
std::vector<torch::Tensor> reson_backward(torch::Tensor gy, torch::Tensor gzr, torch::Tensor gzi, torch::Tensor drive,
                                          torch::Tensor es, torch::Tensor alpha, torch::Tensor adamp, torch::Tensor freq,
                                          torch::Tensor gin, torch::Tensor z0r, torch::Tensor z0i, torch::Tensor zr,
                                          torch::Tensor zi, double sr, double max_decay);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("osc_forward", &osc_forward, "string bank forward");
  m.def("osc_backward", &osc_backward, "string bank backward");
  m.def("bus_forward", &bus_forward, "coupled strings' bus bank forward");
  m.def("bus_backward", &bus_backward, "coupled strings' bus bank backward");
  m.def("long_forward", &long_forward, "longitudinal force forward");
  m.def("long_backward", &long_backward, "longitudinal force backward");
  m.def("reson_forward", &reson_forward, "sympathetic resonators forward");
  m.def("reson_backward", &reson_backward, "sympathetic resonators backward");
}
