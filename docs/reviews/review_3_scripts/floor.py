"""How much of the MR-STFT loss comes from the recording's noise floor that the model cannot produce?"""
import torch, math
from pianonn.diagnostics import _perf
from pianonn.render import load_model
from pianonn.losses import MultiResolutionSTFTLoss, _mag
torch.manual_seed(0)
m = load_model(use_sympathetic=False)
sr = m.cfg.sample_rate
n = 3 * sr
notes = [(60, 0.1, 1.5, 80), (64, 0.6, 2.0, 70), (67, 1.1, 2.8, 90), (48, 0.2, 2.9, 85)]
with torch.no_grad():
    clean = m(_perf(m, n, notes), n, generator=torch.Generator().manual_seed(0))["audio"]
    other = m(_perf(m, n, [(p, a, b, v - 10) for p, a, b, v in notes]), n, generator=torch.Generator().manual_seed(1))["audio"]
peak = clean.abs().max()
loss = MultiResolutionSTFTLoss()
print(f"peak {20*math.log10(peak):.1f} dBFS")
for floor_db in (None, -80, -70, -60, -50):
    tgt = clean.clone()
    if floor_db is not None:
        tgt = tgt + torch.randn_like(tgt) * 10 ** (floor_db / 20)   # stationary noise floor in dBFS
    l_same = loss(clean, tgt).item()        # identical piano, only the floor differs
    l_diff = loss(other, tgt).item()        # every note 10 velocity units softer + different knock seed
    print(f"target floor {str(floor_db):>5} dBFS: loss(model = same piano) {l_same:.3f}   loss(model = 10-velocity-off piano) {l_diff:.3f}")
