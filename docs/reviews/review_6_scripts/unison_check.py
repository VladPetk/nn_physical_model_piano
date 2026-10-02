"""Did the unison mistuning (the beats) ever move from its random draw? Run from the repository root."""
import torch, sys
sys.path.insert(0, ".")
from pianonn.config import PianoConfig
from pianonn.physics import PianoPhysics
for ck in ("runs/phase3/step4/train/last.pt", "runs/phase6/train_run/train/last.pt"):
    st = torch.load(ck, map_location="cpu")
    cfg = PianoConfig.from_dict(st["cfg"])
    init = PianoPhysics(cfg).raw_unison.detach()
    cur = st["model"]["physics.raw_unison"]
    b = lambda r: 5 * torch.tanh(r / 5)
    d = (b(cur) - b(init))
    print(ck, "unison detune cents: init |mean| %.2f, moved by mean |d| %.3f max %.3f; corr init-vs-now %.4f" % (
        b(init).abs().mean(), d.abs().mean(), d.abs().max(), torch.corrcoef(torch.stack([b(init).flatten(), b(cur).flatten()]))[0, 1]))
    pg = 0.92 * torch.tanh(st["model"]["physics.partial_gain"] / 0.92) * 8.686
    print("   partial_gain dB: rms %.2f, |x|>4 dB: %.1f%%" % (pg.pow(2).mean().sqrt(), 100 * (pg.abs() > 4).float().mean()))
    ra = st["model"]["physics.raw_after"]
    print("   raw_after (aftersound level offset, nats): rms %.2f" % ra.pow(2).mean().sqrt())
