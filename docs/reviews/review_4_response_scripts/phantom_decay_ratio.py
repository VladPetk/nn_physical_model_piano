"""Can decay tell a phantom partial from transverse partial 2j? In the fitted model, compare the phantom's decay rate
(alpha_j + alpha_j = 2 alpha_j) with partial 2j's own rate, for the prompt and the aftersound mode."""
import sys
import torch
sys.path.insert(0, ".")
from pianonn.render import load_model

m = load_model("runs/round1_trial/main/best.pt")
for midi in (55, 60, 67, 72, 79):
    ki = torch.tensor([[midi - 21]])
    u = torch.full(ki.shape, 50 / 127)
    with torch.no_grad():
        md = m.physics.modes(ki, u, torch.zeros_like(u), torch.tensor([9]))  # condition 9 = MAESTRO 2018
    a = md["alpha"][0, 0]  # [partials, modes]: 0 = prompt, -1 = aftersound
    print(f"MIDI {midi}: 2 alpha_j / alpha_2j  prompt " + " ".join(f"j={j}:{2 * float(a[j - 1, 0]) / float(a[2 * j - 1, 0]):.2f}" for j in (3, 4, 5, 6, 8))
          + " | aftersound " + " ".join(f"j={j}:{2 * float(a[j - 1, -1]) / float(a[2 * j - 1, -1]):.2f}" for j in (3, 4, 5, 6, 8)))
