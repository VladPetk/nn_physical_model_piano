import math

import numpy as np
import torch
from torch import nn

from pianonn.metrics import clustered_mean, piece_gains_loo
from pianonn.train import WeightAverage


def test_clustered_error_counts_pieces_not_excerpts():
    # two pieces, each internally identical: four excerpts but only two independent values
    m, se, g = clustered_mean([1.0, 1.0, 3.0, 3.0], ["a", "a", "b", "b"])
    assert m == 2.0 and g == 2
    assert math.isclose(se, 1.0)  # the two piece means, 1 and 3: se of their mean is 1
    naive = np.std([1.0, 1.0, 3.0, 3.0], ddof=1) / 2
    assert se > naive


def test_piece_gain_leaves_the_scored_excerpt_out():
    gains, alone = piece_gains_loo([1.0, 1.0, 2.0], [2.0, 2.0, 2.0], ["a", "a", "b"])
    assert np.allclose(gains[:2], 10 * math.log10(2.0))
    assert gains[2] == 0.0 and alone == 1
    # an excerpt's own level does not enter its gain
    g2, _ = piece_gains_loo([1.0, 100.0, 1.0], [2.0, 2.0, 2.0], ["a", "a", "a"])
    assert math.isclose(g2[1], 10 * math.log10(4.0 / 2.0))


def test_weight_average_follows_and_swaps_back():
    net = nn.Linear(2, 1)
    with torch.no_grad():
        net.weight.zero_()
    avg = WeightAverage(net, decay=0.9)
    with torch.no_grad():
        net.weight.fill_(1.0)
    for _ in range(200):
        avg.update(net)
    w = avg.shadow["weight"]
    assert torch.allclose(w, torch.ones_like(w), atol=1e-3)  # converged to the constant weights
    avg2 = WeightAverage(net, decay=0.9)  # starts at 1
    with torch.no_grad():
        net.weight.fill_(-1.0)
    avg2.update(net)
    assert -1.0 < float(avg2.shadow["weight"][0, 0]) < 1.0  # moved part of the way, not all of it
    with avg2.applied(net):
        assert torch.equal(net.weight, avg2.shadow["weight"])
    assert torch.all(net.weight == -1.0)  # the raw weights are back
    state = avg2.state_dict()
    avg3 = WeightAverage(net, decay=0.9)
    avg3.load_state_dict(state)
    assert torch.equal(avg3.shadow["weight"], avg2.shadow["weight"]) and avg3.n == avg2.n
