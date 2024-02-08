from pyrbit.mle_utils import CI_asymptotical, confidence_ellipse
from pyrbit.ef import (
    ExponentialForgetting,
    diagnostics,
)
import numpy
import matplotlib.pyplot as plt

plt.style.use(style="fivethirtyeight")

SEED = None
N = int(1e5)

alpha = 0.001
beta = 0.4

ef = ExponentialForgetting(1, alpha, beta, seed=SEED)
rng = numpy.random.default_rng(seed=SEED)


def simulate_arbitrary_traj(ef, k_vector, deltas):
    recall = []
    for k, d in zip(k_vector, deltas):
        ef.update(0, 0, N=k)
        recall.append(ef.query_item(0, d))
    return recall


# ============== Simulate some data
k_vector = rng.integers(low=0, high=10, size=N)
deltas = rng.integers(low=1, high=5000, size=N)
recall_probs = simulate_arbitrary_traj(ef, k_vector, deltas)
recall = [rp[0] for rp in recall_probs]
k_repetition = [k - 1 for k in k_vector]

# ================ Run diagnostics

fig, (fg, ax, estim) = diagnostics(alpha, beta, k_repetition, deltas, recall)
fig.tight_layout()
plt.show()
