from pyrbit.actr import ACTR, logregplot, gen_data

import numpy
import matplotlib.pyplot as plt

plt.style.use("/home/juliengori/Documents/VC/mplstyles/seaborn-small-fig.mplstyle")


SEED = 1234

d = 0.6
tau = -0.7
s = 0.25


fig, axs = plt.subplots(nrows=1, ncols=2)

N = 2000
actr = ACTR(1, 0.5, 0.25, -0.7, buffer_size=16, seed=SEED)
recalls, deltatis = gen_data(actr, N, seed=SEED)

ax = logregplot(
    0.5,
    deltatis,
    recalls,
    ax=axs[0],
    line_kws={"color": "green"},
    recall_event_kwargs={"scatter_kws": {"marker": "o", "s": 10}},
)
ax.legend()


N = int(1e5)
actr = ACTR(1, 0.5, 0.25, -0.7, buffer_size=16, seed=SEED)
recalls, deltatis = gen_data(actr, N, seed=SEED)

ax = logregplot(
    0.5,
    deltatis,
    recalls,
    ax=axs[1],
    line_kws={"color": "green"},
    recall_event_kwargs={"scatter_kws": {"marker": "o", "s": 10}},
)
ax.legend()
plt.tight_layout()
plt.show()
