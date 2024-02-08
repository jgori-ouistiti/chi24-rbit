from pyrbit.ef import (
    ExponentialForgetting,
    identify_ef_from_recall_sequence,
    covar_delta_method_log_alpha,
    ef_observed_information_matrix,
)
from pyrbit.mle_utils import (
    CI_asymptotical,
)
import numpy
from tqdm import tqdm
import pandas
import json
import matplotlib.pyplot as plt


plt.style.use(style="fivethirtyeight")

SEED = 789
N_logspace = 4
REPETITIONS = 1000


def simulate_arbitrary_traj(ef, k_vector, deltas):
    recall = []
    for k, d in zip(k_vector, deltas):
        ef.update(0, 0, N=k)
        recall.append(ef.query_item(0, d))
    return recall


SIZE = 100
N = 1000
GEN_DATA = False
if GEN_DATA:
    alpha_results = numpy.zeros((SIZE, SIZE))
    beta_results = numpy.zeros((SIZE, SIZE))
    for nalpha, alpha in tqdm(enumerate(numpy.linspace(-5, -1, SIZE))):
        for nbeta, beta in tqdm(
            enumerate(numpy.linspace(start=0.05, stop=0.95, num=SIZE))
        ):
            ALPHA_TRUE = 10**alpha
            BETA_TRUE = beta
            ef = ExponentialForgetting(1, ALPHA_TRUE, BETA_TRUE, seed=SEED)
            rng = numpy.random.default_rng(seed=SEED)
            ################

            k_vector = rng.integers(low=0, high=10, size=N)
            deltas = rng.integers(low=0, high=5000, size=N)
            recall_probs = simulate_arbitrary_traj(ef, k_vector, deltas)
            recall = [rp[0] for rp in recall_probs]
            optim_kwargs = {"method": "L-BFGS-B", "bounds": [(1e-5, 0.1), (0, 0.99)]}

            verbose = False
            guess = (1e-3, 0.5)
            inference_results = identify_ef_from_recall_sequence(
                recall,
                deltas,
                k_vector=[k - 1 for k in k_vector],
                optim_kwargs=optim_kwargs,
                verbose=verbose,
                guess=guess,
                basin_hopping=True,
                basin_hopping_kwargs={"niter": 5},
            )

            alpha_results[nalpha, nbeta] = inference_results.x[0]
            beta_results[nalpha, nbeta] = inference_results.x[1]

    with open("save_data/ef_correlation.json", "w") as _file:
        data_json = [alpha_results.tolist(), beta_results.tolist()]
        json.dump(data_json, _file)
else:
    with open("save_data/ef_correlation.json", "r") as _file:
        data_json = json.load(_file)
        alpha_results, beta_results = numpy.array(data_json[0]), numpy.array(
            data_json[1]
        )


import seaborn
import matplotlib.pyplot as plt

_x = numpy.tile([numpy.linspace(-5, -1, SIZE)], [1, SIZE]).squeeze()
_y = numpy.tile(numpy.linspace(start=0.05, stop=0.95, num=SIZE), [1, SIZE]).squeeze()


_alpha = alpha_results.transpose(1, 0).ravel()
_beta = beta_results.ravel()


fig, axs = plt.subplots(nrows=1, ncols=2)
cp = numpy.corrcoef(_x, numpy.log10(_alpha))[0, 1]

seaborn.regplot(
    x=_x,
    y=numpy.log10(_alpha),
    ax=axs[0],
    label=r"$\log_{10}\alpha$",
    scatter_kws={"marker": ".", "s": 2},
    fit_reg=True,
    line_kws={"color": "orange", "label": rf"$\rho={cp:.3f}$"},
)
cp = numpy.corrcoef(_y, _beta)[0, 1]

seaborn.regplot(
    x=_y,
    y=_beta,
    ax=axs[1],
    label=r"$\beta$",
    fit_reg=True,
    scatter_kws={"marker": ".", "s": 2},
    line_kws={"color": "orange", "label": rf"$\rho={cp:.3f}$"},
)
axs[0].set_xlabel("Ground Truth value")
axs[0].set_ylabel("Estimated value")
axs[1].set_xlabel("Ground Truth value")
axs[1].set_ylabel("Estimated value")
axs[0].legend(markerscale=4)
axs[1].legend(markerscale=4)
plt.tight_layout()
plt.show()
