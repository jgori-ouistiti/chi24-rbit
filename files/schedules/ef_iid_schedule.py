import numpy
import matplotlib.pyplot as plt

plt.style.use(style="fivethirtyeight")

import json

from pyrbit.ef import ExponentialForgetting
from pyrbit.mem_utils import Schedule, experiment, GaussianPopulation
from pyrbit.information import (
    gen_hessians,
    compute_full_observed_information,
)


def play_iid_schedule(ef, N):
    k_vector = rng.integers(low=-1, high=10, size=N)
    deltas = rng.integers(low=1, high=5000, size=N)
    recall_probs = simulate_arbitrary_traj(ef, k_vector, deltas)
    recall = [rp[0] for rp in recall_probs]
    k_repetition = [k for k in k_vector]
    return recall, deltas, k_repetition


def simulate_arbitrary_traj(ef, k_vector, deltas):
    recall = []
    for k, d in zip(k_vector, deltas):
        ef.update(0, 0, N=(k + 1))
        recall.append(ef.query_item(0, d))
    return recall


SEED = 999
N = 150
rng = numpy.random.default_rng(seed=SEED)
REPETITION = 1000

ALPHA_TRUE = 1e-2
BETA_TRUE = 4e-1


SUBSAMPLE = 15
subsample_sequence = numpy.logspace(0.5, numpy.log10(N), int(N / SUBSAMPLE))
GEN_DATA = True

if GEN_DATA:
    ef = ExponentialForgetting(1, ALPHA_TRUE, BETA_TRUE, seed=SEED)
    play_schedule = play_iid_schedule
    play_schedule_args = (N,)

    optim_kwargs = {
        "method": "L-BFGS-B",
        "bounds": [(1e-5, 0.1), (0, 0.99)],
        "guess": (1e-3, 0.7),
        "verbose": False,
    }
    filename = "save_data/ef_schedule_iid.json"
    json_data, _ = gen_hessians(
        N,
        REPETITION,
        [ALPHA_TRUE, BETA_TRUE],
        ef,
        play_schedule,
        subsample_sequence,
        play_schedule_args=play_schedule_args,
        optim_kwargs=optim_kwargs,
        filename=filename,
    )
else:
    with open("save_data/ef_schedule_iid.json", "r") as _file:
        json_data = json.load(
            _file,
        )


recall_array = numpy.asarray(json_data["recall_array"])
observed_hessians = numpy.asarray(json_data["observed_hessians"])
observed_cum_hessians = numpy.asarray(json_data["observed_cum_hessians"])

estimated_parameters = numpy.asarray(json_data["estimated_parameters"])
recall_array = recall_array.transpose(1, 0)

recall_kwargs = {
    "x_bins": 10,
}
observed_information_kwargs = {"x_bins": 10, "cum_color": "orange"}

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
(
    fischer_information,
    agg_data,
    cumulative_information,
    cum_inf,
) = compute_full_observed_information(
    [ALPHA_TRUE, BETA_TRUE],
    recall_array,
    observed_hessians,
    estimated_parameters,
    subsample_sequence,
    axs=axs.ravel(),
    recall_kwargs=recall_kwargs,
    observed_information_kwargs=observed_information_kwargs,
    bias_kwargs=None,
    std_kwargs=None,
)
plt.tight_layout(w_pad=2, h_pad=2)
plt.savefig("images/sup_iid.pdf")
plt.show()
