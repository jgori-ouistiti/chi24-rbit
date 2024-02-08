import numpy
import matplotlib.pyplot as plt

import json

from pyrbit.ef import ExponentialForgetting
from pyrbit.mem_utils import (
    Schedule,
    experiment,
    GaussianPopulation,
)
from pyrbit.information import (
    gen_hessians,
    compute_full_observed_information,
    compute_observed_information,
)

plt.style.use(style="fivethirtyeight")


def serialize_experiment(data, times):
    _, block_size = data.shape
    k_vector = []
    deltas = []
    for i in data:
        k_vector += [k for k in range(-1, block_size - 1)]
        deltas += [numpy.infty] + numpy.diff(numpy.asarray(times)).tolist()
    data = data.reshape(
        -1,
    )
    return data, numpy.asarray(k_vector), numpy.asarray(deltas)


def play_replicated_schedule(population_model, times):
    items = [0 for i in times]
    schedule_one = Schedule(items, times)
    data = experiment(population_model, schedule_one).squeeze()[0, :]
    data = data.transpose(1, 0)
    data, k_vector, deltas = serialize_experiment(data, times)
    return data, deltas, k_vector


SEED = 999
N = 150
rng = numpy.random.default_rng(seed=SEED)
REPETITION = 1000

ALPHA_TRUE = 1e-2
BETA_TRUE = 4e-1

SUBSAMPLE = 10
subsample_sequence = numpy.logspace(0.5, numpy.log10(N), int(N / SUBSAMPLE))
filename = "save_data/ef_schedule_blockreplicated_02.json"

GEN_DATA = True


if GEN_DATA:
    p_recall = 0.2
    deltas = [
        -numpy.log(p_recall) / (ALPHA_TRUE * (1 - BETA_TRUE) ** (k - 1))
        for k in list(range(10))
    ]
    print(deltas)
    times = numpy.cumsum(deltas)
    # times = [0, 200, 400, 600, 800, 2000, 2200, 2400, 86200, 86400]
    # times = numpy.linspace(0, 86400, 10)
    population_model = GaussianPopulation(
        ExponentialForgetting,
        mu=numpy.array([1e-2, 4e-1]),
        sigma=1e-6 * numpy.eye(2),
        population_size=int(N / len(times)),
        seed=None,
    )

    play_schedule = play_replicated_schedule
    play_schedule_args = (times,)

    optim_kwargs = {
        "method": "L-BFGS-B",
        "bounds": [(1e-5, 0.1), (0, 0.99)],
        "guess": (1e-3, 0.7),
        "verbose": False,
    }

    json_data, _ = gen_hessians(
        N,
        REPETITION,
        [ALPHA_TRUE, BETA_TRUE],
        population_model,
        play_schedule,
        subsample_sequence,
        play_schedule_args=play_schedule_args,
        optim_kwargs=optim_kwargs,
        filename=filename,
    )

else:
    with open(filename, "r") as _file:
        json_data = json.load(
            _file,
        )


recall_array = numpy.asarray(json_data["recall_array"])
observed_hessians = numpy.asarray(json_data["observed_hessians"])
observed_cum_hessians = numpy.asarray(json_data["observed_cum_hessians"])
estimated_parameters = numpy.asarray(json_data["estimated_parameters"])
recall_array = recall_array.transpose(1, 0)

recall_kwargs = {
    "x_bins": 20,
}

observed_information_kwargs = {"x_bins": 20, "cumulative": True, "cum_color": "orange"}

# fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
# (mean_observed_information, information, cum_inf, _) = compute_observed_information(
#     observed_hessians,
#     axs=axs,
#     observed_information_kwargs=observed_information_kwargs,
# )
# plt.tight_layout(w_pad=-2, h_pad=0)
# plt.savefig("images/information_block_repet_02.pdf")
# plt.show()
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
(fischer_information, agg_data, inf, cum_inf) = compute_full_observed_information(
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
# axs[0, 0].set_xscale("log")
# axs[0, 0].tick_params(axis="x", which="minor", bottom=False)
# axs[0, 0].set_xticks(
#     ticks=[int(ss) for ss in subsample_sequence],
#     labels=[str(int(ss)) for ss in subsample_sequence],
# )


# axs[0, 1].set_xscale("log")
# axs[0, 1].set_xticks(
#     ticks=[int(ss) for ss in subsample_sequence],
#     labels=[str(int(ss)) for ss in subsample_sequence],
# )

# plt.tight_layout(w_pad=2, h_pad=2)
# plt.get_current_fig_manager().full_screen_toggle()
plt.savefig("images/sup_replicated_02.pdf")
plt.show()
