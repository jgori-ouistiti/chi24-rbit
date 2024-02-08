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


def play_linear_schedule(population_model, N):
    times = numpy.linspace(0, 20 * 86400, N)
    items = [0 for i in times]
    schedule_one = Schedule(items, times)
    recall = experiment(population_model, schedule_one).squeeze()[0, :]
    deltas = [numpy.infty] + numpy.diff(times).tolist()
    k_repetition = [-1 + i for i in range(N)]
    return recall, deltas, k_repetition


SEED = 456
N = 100
rng = numpy.random.default_rng(seed=SEED)
REPETITION = 1000

ALPHA_TRUE = 1e-2
BETA_TRUE = 4e-1

SUBSAMPLE = 15
subsample_sequence = numpy.logspace(0.5, numpy.log10(N), int(N / SUBSAMPLE))
GEN_DATA = True


if GEN_DATA:

    population_model = GaussianPopulation(
        ExponentialForgetting,
        mu=numpy.array([1e-2, 4e-1]),
        sigma=1e-6 * numpy.eye(2),
        seed=None,
    )
    play_schedule = play_linear_schedule
    play_schedule_args = (N,)

    optim_kwargs = {
        "method": "L-BFGS-B",
        "bounds": [(1e-5, 0.1), (0, 0.99)],
        "guess": (1e-3, 0.7),
        "verbose": False,
    }
    filename = "save_data/ef_schedule_linear.json"

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
    with open("save_data/ef_schedule_linear.json", "r") as _file:
        json_data = json.load(
            _file,
        )


recall_array = numpy.asarray(json_data["recall_array"])
observed_hessians = numpy.asarray(json_data["observed_hessians"])
estimated_parameters = numpy.asarray(json_data["estimated_parameters"])
recall_array = recall_array.transpose(1, 0)

recall_kwargs = {
    "x_bins": 10,
}

observed_information_kwargs = {"x_bins": 10, "cum_color": "orange"}


fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

import seaborn

X = numpy.ones((recall_array.shape))
X = numpy.cumsum(X, axis=1)
seaborn.regplot(
    x=X.ravel(),
    y=recall_array.ravel(),
    scatter=True,
    fit_reg=False,
    ci=None,
    ax=axs,
    label="events",
)
default_recall_kwargs = {
    "fit_reg": False,
    "ci": None,
    "label": "estimated probabilities",
    "x_bins": 20,
}
seaborn.regplot(x=X.ravel(), y=recall_array.ravel(), ax=axs, **default_recall_kwargs)
# axs.set_ylim([-0.05, 1.05])
# axs.set_xlabel("N")
# axs.set_ylabel("Recalls")
# axs.legend()
# plt.tight_layout(w_pad=-2, h_pad=0)
# # plt.show()
# plt.savefig('images/recall_linear.pdf')


# fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
# (mean_observed_information, information, cum_inf, _) = compute_observed_information(
#     observed_hessians,
#     axs=axs,
#     observed_information_kwargs=observed_information_kwargs,
# )
# axs.legend(loc = 'center right')
# # fig.axes[1].legend(loc = 'center left')
# plt.tight_layout(w_pad=-2, h_pad=0)
# # plt.show()
# plt.savefig("images/information_linear.pdf")

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
fischer_information, agg_data, information, cum_inf = compute_full_observed_information(
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
axs[0, 0].set_xscale("log")
axs[0, 0].tick_params(axis="x", which="minor", bottom=False)
axs[0, 0].set_xticks(
    ticks=[int(ss) for ss in subsample_sequence],
    labels=[str(int(ss)) for ss in subsample_sequence],
)


axs[0, 1].set_xscale("log")
axs[0, 1].set_xticks(
    ticks=[int(ss) for ss in subsample_sequence],
    labels=[str(int(ss)) for ss in subsample_sequence],
)
axs[1, 1].set_ylim([1e-2, 1e-1])
plt.tight_layout(w_pad=2, h_pad=2)
# plt.get_current_fig_manager().full_screen_toggle()
plt.savefig("images/sup_linear_all.pdf")
# plt.show()
