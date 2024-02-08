import argparse
import numpy
import matplotlib.pyplot as plt

plt.style.use("/home/juliengori/Documents/VC/mplstyles/seaborn-small-fig.mplstyle")

from pyrbit.ef import ExponentialForgetting
from pyrbit.information import gen_hessians, compute_observed_information

# Instantiate the parser
parser = argparse.ArgumentParser(
    description="Evaluate Fischer Information for iid schedule as a function of recall probability"
)
parser.add_argument("p_recall", type=float, help="The target probability of recall")
parser.add_argument(
    "N", type=int, nargs="?", help="The length of the training sequence"
)
parser.add_argument(
    "--alpha", type=float, help="Alpha parameter of the Exponential Forgetting model"
)
parser.add_argument(
    "--beta", type=int, help="Beta parameter of the Exponential Forgetting model"
)

args = parser.parse_args()
p_recall = args.p_recall
N = args.N
if N is None:
    N = 250
alpha = args.alpha
if alpha is None:
    alpha = 1e-3
beta = args.beta
if beta is None:
    beta = 0.4


def play_iid_schedule(ef, p_recall, N, alpha=1e-2, beta=0.4):
    k_vector = rng.integers(low=0, high=10, size=N)
    deltas = [-numpy.log(p_recall) / (alpha * (1 - beta) ** (k - 1)) for k in k_vector]
    recall_probs = simulate_arbitrary_traj(ef, k_vector, deltas)
    recall = [rp[0] for rp in recall_probs]
    k_repetition = [k - 1 for k in k_vector]
    return recall, deltas, k_repetition


def simulate_arbitrary_traj(ef, k_vector, deltas):
    recall = []
    for k, d in zip(k_vector, deltas):
        ef.update(0, 0, N=(k))  #### presentations and not repetitions
        recall.append(ef.query_item(0, d))
    return recall


# def play_iid_schedule(ef, p_recall, N, alpha=1e-2, beta=0.4):
#     k_vector = rng.integers(low=-1, high=10, size=N)
#     deltas = [-numpy.log(p_recall) / (alpha * (1 - beta) ** k) for k in k_vector]
#     recall_probs = simulate_arbitrary_traj(ef, k_vector, deltas)
#     recall = [rp[0] for rp in recall_probs]
#     k_repetition = [k for k in k_vector]
#     return recall, deltas, k_repetition


# def simulate_arbitrary_traj(ef, k_vector, deltas):
#     recall = []
#     for k, d in zip(k_vector, deltas):
#         ef.update(0, 0, N=(k + 1))  #### presentations and not repetitions
#         recall.append(ef.query_item(0, d))
#     return recall


SEED = 456
rng = numpy.random.default_rng(seed=SEED)
REPETITION = 1000

ALPHA_TRUE = alpha
BETA_TRUE = beta


SUBSAMPLE = 10
subsample_sequence = numpy.logspace(0.5, numpy.log10(N), int(N / SUBSAMPLE))


ef = ExponentialForgetting(1, ALPHA_TRUE, BETA_TRUE, seed=SEED)
play_schedule = play_iid_schedule
play_schedule_args = (p_recall, N)

optim_kwargs = {
    "method": "L-BFGS-B",
    "bounds": [(1e-5, 0.1), (0, 0.99)],
    "guess": (1e-3, 0.7),
    "verbose": False,
    "basin_hopping": True,
}
filename = f"save_data/ef_schedule_iid_{int(p_recall*10)}.json"
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
recall_array = numpy.asarray(json_data["recall_array"])
observed_hessians = numpy.asarray(json_data["observed_hessians"])
observed_score = numpy.asarray(json_data["observed_score"])
estimated_parameters = numpy.asarray(json_data["estimated_parameters"])
recall_array = recall_array.transpose(1, 0)

recall_kwargs = {
    "x_bins": 10,
}
observed_information_kwargs = {"x_bins": 10}

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
fischer_information, agg_data, inf, cum_inf = compute_observed_information(
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

plt.savefig(f"images/recall/{p_recall}.pdf".replace("0.", "0_"))
plt.close()
