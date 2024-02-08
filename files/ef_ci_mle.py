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
TEST = True

if TEST:
    N_logspace = 2
    REPETITIONS = 50
    Nmax = 2
    _str = "_test"
else:
    N_logspace = 4
    REPETITIONS = 1000
    Nmax = 4
    _str = ""


def simulate_arbitrary_traj(ef, k_vector, deltas):
    recall = []
    for k, d in zip(k_vector, deltas):
        ef.update(0, 0, N=k)
        recall.append(ef.query_item(0, d))
    return recall


ALPHA_TRUE = 1e-2
BETA_TRUE = 0.4
ef = ExponentialForgetting(1, ALPHA_TRUE, BETA_TRUE, seed=SEED)
rng = numpy.random.default_rng(seed=SEED)
################

results = {}
for _N in numpy.logspace(1, Nmax, N_logspace):
    N = int(numpy.round(_N))
    print(N)
    result = numpy.zeros((2, 4, REPETITIONS))
    for i in tqdm(range(REPETITIONS)):
        k_vector = rng.integers(low=0, high=10, size=N)
        deltas = rng.integers(low=0, high=5000, size=N)
        recall_probs = simulate_arbitrary_traj(ef, k_vector, deltas)
        recall = [rp[0] for rp in recall_probs]
        optim_kwargs = {"method": "BFGS"}

        verbose = False
        guess = (1e-3, 0.5)
        inference_results = identify_ef_from_recall_sequence(
            recall,
            deltas,
            k_vector=[k - 1 for k in k_vector],
            optim_kwargs=optim_kwargs,
            verbose=verbose,
            guess=guess,
        )
        # reject solution if not within bounds. Normally using L-BFGS-B but here BFGS just for the hess_inv computation to compare.
        if (
            inference_results.x[0] <= 1e-5
            or inference_results.x[0] >= 0.1
            or inference_results.x[1] <= 0
            or inference_results.x[1] >= 1
        ):
            result[..., i] = numpy.nan
            continue

        J = ef_observed_information_matrix(
            recall, deltas, *inference_results.x, k_vector=[k - 1 for k in k_vector]
        )
        try:
            covar = numpy.linalg.inv(J)
        except numpy.linalg.LinAlgError:
            result[..., i] = numpy.nan
            result[:, 0, i] = inference_results.x
            continue
        hess_inv = inference_results.hess_inv
        # test coverage and plot
        cis = CI_asymptotical(covar, inference_results.x, critical_value=1.96)
        cis_hess_inv = CI_asymptotical(
            hess_inv, inference_results.x, critical_value=1.96
        )

        transformed_covar = covar_delta_method_log_alpha(inference_results.x[0], covar)
        x = [numpy.log10(inference_results.x[0]), inference_results.x[1]]
        cis_log = CI_asymptotical(transformed_covar, x, critical_value=1.96)

        result[:, 0, i] = inference_results.x
        TRUE_VALUES = [ALPHA_TRUE, BETA_TRUE]
        TRANSFORM_TRUE_VALUES = [numpy.log10(ALPHA_TRUE), BETA_TRUE]
        for n in range(2):
            if cis[n][0] <= TRUE_VALUES[n] and TRUE_VALUES[n] <= cis[n][1]:
                result[n, 1, i] = True
            else:
                result[n, 1, i] = False
            if cis_hess_inv[n][0] <= TRUE_VALUES[n] and TRUE_VALUES[n] <= cis[n][1]:
                result[n, 2, i] = True
            else:
                result[n, 2, i] = False
            if (
                cis_log[n][0] <= TRANSFORM_TRUE_VALUES[n]
                and TRANSFORM_TRUE_VALUES[n] <= cis_log[n][1]
            ):
                result[n, 3, i] = True
            else:
                result[n, 3, i] = False

    results[str(N)] = result
with open(f"save_data/ef_asymptotic{_str}.json", "w") as _file:
    results = {key: value.tolist() for key, value in results.items()}
    json.dump(results, _file)
results = {key: numpy.array(value) for key, value in results.items()}
