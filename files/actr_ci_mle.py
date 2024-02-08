from pyrbit.actr import (
    ACTR,
    identify_actr_from_recall_sequence,
    actr_observed_information_matrix,
    gen_data,
)
from pyrbit.mle_utils import CI_asymptotical
import numpy
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

SEED = 789
TEST = True

if TEST:
    N_logspace = 2
    REPETITIONS = 50
    Nmax = 2
    _str = "_test"
else:
    N_logspace = 3
    REPETITIONS = 500
    Nmax = 3
    _str = ""

rng = numpy.random.default_rng(seed=SEED)


D_TRUE_VALUE = 0.4
S_TRUE_VALUE = 0.1
TAU_TRUE_VALUE = -0.5
actr = ACTR(1, d=D_TRUE_VALUE, s=S_TRUE_VALUE, tau=TAU_TRUE_VALUE, seed=SEED)


# optimizer --- same options for any N
# optim_kwargs = {"method": "L-BFGS-B", "bounds": [(0, 1), (-5, 5), (-5, 5)]}
optim_kwargs = {"method": "BFGS"}

verbose = False
# d, s, tau
guess = (0.5, 0.25, -0.7)


results = {}
for _N in numpy.logspace(1, Nmax, N_logspace):
    N = int(numpy.round(_N))
    result = numpy.zeros((3, 3, REPETITIONS))
    for i in tqdm(range(REPETITIONS)):
        recalls, deltatis = gen_data(actr, N, seed=SEED)
        inference_results = identify_actr_from_recall_sequence(
            recalls,
            deltatis,
            optim_kwargs=optim_kwargs,
            verbose=verbose,
            guess=guess,
            basin_hopping=True,
            basin_hopping_kwargs={"niter": 3},
        )

        if (
            inference_results.lowest_optimization_result.x[0] <= 0
            or inference_results.lowest_optimization_result.x[0] >= 1
            or inference_results.lowest_optimization_result.x[1] <= -5
            or inference_results.lowest_optimization_result.x[1] >= 5
            or inference_results.lowest_optimization_result.x[2] <= -5
            or inference_results.lowest_optimization_result.x[2] >= 5
        ):
            result[..., i] = numpy.nan
            continue

        J = actr_observed_information_matrix(
            recalls, deltatis, *inference_results.lowest_optimization_result.x
        )
        try:
            covar = numpy.linalg.inv(J)
        except numpy.linalg.LinAlgError:
            result[..., i] = numpy.nan
            result[:, 0, i] = inference_results.lowest_optimization_result.x
            continue
        hess_inv = inference_results.lowest_optimization_result.hess_inv
        # test coverage and plot
        cis = CI_asymptotical(
            covar,
            inference_results.lowest_optimization_result.x,
            critical_value=1.96,
        )
        cis_hess_inv = CI_asymptotical(
            hess_inv,
            inference_results.lowest_optimization_result.x,
            critical_value=1.96,
        )

        result[:, 0, i] = inference_results.lowest_optimization_result.x
        TRUE_VALUES = [D_TRUE_VALUE, S_TRUE_VALUE, TAU_TRUE_VALUE]
        for n in range(3):
            if cis[n][0] <= TRUE_VALUES[n] and TRUE_VALUES[n] <= cis[n][1]:
                result[n, 1, i] = True
            else:
                result[n, 1, i] = False
            if cis_hess_inv[n][0] <= TRUE_VALUES[n] and TRUE_VALUES[n] <= cis[n][1]:
                result[n, 2, i] = True
            else:
                result[n, 2, i] = False

    results[str(N)] = result.tolist()
with open(f"save_data/actr_asymptotic{_str}.json", "w") as _file:
    json.dump(results, _file)
    results = {key: numpy.array(value) for key, value in results.items()}
