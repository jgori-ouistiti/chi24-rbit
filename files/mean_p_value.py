from pyrbit.mem_utils import (
    GaussianPopulation,
    BlockBasedSchedule,
)
from pyrbit.ef import (
    ExponentialForgetting,
)
from pyrbit.design import get_p_values_frequency


import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


if __name__ == "__main__":

    plt.style.use(style="fivethirtyeight")

    GEN = True

    # Shared parameters
    BETA = 0.4
    SIGMA = 1e-7 * numpy.array(
        [[1, 0], [0, 0.1]]
    )  # same magnitude variation for both parameters
    repet_trials = 1
    nitems = 1
    pop_size = 100
    replications = 1
    RECALL_BLOCKS = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    ALPHA_A = 10 ** (-2.1)
    ALPHA_C = 10 ** (-1.9)
    ALPHA = 10 ** (-2)
    N = 100
    REPET = 100
    SEED = 123

    optim_kwargs = {
        "method": "L-BFGS-B",
        "bounds": [(1e-5, 0.1), (0, 0.99)],
        "guess": (1e-3, 0.7),
        "verbose": False,
    }

    p_value_container = numpy.zeros((REPET, 9, 2))
    intertrial_time = 0

    if GEN:
        for ni, i in enumerate(tqdm(range(1, 10))):
            interblock_time = [
                -numpy.log(i / 10) / (ALPHA * (1 - BETA) ** (k)) for k in range(9)
            ]

            schedule = BlockBasedSchedule(
                1,
                15,
                interblock_time,
                repet_trials=1,
                seed=123,
                sigma_t=None,
            )

            population_model_one = [
                GaussianPopulation(
                    ExponentialForgetting,
                    mu=[ALPHA_A, BETA],
                    sigma=1e-7 * numpy.array([[0.01, 0], [0, 1]]),
                    population_size=24 * 2,
                    n_items=1,
                    seed=SEED + i,
                )
                for i in range(REPET)
            ]

            population_model_two = [
                GaussianPopulation(
                    ExponentialForgetting,
                    mu=[ALPHA_C, BETA],
                    sigma=1e-7 * numpy.array([[0.01, 0], [0, 1]]),
                    population_size=24 * 2,
                    n_items=1,
                    seed=SEED + REPET + i,
                )
                for i in range(REPET)
            ]

            pos_two, neg_two, p_container = get_p_values_frequency(
                population_model_one,
                population_model_two,
                schedule,
                combine_pvalues=["stouffer", "fisher"],
                test_blocks=None,
                significance_level=0.05,
            )
            p_value_container[:, ni, :] = p_container

        with open("data.pkl", "wb") as _file:
            pickle.dump(p_value_container, _file)

    else:
        with open("data.pkl", "rb") as _file:
            p_value_container = pickle.load(_file)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    labels = [
        "p=0.1",
        "p=0.2",
        "p=0.3",
        "p=0.4",
        "p=0.5",
        "p=0.6",
        "p=0.7",
        "p=0.8",
        "p=0.9",
    ]
    ps = [i / 10 for i in list(range(1, 10))]
    p_values = numpy.nanmean(p_value_container, axis=0)
    axs.plot(ps, p_values[:, 0], "o", label="fisher p-values")
    axs.plot(ps, p_values[:, 1], "D", label="stouffer p-values")
    axs.set_yscale("log")
    axs.legend()
    axs.set_xlabel("schedule aimed probability")
    axs.set_ylabel("combined p-values")
    plt.tight_layout(w_pad=-2, h_pad=0)
    plt.show()
