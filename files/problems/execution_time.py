from pyrbit.mem_utils import experiment, GaussianPopulation
from pyrbit.ef import (
    ExponentialForgetting,
    diagnostics,
    identify_ef_from_recall_sequence,
    ef_observed_information_matrix,
    covar_delta_method_log_alpha,
)
from pyrbit.mle_utils import confidence_ellipse

import numpy
from pyrbit.mem_utils import BlockBasedSchedule
import matplotlib.pyplot as plt
import seaborn
import numpy
import pandas
import scipy
from matplotlib import ticker


def simulator_block_ef(
    ALPHA,
    BETA,
    SIGMA,
    repet_trials,
    nitems,
    pop_size,
    replications,
    intertrial_time,
    interblock_time,
):
    if repet_trials != 1:
        raise NotImplementedError

    default_population_kwargs = {
        "mu": [ALPHA, BETA],
        "sigma": SIGMA,
        "seed": None,
        "n_items": nitems,
        "population_size": pop_size,
    }
    schedule = BlockBasedSchedule(
        nitems, intertrial_time, interblock_time, repet_trials=repet_trials
    )

    population_model = GaussianPopulation(
        ExponentialForgetting,
        **default_population_kwargs,
    )
    data = experiment(population_model, schedule, replications=replications)
    nblock = len(schedule.interblock_time) + 1

    data = (
        data[0, 0, ...]
        .transpose(1, 0)
        .reshape(
            population_model.pop_size,
            nblock,
            schedule.nitems * schedule.repet_trials,
        )
    )

    times = numpy.array(schedule.times).reshape(-1, repet_trials * nitems)
    deltas = numpy.zeros(times.shape)
    deltas[1:, :] = numpy.diff(times, axis=0)
    deltas[0, :] = numpy.inf
    blocks = numpy.array(schedule.blocks).reshape(-1, repet_trials * nitems)
    deltas_full = numpy.repeat(deltas[numpy.newaxis, :, :], pop_size, axis=0)
    blocks_full = numpy.repeat(blocks[numpy.newaxis, :, :], pop_size, axis=0)
    k_repetition = blocks_full - 1

    return data, deltas_full, k_repetition


def diff_eval(
    ALPHA, intertrial_time_A, intetrial_timeB, interblock_time, condition_names, *args
):
    data_C, deltas_C, k_repetition_C = simulator_block_ef(
        ALPHA, *args, intertrial_time_A, interblock_time
    )
    block_average_mean_C = numpy.mean(data_C, axis=2)

    data_A, deltas_A, k_repetition_A = simulator_block_ef(
        ALPHA, *args, intetrial_timeB, interblock_time
    )
    block_average_mean_A = numpy.mean(data_A, axis=2)

    identifier = numpy.tile(numpy.array(range(9)), (block_average_mean_C.shape[0], 1))

    mean_C_recall = block_average_mean_C[:, RECALL_BLOCKS]
    identifier_C = identifier[:, RECALL_BLOCKS]
    mean_A_recall = block_average_mean_A[:, RECALL_BLOCKS]
    identifier_A = identifier[:, RECALL_BLOCKS]
    df = pandas.DataFrame(
        {
            "block recall %": numpy.concatenate(
                (mean_C_recall.ravel(), mean_A_recall.ravel()), axis=0
            ),
            "block": numpy.concatenate(
                (identifier_C.ravel(), identifier_A.ravel()), axis=0
            ),
            "Condition": numpy.concatenate(
                (
                    numpy.full(mean_C_recall.ravel().shape, condition_names[0]),
                    numpy.full(mean_A_recall.ravel().shape, condition_names[1]),
                ),
                axis=0,
            ),
        }
    )
    return (
        df,
        data_C,
        deltas_C,
        k_repetition_C,
        data_A,
        deltas_A,
        k_repetition_A,
        mean_C_recall,
        mean_A_recall,
    )


def ML_plot_CE(data, deltas, krepet, ax, colors=["#B0E0E6", "#87CEEB"], **kwargs):
    recall, deltas, krepet = data.ravel(), deltas.ravel(), krepet.ravel()
    optim_kwargs = {"method": "L-BFGS-B", "bounds": [(1e-5, 0.1), (0, 0.99)]}
    verbose = False
    guess = (1e-3, 0.8)  # start with credible guess

    inference_results = identify_ef_from_recall_sequence(
        recall_sequence=recall,
        deltas=deltas,
        k_vector=krepet,
        optim_kwargs=optim_kwargs,
        verbose=verbose,
        guess=guess,
    )
    J = ef_observed_information_matrix(
        recall, deltas, *inference_results.x, k_vector=krepet
    )
    covar = numpy.linalg.inv(J)
    transformed_covar = covar_delta_method_log_alpha(inference_results.x[0], covar)
    x = [numpy.log10(inference_results.x[0]), inference_results.x[1]]
    ax_log = confidence_ellipse(x, transformed_covar, ax=ax, colors=colors, **kwargs)
    ax_log.set_title("CE with alpha log scale")
    return ax, inference_results, transformed_covar


#############


if __name__ == "__main__":

    # Shared parameters
    BETA = 0.5
    SIGMA = 1e-7 * numpy.array(
        [[1, 0], [0, 0.1]]
    )  # same magnitude variation for both parameters
    repet_trials = 1
    nitems = 50
    pop_size = 24
    replications = 1
    RECALL_BLOCKS = numpy.array([1, 3, 5])
    ALPHA = 10 ** (-2.5)
    interblock_time = [10, 200, 200, 200, 200, 100000, 200, 200]  # À la BodyLoci

    ## === A==B but different intertrial times

    intertrial_time_A = 10
    intertrial_time_B = 5
    (
        df1,
        data_A,
        deltas_A,
        k_repetition_A,
        data_B,
        deltas_B,
        k_repetition_B,
        mean_A_recall,
        mean_B_recall,
    ) = diff_eval(
        ALPHA,
        intertrial_time_A,
        intertrial_time_B,
        interblock_time,
        ["A", "B"],
        BETA,
        SIGMA,
        repet_trials,
        nitems,
        pop_size,
        replications,
    )

    ## ==== classical comparison
    plt.style.use(style="fivethirtyeight")
    pvalues = scipy.stats.ttest_rel(mean_A_recall, mean_B_recall, axis=0).pvalue
    print(pvalues)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    seaborn.barplot(
        data=df1,
        x="block",
        y="block recall %",
        hue="Condition",
        errorbar="se",
        ax=axs[0],
    )

    ## ==== Model-based comparison
    # A
    recall, deltas, krepet = (
        data_A[:, RECALL_BLOCKS, :],
        deltas_A[:, RECALL_BLOCKS, :],
        k_repetition_A[:, RECALL_BLOCKS, :],
    )
    ax, inf, covar = ML_plot_CE(
        recall,
        deltas,
        krepet,
        axs[1],
        colors=["#ff99bb", "#ff1a66"],
        plot_kwargs={"color": "#ff1a66", "marker": "D", "label": "A"},
    )
    # B2
    recall, deltas, krepet = (
        data_B[:, RECALL_BLOCKS, :],
        deltas_B[:, RECALL_BLOCKS, :],
        k_repetition_B[:, RECALL_BLOCKS, :],
    )
    ax, inf, covar = ML_plot_CE(
        recall,
        deltas,
        krepet,
        axs[1],
        colors=["#ffd699", "#ffa31a"],
        plot_kwargs={"color": "#ffa31a", "marker": "D", "label": "B"},
    )

    axs[1].set_xlabel(r"$\log \hat{\alpha}$")
    axs[1].set_ylabel(r"$\hat{\beta}$")
    axs[1].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    axs[0].set_title("Simulated Experiment: A vs B")
    axs[1].set_title("ML estimates for A and B")
    axs[1].legend()
    plt.tight_layout()
    plt.show()
