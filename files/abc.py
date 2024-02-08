from pyrbit.abc import ef_simulator, ef_infer_abc, plot_ihd_contours
import numpy
import pickle

import warnings

warnings.filterwarnings("ignore")


simulation_kwargs = {
    "pop_size": 15 * 24,
    "nitems": 1,
    "seed": None,
    "repet_trials": 1,
    "intertrial_time": 10,
    "interblock_time": [0, 300, 300, 300, 300, 80000, 300, 300],
    "test_blocks": [1, 3, 5, 6, 8],
    "replications": 1,
    "sigma_t": None,
    "alpha": 10 ** (-2.5),
    "beta": 0.77,
    "sigma": 1e-8 * numpy.eye(2),
}

mean, se, data = ef_simulator(**simulation_kwargs)


def sim_ef_simulator(rng, a, b, size=None):
    simulation_kwargs = {
        "pop_size": 15 * 24,
        "nitems": 1,
        "seed": None,
        "repet_trials": 1,
        "intertrial_time": 10,
        "interblock_time": [0, 300, 300, 300, 300, 80000, 300, 300],
        "test_blocks": [1, 3, 5, 6, 8],
        "replications": 1,
        "sigma_t": None,
        "alpha": 10 ** (a),
        "beta": b,
        "sigma": 1e-8 * numpy.eye(2),
    }
    return ef_simulator(**simulation_kwargs)[0]


simulator_kwargs = {"epsilon": 10}

# observed_data = numpy.array([0.30, 0.60, 0.73, 0.53, 0.77])

observed_data = mean
GEN_DATA = False
if GEN_DATA:
    idata = ef_infer_abc(
        observed_data, sim_ef_simulator, simulator_kwargs=simulator_kwargs
    )

    with open("save_data/abc_10.pkl", "wb") as _file:
        pickle.dump(idata, _file)
else:
    with open("save_data/abc_10.pkl", "rb") as _file:
        idata = pickle.load(_file)


import arviz as az
import matplotlib.pyplot as plt


plt.style.use(style="fivethirtyeight")
print(az.summary(idata, kind="stats"))

plt.ion()

ax = az.plot_forest(
    idata,
    var_names=["log10alpha", "b"],
    combined=True,
    hdi_prob=0.95,
    textsize=16,
)[0]
ax.set_xlabel("Parameter value")
ax.figure.tight_layout()
exit()
ax = plot_ihd_contours(idata)
ax.set_xlabel(r"$\log_{10}\alpha$")
ax.set_ylabel(r"$\beta$")
plt.tight_layout()
plt.show()
