import json
import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

ALPHA_TRUE = 1e-2
BETA_TRUE = 0.4

TRUE_VALUE = numpy.array([ALPHA_TRUE, BETA_TRUE])

plt.style.use(style="fivethirtyeight")

# with open("save_data/ef_asymptotic.json", mode="r") as _file:
with open("save_data/ci_with_log.json", mode="r") as _file:

    results = json.load(_file)
    results = {key: numpy.array(value) for key, value in results.items()}

n = len(results)
agg_data = numpy.zeros((2 * n, 8))

for n, (key, value) in enumerate(results.items()):
    agg_data[n * 2 : (n + 1) * 2, 0] = (
        numpy.nanmean(value[:, 0, :], axis=1) - TRUE_VALUE
    )
    agg_data[n * 2 : (n + 1) * 2, 1] = numpy.abs(
        numpy.nanmean(value[:, 0, :], axis=1) - TRUE_VALUE
    )
    agg_data[n * 2 : (n + 1) * 2, 2] = numpy.nanstd(value[:, 0, :], axis=1)
    agg_data[n * 2 : (n + 1) * 2, 3] = numpy.nanmean(value[:, 1, :], axis=1)
    agg_data[n * 2 : (n + 1) * 2, 4] = numpy.nanmean(value[:, 2, :], axis=1)
    agg_data[n * 2 : (n + 1) * 2, 5] = numpy.nanmean(value[:, 3, :], axis=1)
    agg_data[n * 2 : (n + 1) * 2, 6] = int(float(key))
    agg_data[n * 2 : (n + 1) * 2, 7] = [0, 1]


df = pandas.DataFrame(
    agg_data,
    columns=[
        "Bias",
        "|Bias|",
        "Std dev",
        "Coverage H",
        "Coverage BFGS",
        "Coverage log-H",
        "N",
        "parameter",
    ],
)
mapping = {"0": r"$\alpha$", "1": r"$\beta$"}
df["parameter"] = df["parameter"].map(lambda s: mapping.get(str(int(s))))


fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (15,5))
ax = seaborn.barplot(
    data=df, x="N", y="|Bias|", hue="parameter", alpha=0.8, ax=axs[0]
)
ax.set_yscale("log")


ax = seaborn.barplot(
    data=df, x="N", y="Std dev", hue="parameter", alpha=0.8, ax=axs[1]
)
ax.set_yscale("log")

df = df.melt(
    id_vars=["parameter", "N"],
    value_vars=["Coverage H", "Coverage BFGS", "Coverage log-H"],
)
ax = seaborn.lineplot(
    data=df,
    x="N",
    y="value",
    hue="parameter",
    style="variable",
    alpha=0.8,
    ax=axs[2],
)
ax.plot([10, 1e4], [0.95, 0.95], "-r", lw=2, label="Nominal coverage")
ax.set_xscale("log")
ax.set_ylabel("Coverage")
ax.set_ylim([0.5,1.05])
ax.legend(loc='lower center')
plt.tight_layout(pad=0.2)
plt.show()
