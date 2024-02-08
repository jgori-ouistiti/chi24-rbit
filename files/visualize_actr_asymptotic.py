import json
import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

plt.style.use(style="fivethirtyeight")


D_TRUE_VALUE = 0.4
S_TRUE_VALUE = 0.1
TAU_TRUE_VALUE = -0.5

TRUE_VALUE = numpy.array([D_TRUE_VALUE, S_TRUE_VALUE, TAU_TRUE_VALUE])

with open("save_data/actr_asymptotic.json", "r") as _file:
    results = json.load(_file)
    results = {key: numpy.array(value) for key, value in results.items()}

n = len(results)
agg_data = numpy.zeros((3 * n, 7))

for n, (key, value) in enumerate(results.items()):
    agg_data[n * 3 : (n + 1) * 3, 0] = (
        numpy.nanmean(value[:, 0, :], axis=1) - TRUE_VALUE
    )
    agg_data[n * 3 : (n + 1) * 3, 1] = numpy.abs(
        numpy.nanmean(value[:, 0, :], axis=1) - TRUE_VALUE
    )
    agg_data[n * 3 : (n + 1) * 3, 2] = numpy.nanstd(value[:, 0, :], axis=1)
    agg_data[n * 3 : (n + 1) * 3, 3] = numpy.nanmean(value[:, 1, :], axis=1)
    agg_data[n * 3 : (n + 1) * 3, 4] = numpy.nanmean(value[:, 2, :], axis=1)
    agg_data[n * 3 : (n + 1) * 3, 5] = int(float(key))
    agg_data[n * 3 : (n + 1) * 3, 6] = [0, 1, 2]


df = pandas.DataFrame(
    agg_data,
    columns=[
        "Bias",
        "|Bias|",
        "Std dev",
        "Coverage H",
        "Coverage BFGS",
        "N",
        "parameter",
    ],
)
mapping = {"0": r"$d$", "1": r"$s$", "2": r"$\tau$"}
df["parameter"] = df["parameter"].map(lambda s: mapping.get(str(int(s))))


fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (15,5))

ax = seaborn.barplot(
    data=df, x="N", y="|Bias|", hue="parameter", alpha=0.8, ax=axs[0]
)
ax.set_yscale("log")


ax = seaborn.barplot(
    data=df, x="N", y="Std dev", hue="parameter", alpha=0.8, ax=axs[1]
)
df = df.melt(id_vars=["parameter", "N"], value_vars=["Coverage H", "Coverage BFGS"])
ax.set_yscale("log")

ax = seaborn.lineplot(
    data=df,
    x="N",
    y="value",
    hue="parameter",
    style="variable",
    alpha=0.8,
    ax=axs[2],
)
ax.plot([10, 1e3], [0.95, 0.95], "-r", lw=2, label="Nominal coverage")
ax.set_xscale("log")
ax.set_ylabel("Coverage")
ax.legend()
plt.tight_layout(pad=0.2)
plt.show()
