import os
import json
import numpy
from pyrbit.information import compute_observed_information
from tqdm import tqdm


def dir_plot(dirname):
    filenames = os.listdir(dirname)
    filenames = [filename for filename in filenames if "ef_schedule_iid_" in filename]

    p = []
    max_inf = []
    for filename in tqdm(filenames):
        with open("/".join([dirname, filename]), "r") as _file:
            content = json.load(_file)
            observed_hessians = numpy.asarray(content["observed_hessians"])
            mean_observed_information, information, cum_inf = (
                compute_observed_information(observed_hessians)
            )
            p.append(float(filename.split(".")[0].split("_")[-1]) / 10)
            max_inf.append(cum_inf[-1])

    p, max_inf = [numpy.array(t) for t in zip(*sorted(zip(p, max_inf)))]
    return p, max_inf


import matplotlib.pyplot as plt

plt.style.use(style="fivethirtyeight")
import seaborn

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
x, y = dir_plot("save_data/iid_1e-2_4e-1")
seaborn.scatterplot(
    x=x,
    y=y,
    s=80,
    ax=ax,
    label=r"$\alpha=1e^{-2}$",
)
x, y = dir_plot("save_data/iid_1e-3_4e-1")
seaborn.scatterplot(
    x=x,
    y=y,
    s=80,
    ax=ax,
    label=r"$\alpha=1e^{-3}$",
)
color_cycle = ax._get_lines.prop_cycler
next(color_cycle)
next(color_cycle)
color = next(color_cycle)["color"]
ax.plot([0, 1], [0, 0], "o", color=color, label="theoretical values")
ax.legend()
ax.set_xlabel("probability of recall")
ax.set_ylabel("Sequence Fisher information")
plt.tight_layout()
plt.show()
