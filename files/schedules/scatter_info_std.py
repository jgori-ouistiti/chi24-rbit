import json
import os
from tqdm import tqdm
import numpy
import seaborn
import matplotlib.pyplot as plt

plt.style.use(style="fivethirtyeight")


from pyrbit.mle_utils import compute_summary_statistics_estimation
from pyrbit.information import compute_observed_information

main_data_dir = "save_data"
docs = os.listdir(main_data_dir)

files = []
for doc in docs:
    path = "/".join([main_data_dir, doc])
    if not os.path.isdir(path):
        files.append(path)
    else:
        prefix = doc
        for _doc in os.listdir(path):
            files.append("/".join([main_data_dir, prefix, _doc]))


alpha_std_container = []
information_container = []
beta_std_container = []
hue_container = []
for filename in tqdm(files):
    with open(filename, "r") as _file:
        content = json.load(_file)
        observed_hessians = numpy.asarray(content["observed_hessians"])
        recall_array = numpy.asarray(content["recall_array"])
        estimated_parameters = numpy.asarray(content["estimated_parameters"])

        N = observed_hessians.shape[1]
        SUBSAMPLE = int(N / estimated_parameters.shape[1])
        subsample_sequence = numpy.logspace(0.5, numpy.log10(N), int(N / SUBSAMPLE))

        mean_observed_information, information, cum_inf = compute_observed_information(
            observed_hessians
        )

        if "1e-3" in filename:
            TRUE_VALUE = [1e-3, 0.4]
            _hue = r"$\alpha=1e^{-3}$"
        else:
            TRUE_VALUE = [1e-2, 0.4]
            _hue = r"$\alpha=1e^{-2}$"

        agg_data, df = compute_summary_statistics_estimation(
            estimated_parameters,
            subsample_sequence,
            TRUE_VALUE,
            ax=None,
            bias_kwargs=None,
            std_kwargs=None,
        )
        k = estimated_parameters.shape[1]
        n = estimated_parameters.shape[0]
        alpha_std_container.extend(agg_data[:k, 1].tolist())
        beta_std_container.extend(agg_data[k : n * k, 1].tolist())
        subsample_sequence = [int(m - 1) for m in agg_data[:k, 2]]
        hue_container.extend(numpy.full((k,), fill_value=_hue).tolist())
        information_container.extend(cum_inf[numpy.array(subsample_sequence)])


fig, axs = plt.subplots(nrows=2, ncols=1)
axs[0].set_xlabel("Sequence Fisher Information")
axs[0].set_ylabel(r"$\sigma$")
axs[1].set_xlabel("Sequence Fisher Information")
axs[1].set_ylabel(r"$\sigma$")
axs[0].set_title(r"$\sigma(\alpha)$")
axs[1].set_title(r"$\sigma(\beta)$")

seaborn.scatterplot(
    x=information_container, y=alpha_std_container, hue=hue_container, ax=axs[0]
)
seaborn.scatterplot(
    x=information_container, y=beta_std_container, hue=hue_container, ax=axs[1]
)
plt.tight_layout()
plt.show()
