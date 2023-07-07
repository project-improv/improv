import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_area(
    data,
    cmap,
    labels,
    title,
    xticks,
    out_path,
    file_name,
    context="poster",
    rc=None,
    save=False,
):
    """ """
    plt.stackplot(data, color=cmap)
    plt.xticks(xticks)
    plt.xlabel("Frames", fontname="Arial")
    plt.ylabel("Time (ms)", fontname="Arial")
    plt.title(title)

    plt.legend(loc="upper right", bbox_to_anchor=(1.04, 1))

    sns.set_context(context)

    plt.legend(labels)

    if rc is None:
        rc = {
            "font.size": 14,
            "font.family": "Arial",
            "axes.labelsize": 10,
            "legend.fontsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "savefig.format": "svg",
        }

    sns.set_style(
        "ticks",
        {
            "axes.edgecolor": "k",
            "axes.linewidth": 1,
            "axes.grid": False,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
        },
    )

    if save is True:
        plt.savefig(
            os.path.join(out_path, file_name) + ".svg",
            format="svg",
            bbox_inches="tight",
            dpi=300,
        )

    plt.close()


acq_path = "output/GPU/timing/acq_timing.csv"
proc_path = "output/GPU/timing/proc_timing.csv"

acq_df = pd.read_csv(acq_path)
proc_df = pd.read_csv(proc_path)

acq = acq_df["acq_total_times"][0:300]
# put_img_time, put_lab_time, put_out_time

preproc = ["get_img_out", "proc_img_time"]

analysis = [
    "to_device",
    "inference_time",
    "out_to_np",
    "put_out_store",
    "put_q_out",
    "pred_time",
    "put_pred_store",
]
# to_device,inference_time
# out_to_np,put_out_store,put_q_out
# pred_time,put_pred_store

preproc = proc_df[preproc][0:300]
analysis = proc_df[analysis][0:300]

preproc = preproc.sum(axis=1)
analysis = analysis.sum(axis=1)

acq.index = acq.index + 1
preproc.index = preproc.index + 1
analysis.index = analysis.index + 1

# ORANGE:
# e26e3d
# e57e52
# e88e68
# eb9e7d
# eeae93
# f2bea8
# f5cebe
# f8ded3
# fbeee9
# ffffff

# GREEN:
# 3de26e
# 52e57e
# 68e88e
# 7deb9e
# 93eeae
# a8f2be
# bef5ce
# d3f8de
# e9fbee
# ffffff

# PURPLE:
# 8e69ce
# 9a79d3
# a78ad8
# b39bde
# c0abe3
# ccbce9
# d9cdee
# e5ddf4
# f2eef9
# ffffff

color_map_full = ["#e26e3d", "#3de26e", "#8e69ce"]

color_map_acq = ["#e57e52", "#e88e68", "#eb9e7d"]
color_map_proc = ["#52e57e", "#68e88e"]
color_map_anal = [
    "#9a79d3",
    "#a78ad8",
    "#b39bde",
    "#c0abe3",
    "#ccbce9",
    "#d9cdee",
    "#e5ddf4",
    "#f2eef9",
]

# Area Plot
x = range(2, 300)
y = [acq[2:], preproc[2:], analysis[2:]]
plt.stackplot(
    x, y, labels=["Acquisition", "Preprocessing", "Analysis"], colors=color_map_full
)

# Line Plots
plt.plot(acq, color_map_full[0], label="Acquisition")
plt.plot(preproc, color_map_full[1], label="Preprocessing")
plt.plot(analysis, color_map_full[2], label="Analysis")
plt.legend(loc="upper right", bbox_to_anchor=(1.04, 1))
plt.xticks([1, 50, 100, 150, 200, 250, 300])
plt.xlabel("Frames")
plt.ylabel("Time (ms)")
plt.show()

plt.plot(acq[0:5], color_map_full[0], label="Acquisition")
plt.plot(preproc[0:5], color_map_full[1], label="Preprocessing")
plt.plot(analysis[0:5], color_map_full[2], label="Analysis")
plt.legend(loc="upper right", bbox_to_anchor=(1.04, 1))
plt.xticks(range(1, 6))
plt.xlabel("Frames")
plt.ylabel("Time (ms)")
plt.show()

plt.plot(acq[2:], color_map_full[0], label="Acquisition")
plt.plot(preproc[2:], color_map_full[1], label="Preprocessing")
plt.plot(analysis[2:], color_map_full[2], label="Analysis")
plt.legend(loc="upper right", bbox_to_anchor=(1.04, 1))
plt.xticks([3, 50, 100, 150, 200, 250, 300])
plt.xlabel("Frames")
plt.ylabel("Time (ms)")
plt.show()
