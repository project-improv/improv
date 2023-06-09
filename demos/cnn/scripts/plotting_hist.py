import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist(data, title, out_path, file_name, bins="auto", range=None, save=True):
    plt.hist(data, bins=bins, range=range)
    plt.xlabel("Times (ms)", fontname="Arial")
    plt.ylabel("Frequency", fontname="Arial")
    plt.title(title)

    if save is True:
        plt.savefig(os.path.join(out_path, file_name), dpi=300)

    plt.close()


acq_out_plots_path = "output/GPU/plots/acquire"
os.makedirs(acq_out_plots_path, exist_ok=True)
acq_data_path = "output/GPU/timing/acq_vars.txt"
with open(acq_data_path, "r") as f:
    acq_data = eval(f.read())
    f.close()

acq_keys = ["put_img_time", "put_out_time", "acq_total_times", "put_lab_time"]
acq_titles = [
    "Image -> Store Time",
    "Image+Label -> q_out Time",
    "Total Acquisition Time",
    "Label -> Store Time",
]
acq_files = [
    "put_img_time.png",
    "put_out_time.png",
    "acq_total_times.png",
    "put_lab_time.png",
]

proc_out_plots_path = "output/GPU/plots/process"
os.makedirs(proc_out_plots_path, exist_ok=True)
proc_data_path = "output/GPU/timing/proc_vars.txt"
with open(proc_data_path, "r") as f:
    proc_data = eval(f.read())
    f.close()

proc_keys = [
    "get_img_out",
    "proc_img_time",
    "to_device",
    "inference_time",
    "out_to_np",
    "put_out_store",
    "put_q_out",
    "pred_time",
    "put_pred_store",
    "total_times",
]
proc_titles = [
    "Stored Image -> Processor Time",
    "Process Image Time",
    "Image -> Device Time",
    "Inference Time",
    "Output -> np.array Time",
    "Output -> Store",
    "Output -> q_out",
    "Prediction Time",
    "Classification/Prediction Output -> Store",
    "Total Processing Time",
]
proc_files = [
    "get_img_out.png",
    "proc_img_time.png",
    "to_device.png",
    "inference_time.png",
    "out_to_np.png",
    "put_out_store.png",
    "put_q_out.png",
    "pred_time.png",
    "put_pred_store.png",
    "total_proc_time.png",
]

for key, title, file_name in zip(acq_keys, acq_titles, acq_files):
    data = eval(acq_data[key])[5:]
    plot_hist(data, title, acq_out_plots_path, file_name)

for key, title, file_name in zip(proc_keys, proc_titles, proc_files):
    data = eval(proc_data[key])[5:]
    plot_hist(data, title, proc_out_plots_path, file_name)
