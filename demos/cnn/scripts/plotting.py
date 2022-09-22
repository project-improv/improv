import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size' : 14,
    'font.family' : 'Arial',
    'lines.linewidth': 0,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 500,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.edgecolor': 'k', 
    'axes.linewidth': 1, 
    'axes.grid': False,
    'xtick.major.width': .5,
    'ytick.major.width': .5,
    'axes.autolimit_mode': 'data',
    'savefig.format': 'svg'})

def plot_hist(data, cmap, title, out_path, file_name, bins='auto', range=None, rc_add=None, save=False, show=False):
    '''
    '''
    fig = plt.figure(figsize=(6.4,4.8))
    ax = plt.subplot(111) 

    if rc_add is not None:
        plt.rcParams.update(rc_add)

    ax.hist(data, bins=bins, range=range)
    ax.set_xlabel('Times (ms)', fontname='Arial')
    ax.set_ylabel('Frequency', fontname='Arial')
    plt.title(title)

    if title is not None:
        ax.set_title(title)

    if save is True:
        fig.savefig(os.path.join(out_path, file_name) + '.svg',
        format='svg',
        bbox_inches='tight',
        dpi=500)

    if show is True:
        pass

    plt.close()

    return fig
    
def plot_time(x, y, logscale=False, labels=list, label_col=int, cmap=list, xticks=list, title=None, plot_type=str, out_path=None, file_name=None, rc_add=None, save=False, show=False):
    '''
    '''
    fig = plt.figure(figsize=(6.4,4.8))

    if rc_add is not None:
        plt.rcParams.update(rc_add)

    if plot_type == 'area':
        if type(y) is pd.DataFrame:
            ax = y.plot.area(color=cmap)
        else:
            ax = plt.subplot(111) 
            ax.stackplot(x, y, labels=labels, colors=cmap)
    
    if plot_type == 'line':
        if type(y) is pd.DataFrame:
            ax = y.plot.line(use_index=True, color=cmap)
        elif type(y) is list and len(y) > 0:
            ax = plt.subplot(111)
            for i in range(len(y)):
                ax.plot(y[i], label=labels[i], color=cmap[i])
    
    if logscale is True:
        plt.yscale('log')
    ax.set_xticks(xticks)
    ax.set_xlabel('Frames', fontname='Arial')
    ax.set_ylabel('Time (ms)', fontname='Arial')
    ax.set_label(labels)  
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.1), ncol=label_col, frameon=False)
    # ax.legend(loc='upper right', bbox_to_anchor=(.15, 1), borderaxespad=0., bbox_transform=fig.transFigure)
    
    if title is not None:
        ax.set_title(title)

    if save is True:
        plt.savefig(os.path.join(out_path, file_name) + '.svg',
        format='svg',
        bbox_inches='tight',
        dpi=500)

    if show is True:
        plt.show()

    plt.close()

    # return fig

# DATA
# HUGE SPIKE @ 0:3 -> start @ 3
n_imgs = 10000
acq_path = 'output/GPU/timing/acq_timing_' + str(n_imgs) + '.csv'
# acq_path = 'output/GPU/timing/acq_timing_09152022.csv'
acq_df = pd.read_csv(acq_path)[:n_imgs]
acq_total_df = acq_df['acq_total_times']
acq_total_df.index = acq_total_df.index + 1
acq_parts = ['get_img_time', 'put_img_time', 'get_lab_time','put_lab_time', 'put_out_time']
acq_parts_df = acq_df[acq_parts]
acq_parts_df.index = acq_parts_df.index + 1

acq_add_parts_df = acq_parts_df.copy(deep=True)
for i in range(len(acq_parts)-1):
    acq_add_parts_df[acq_parts[1+i]] = acq_add_parts_df[[acq_parts[i], acq_parts[i+1]]].sum(axis=1)

proc_path = 'output/GPU/timing/proc_timing_' + str(n_imgs) + '.csv'
proc_df = pd.read_csv(proc_path)[:n_imgs]
preproc = ['get_img_out', 'proc_img_time']
preproc_parts_df = proc_df[preproc]
preproc_parts_df.index = preproc_parts_df.index + 1
preproc_total_df = preproc_parts_df.sum(axis=1)
preproc_total_df.index = preproc_total_df.index

preproc_add_parts_df = preproc_parts_df.copy(deep=True)
preproc_add_parts_df[preproc[0]] = preproc_add_parts_df[preproc[0]] + acq_parts_df['put_out_time']
for i in range(len(preproc)-1):
    preproc_add_parts_df[preproc[1+i]] = preproc_add_parts_df[[preproc[i], preproc[i+1]]].sum(axis=1)

analysis = ['to_device', 'inference_time', 'out_to_np', 'put_out_store', 'pred_time', 'put_pred_store', 'put_q_out']
analysis_parts_df = proc_df[analysis]
analysis_parts_df.index = analysis_parts_df.index + 1
analysis_total_df = analysis_parts_df.sum(axis=1)
analysis_total_df.index = analysis_total_df.index

analysis_add_parts_df = analysis_parts_df.copy(deep=True)
analysis_add_parts_df[analysis[0]] = analysis_add_parts_df[analysis[0]] + preproc_add_parts_df['proc_img_time']
for i in range(len(analysis)-1):
    analysis_add_parts_df[analysis[1+i]] = analysis_add_parts_df[[analysis[i], analysis[i+1]]].sum(axis=1)

# COLORS
# purple for acquisition, gold for preprocessing, blacks for analysis
# COPY-PASTE HEX, SINGLE HUE + SCALE
color_map_full = ['#55185d', '#98730C', '#000000']
color_map_acq = ['#75337b', '#954e9b', '#b86abb', '#db87dd', '#ffa5ff']
color_map_preproc = ['#cca10a', '#ffd300']
# color_map_analysis = ['#212121', '#3c3c3c', '#595959', '#787878', '#989898', '#bababa', '#dddddd'].reverse()
color_map_analysis = ['#bebebe', '#a1a1a1', '#848484', '#686868', '#4e4e4e', '#353535', '#1d1d1d']

# PLOTS
area_dir = 'output/GPU/plots/area'
line_dir = 'output/GPU/plots/line'

os.makedirs(area_dir, exist_ok=True)
os.makedirs(line_dir, exist_ok=True)

# FULL
start = 0
full_cycle_df = [acq_total_df, preproc_total_df, analysis_total_df]
full_labels = ['Acquisition', 'Preprocessing', 'Analysis']
if n_imgs == 1000:
    xticks = [start+1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
elif n_imgs == 10000:
    xticks = [start+1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
x = range(start+1, n_imgs+1)

plot_time(x=x, y=full_cycle_df, logscale=False, labels=full_labels, label_col=len(full_labels), cmap=color_map_full, xticks=xticks, plot_type='area', out_path=area_dir, file_name='full_timing_1000', save=True, show=True)

# FULL from 2:300 to avoid init spike
start = 2
if n_imgs == 1000:
    xticks = [start+1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
elif n_imgs == 10000:
    xticks = [start+1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
full_cycle_df = [acq_total_df[start:], preproc_total_df[start:], analysis_total_df[start:]]
x = range(start+1, n_imgs+1)

# ACQ_PARTS
acq_labels = ['Image from File', 'Image to Store', 'Label from File', 'Label to Store', 'Enqueue']
acq_parts_df.columns = acq_labels

# PROC_PARTS
preproc_labels = ['Get Image Out', 'Preprocess Image']
preproc_parts_df.columns = preproc_labels

# ANALYSIS_PARTS
analysis_labels = ['To Device', 'Inference Time', 'Output to Numpy', 'Output to Store', 'Classify', 'Prediction to Store', 'Enqueue']
analysis_parts_df.columns = analysis_labels

# HISTOGRAMS
# plot_hist()

# AREA PLOTS

ys = [full_cycle_df, acq_parts_df[start:], preproc_parts_df[start:], analysis_parts_df[start:]]
labels = [full_labels, acq_labels, preproc_labels, analysis_labels]
n_col = [3, 3, 2, 3]
cmaps = [color_map_full, color_map_acq, color_map_preproc, color_map_analysis]
fnames = ['full_timing_' + str(start) + ':' + str(n_imgs), 'acq_timing_' + str(n_imgs), 'preproc_timing_' + str(n_imgs), 'analysis_timing_'  + str(n_imgs)]

for p_type, p_dir, p_rc in zip(['area', 'line'], [area_dir, line_dir], [None, {'lines.linewidth': 1.5}]):
    for y, l, n, c, f in zip(ys, labels, n_col, cmaps, fnames):
        plot_time(x=x, y=y, labels=l, logscale=False, label_col=n, cmap=c, xticks=xticks, plot_type=p_type, out_path=p_dir, file_name=f, rc_add=p_rc, save=True, show=False)
        
        # fig.savefig(os.path.join(p_dir, f) + '.svg', format='svg', bbox_inches='tight', dpi=500)

acq_add_parts_df.columns = acq_labels
preproc_add_parts_df.columns = preproc_labels
analysis_add_parts_df.columns = analysis_labels

total_parts_df = pd.concat([acq_add_parts_df[start:], preproc_add_parts_df[start:], analysis_add_parts_df[start:]], axis=1)

total_labels = acq_labels
total_labels.append(preproc_labels)
total_labels.append(analysis_labels)

total_cmap = ['#75337b', '#954e9b', '#b86abb', '#db87dd', '#ffa5ff', '#cca10a', '#ffd300', '#dddddd', '#bababa', '#989898', '#787878', '#595959', '#3c3c3c', '#212121']

f = 'total_parts' + str(n_imgs)
plot_time(x=x, y=total_parts_df, logscale=False, labels=total_labels, label_col=3, cmap=total_cmap, xticks=xticks, plot_type='line', out_path=line_dir, file_name=f, rc_add={'lines.linewidth': 1.5}, save=True, show=True)

f = 'total_parts_log' + str(n_imgs)
plot_time(x=x, y=total_parts_df, logscale=True, labels=total_labels, label_col=3, cmap=total_cmap, xticks=xticks, plot_type='line', out_path=line_dir, file_name=f, rc_add={'lines.linewidth': 1.5}, save=True, show=True)