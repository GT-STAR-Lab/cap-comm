import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def preamble():
    
    sns.set(style="white", font_scale=2.5)

    colors = ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"]
    palette = sns.color_palette(colors, n_colors=5)
    
    sns.set_palette(palette=palette)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    sns.set(rc={"font.size": 16.0})
    x_labels = ax.get_xlabel()
    return fig, ax

def set_color_by_order(order):
    # order = ["ID(MLP)", "ID(GNN)", "CA(MLP)", "CA(GNN)", "CA+CC(GNN)"]
    _colors = {"ID(MLP)":"#003f5c", "ID(GNN)": "#58508d", "CA(MLP)": "#bc5090", "CA(GNN)": "#ff6361", "CA+CC(GNN)":"#ffa600"}
    colors = []
    for alg in order:
        colors.append(_colors[alg])
    palette = sns.color_palette(colors)
    sns.set_palette(palette=palette)

def outtro(kwargs):
    fig, ax = plt.gcf(), plt.gca()
    for key, value in kwargs.items():
        if(key=="title"):
            plt.title(value)
        if(key=="ylabel"):
            plt.ylabel(value)
        if(key=="xlabel"):
            plt.xlabel(value)
        if(key=="xlabel_fontsize"):
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=value)
        if(key=="ylim"):
            plt.ylim(value)
        if(key=="xlim"):
            plt.xlim(value)

    if("save_path" in kwargs.keys()):
        value = kwargs["save_path"]
        plt.savefig(os.path.join(value, kwargs["fig_name"]), dpi=300, bbox_inches="tight")
        

def template(data, x="models", y=None, **kwargs):
    fig, ax = preamble()



    outtro(kwargs)
    return(fig, ax)

def boxplot_v0(data, x="models", y=None, order=None, **kwargs):
    fig, ax = preamble()
    set_color_by_order(order)
    sns.boxplot(data=data, ax=ax, x=x, y=y, showfliers=False, order=order, width=0.6,
                boxprops={'edgecolor': 'gray'}, 
                whiskerprops={'color': 'gray'},
                capprops={'color':'gray'})
    # sns.stripplot(data=data, ax=ax, jitter=0.2, color='black', x=x, y=y, alpha=0.01)
    outtro(kwargs)
    return(fig, ax)

    
    outtro(kwargs)
    return(fig, ax)

def boxplot_connected_components(data, **kwargs):
    fig, ax = preamble()
    df_exploded = data.explode('connected components', ignore_index=False)
    df_exploded["Index"] = df_exploded.groupby(df_exploded.index).cumcount() + 1

    sns.boxplot(data=df_exploded, ax=ax, x="Index", y="connected components", 
                hue="models", showfliers=False, 
                boxprops={'edgecolor': 'gray'}, 
                whiskerprops={'color': 'gray'},
                capprops={'color':'gray'})
    
    plt.legend(loc="best")
    outtro(kwargs)
    return(fig, ax)

def plot_single_connected_components(data, order=None, **kwargs):
    fig, ax = preamble()
    df_exploded = data.explode("percentage of single connected components", ignore_index=False)
    df_exploded["time_step"] = df_exploded.groupby(df_exploded.index).cumcount() + 1
    df_exploded = df_exploded[["models", "percentage of single connected components", "time_step"]]
    
    sns.lineplot(data=df_exploded, ax=ax, x="time_step", y="percentage of single connected components", hue="models", hue_order=order)

    outtro(kwargs)
    return(fig, ax)

def plot_percentage_of_full_connected_episodes(data, order=None, **kwargs):
    fig, ax = preamble()
    set_color_by_order(order)

    def uncum(x):
        x = np.array(x)
        x = x * np.arange(1, 81+1)
        x = np.diff(x, prepend=0).tolist()
        return(x)
    data["connected components count at step"] = data["percentage of single connected components"].apply(uncum)

    df_exploded = data.explode("connected components count at step", ignore_index=False)
    df_exploded["time_step"] = df_exploded.groupby(df_exploded.index).cumcount() + 1
    df_exploded = df_exploded[["models", "connected components count at step", "time_step"]]
    
    sns.lineplot(data=df_exploded, ax=ax, x="time_step", y="connected components count at step", hue="models", hue_order=order, linewidth=4)

    outtro(kwargs)
    return(fig, ax)

def plot_training_curves(data, smoothing_factor=1.0, hue_order=None, **kwargs):
    fig, ax = preamble()
    set_color_by_order(hue_order)

    def smooth_data(group, smoothing_factor=smoothing_factor):
        group['smoothed_y'] = group['values'].ewm(alpha=smoothing_factor).mean()
        return group

    # smooth each model with each seed
    smoothed_data = data.groupby(['model', 'seed']).apply(smooth_data)
    sns.lineplot(data=smoothed_data, x="steps", y="smoothed_y", 
                 errorbar='sd', hue="model", hue_order=hue_order, linewidth=4)

    outtro(kwargs)
    return(fig, ax)
