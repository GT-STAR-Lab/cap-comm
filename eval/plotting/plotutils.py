import matplotlib.pyplot as plt
import seaborn as sns
import os

def preamble():
    
    sns.set(style="white", font_scale=1.2)
    sns.set_palette(sns.color_palette('bright'))
    
    fig, ax = plt.subplots()
    sns.despine(fig=fig, ax=ax)
    return fig, ax

def outtro(kwargs):
    for key, value in kwargs.items():
        if(key=="title"):
            plt.title(value)
        if(key=="ylabel"):
            plt.ylabel(value)
        if(key=="xlabel"):
            plt.xlabel(value)
        if(key=="save_path"):
            plt.savefig(os.path.join(value, kwargs["fig_name"]), bbox_inches="tight")

def template(data, x="models", y=None, **kwargs):
    fig, ax = preamble()



    outtro(kwargs)
    return(fig, ax)

def boxplot_v0(data, x="models", y=None, **kwargs):
    fig, ax = preamble()
    sns.boxplot(data=data, ax=ax, x=x, y=y, showfliers=False, 
                boxprops={'edgecolor': 'gray'}, 
                whiskerprops={'color': 'gray'},
                capprops={'color':'gray'})
    sns.stripplot(data=data, ax=ax, jitter=0.2, color='black', x=x, y=y, alpha=0.01)
    
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

def plot_training_curves(data, smoothing_factor=1.0, hue_order=None, **kwargs):
    fig, ax = preamble()

    def smooth_data(group, smoothing_factor=smoothing_factor):
        group['smoothed_y'] = group['values'].ewm(alpha=smoothing_factor).mean()
        return group

    # smooth each model with each seed
    smoothed_data = data.groupby(['model', 'seed']).apply(smooth_data)
    sns.lineplot(data=smoothed_data, x="steps", y="smoothed_y", 
                 errorbar='sd', hue="model", hue_order=hue_order)

    outtro(kwargs)
    return(fig, ax)