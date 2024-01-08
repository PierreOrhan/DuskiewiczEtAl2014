import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from scripts.NatNeuro.utils import *
from matplotlib.colors import Colormap

def plot_besttc(Y):
    _, fourier_power = get_Fourier_power(Y)
    best_cells = [[np.argmax(fourier_power[j, :, i]) for i in range(3)] for j in range(10)]
    fig, ax = plt.subplots(3, 10, subplot_kw={'projection': 'polar'}, figsize=(20, 6))
    for j in range(10):
        for i in range(3):
            ax[i, j].plot(theta, Y[j, best_cells[j][i], :], c="black")
            ax[i, j].set_ylim(0, np.max(Y[j, best_cells[j][i], :]) * 1.2)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    fig.show()
    return fig

def plot_proportions(proportions,sigmasonmeans,proportion_true):
    if len(proportions.shape)>2:
        prop = np.mean(proportions,axis=0)
        prop_std = np.std(proportions,axis=0)
    else:
        prop = proportions
        prop_std = prop*0

    colors = ["darkorange","darkgreen","darkred"]
    fig,ax = plt.subplots()
    [ax.plot(sigmasonmeans,prop[:,i],c=colors[i]) for i in range(3)]

    # [ax.plot(sigmasonmeans,np.min(proportions,axis=0)[:,i],c=colors[i],alpha=0.2) for i in range(3)]
    # [ax.plot(sigmasonmeans,np.max(proportions,axis=0)[:,i],c=colors[i],alpha=0.2) for i in range(3)]

    [ax.fill_between(sigmasonmeans,prop[:,i]-prop_std[:,i],
                     prop[:,i]+prop_std[:,i],color=colors[i],alpha=0.3) for i in range(3)]
    [ax.hlines(proportion_true[i],sigmasonmeans[0],sigmasonmeans[-1],color=colors[i],linestyle="--") for i in range(3)]
    ax.set_xscale("log")
    ax.set_xticks([1,10,100,1000,10000])
    ax.set_xticklabels(["",10,100,1000,""],fontsize="x-large")
    ax.set_yticks([0,0.5,1])
    ax.vlines(192,0,1,color="black",label="experimental \n standard deviation")
    ax.legend()
    ax.set_ylim(-0.01,1.01)
    ax.set_yticklabels([0,0.5,1],fontsize="x-large")
    ax.set_ylabel("proportion",fontsize="x-large")
    ax.set_xlabel("weight standard deviation (% of weight mean)",fontsize="x-large")
    fig.show()
    return fig


def _plot_proportions(fig,ax,proportions,sigmasonmeans,id_c):
    if len(proportions.shape)>2:
        prop = np.mean(proportions,axis=0)
        prop_std = np.std(proportions,axis=0)
    else:
        prop = proportions
        prop_std = prop*0

    colors = [plt.get_cmap(o) for o in ["Oranges","Greens","Reds"]]
    [ax.plot(sigmasonmeans,prop[:,i],c=colors[i](id_c)) for i in range(3)]

    # [ax.plot(sigmasonmeans,np.min(proportions,axis=0)[:,i],c=colors[i],alpha=0.2) for i in range(3)]
    # [ax.plot(sigmasonmeans,np.max(proportions,axis=0)[:,i],c=colors[i],alpha=0.2) for i in range(3)]

    [ax.fill_between(sigmasonmeans,prop[:,i]-prop_std[:,i],
                     prop[:,i]+prop_std[:,i],color=colors[i](id_c),alpha=0.3) for i in range(3)]
    return fig

def _legend_proportions(fig,ax,sigmasonmeans,proportion_true):
    colors = ["darkorange","darkgreen","darkred"]

    [ax.hlines(proportion_true[i],sigmasonmeans[0],sigmasonmeans[-1],color=colors[i],linestyle="--") for i in range(3)]
    ax.set_xscale("log")
    ax.set_xticks([1,10,100,1000,10000])
    ax.set_xticklabels(["",10,100,1000,""],fontsize="x-large")
    ax.set_yticks([0,0.5,1])
    ax.vlines(192,0,1,color="black",label="experimental \n standard deviation")
    ax.legend()
    ax.set_ylim(-0.01,1.01)
    ax.set_yticklabels([0,0.5,1],fontsize="x-large")
    ax.set_ylabel("proportion",fontsize="x-large")
    ax.set_xlabel("weight standard deviation (% of weight mean)",fontsize="x-large")
    fig.show()
    return fig

def _legend_proportions_variability(fig,ax):
    ax.set_xscale("log")
    ax.set_xticks([1,10,100,1000,10000])
    ax.set_xticklabels(["",10,100,1000,""],fontsize="x-large")
    ax.set_yticks([0,0.2,0.4])
    ax.vlines(192,0,1,color="black",label="experimental \n standard deviation")
    ax.legend()
    ax.set_ylim(-0.01,0.41)
    ax.set_yticklabels([0,0.2,0.4],fontsize="x-large")
    ax.set_ylabel("standard deviation of proportions \n across simulations with different inputs",fontsize="x-large")
    ax.set_xlabel("weight standard deviation (% of weight mean)",fontsize="x-large")
    return fig


def _plot_mis(fig,ax,mis,sigmasonmeans,id_c):
    if len(mis.shape)>2:
        prop = np.mean(mis,axis=(0,-1))
    else:
        prop = np.mean(mis,axis=-1)
    colors = plt.get_cmap("Blues")
    ax.plot(sigmasonmeans,prop,c=colors(id_c))
    return fig

def _plot_std_mis2(fig,ax,mis,sigmasonmeans,id_c):
    if len(mis.shape)>2:
        prop = np.mean(np.std(mis,axis=(-1)),axis=0)
    else:
        prop = np.std(mis,axis=-1)
    colors = plt.get_cmap("Blues")
    ax.plot(sigmasonmeans,prop,c=colors(id_c))
    return fig

def _legend_mis(fig,ax,ylabel="mutual information"):
    ax.set_xscale("log")
    ax.set_xticks([1,10,100,1000,10000])
    ax.set_xticklabels(["",10,100,1000,""],fontsize="x-large")
    # ax.set_yticks([0,0.5,1])
    ax.vlines(192,0,1,color="black",label="experimental \n standard deviation")
    ax.legend()
    # ax.set_ylim(-0.01,1.01)
    # ax.set_yticklabels([0,0.5,1],fontsize="x-large")
    ax.set_ylabel(ylabel,fontsize="x-large")
    ax.set_xlabel("weight standard deviation (% of weight mean)",fontsize="x-large")
    return fig

class shifted_cmap(Colormap):
    def __init__(self,cmap,shift,d_len,name):
        super(shifted_cmap, self).__init__(name)

        self.cmap = cmap
        self.shift = shift
        self.d_len = d_len
    def __call__(self,X, alpha=None, bytes=False):
        return self.cmap((X*self.d_len+self.shift)/(self.shift+self.d_len))

def compose_mi(label,xs,sigmasonmeans,target,plt_func):
    fig, ax = plt.subplots()
    for idp, p in enumerate(xs):
        fig = plt_func(fig, ax, p, sigmasonmeans, (idp + 5) / (len(xs) + 5))
    fig = _legend_mis(fig, ax, ylabel=label)
    ax.hlines(target, sigmasonmeans[0], sigmasonmeans[-1], color="black")
    plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0, len(xs)),
                                       cmap=shifted_cmap(plt.get_cmap("Blues"), 2, len(xs),
                                                         name="shift_blue")),
                 ax=ax, label="% of inputs")
    ax.set_yscale("log")
    fig.show()

def _plot_kls(fig,ax,kl,kl_std,sigmasonmeans,id_c):
    colors = plt.get_cmap("Blues")
    ax.plot(sigmasonmeans,kl,c=colors(id_c))
    ax.fill_between(sigmasonmeans,kl-kl_std,
                     kl+kl_std,color=colors(id_c),alpha=0.3)
    return fig
def _legend_kls(fig,ax,ylabel="mutual information"):
    ax.set_xscale("log")
    ax.set_xticks([1,10,100,1000,10000])
    ax.set_xticklabels(["",10,100,1000,""],fontsize="x-large")
    # ax.set_yticks([0,0.5,1])
    ax.vlines(192,-10,2,color="black",label="experimental \n standard deviation")
    ax.legend()
    # ax.set_ylim(-0.01,1.01)
    # ax.set_yticklabels([0,0.5,1],fontsize="x-large")
    ax.set_ylabel(ylabel,fontsize="x-large")
    ax.set_xlabel("weight standard deviation (% of weight mean)",fontsize="x-large")
    ax.set_ylim(-8,2)
    return fig
