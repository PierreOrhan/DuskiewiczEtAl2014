import os.path
import scipy.io

from scripts.NatNeuro.all_plots import *
from scripts.NatNeuro.all_plots import _plot_proportions,_plot_kls,_plot_mis,_plot_std_mis2,_legend_mis,_legend_proportions
from scripts.NatNeuro.all_plots import _legend_proportions_variability,_legend_kls
import tqdm
import zarr as zr
import hdf5storage
from scripts.load_tc import *

path_data = os.path.join(os.path.abspath(__file__),"..","..","..","data")
os.makedirs(os.path.join(path_data,"dataSimulations"),exist_ok=True)
proportion_true = get_propotion(tcfs_pos)

tc_x = np.concatenate([tcpyr_ad,tcpyr_pos],axis=0)

sparse = 0.3
sparse_id = np.random.choice(tc_x.shape[0], int(np.floor(tc_x.shape[0]*sparse)), replace=False)
sparse_tc = tc_x[sparse_id,:]

means = 5
sigmas = np.logspace(0.2, 2, 40)
sigmasonmeans = 100 * sigmas / means

if os.path.exists(os.path.join(path_data,"dataSimulations","figure3_DE.mat")):
    Y_acrossrun = hdf5storage.loadmat(os.path.join(path_data,"dataSimulations", "figure3_DE.mat"))["Y_acrossruns"]
else:
    Y_acrossrun  =[]
    for i in tqdm.tqdm(range(10)):
        W2cell = np.stack([LogNormal2(means / sparse_tc.shape[0], sigma / sparse_tc.shape[0], (1000, sparse_tc.shape[0]))
                       for sigma in sigmas], axis=0)
        Y = W2cell@sparse_tc
        Y_acrossrun+=[Y]
    Y_acrossrun = np.stack(Y_acrossrun,axis=0)

### Plot of figure 3D:
# This figure plots example tuning curves from one simulation
# as a function of the weights coefficient of variation (std over mean).
# The objective is to illustrate the change in shape of the tuning curves.
plot_besttc(Y_acrossrun[0,::4,...])

### Plot of figure 3G:
# Plots the proportions of cells with a certain prefer folding, as a function
# of the weights coefficient of variations:
proportions = get_propotion(Y_acrossrun)
plot_proportions(proportions,sigmasonmeans,proportion_true)

# additionally compute the snrs and mutual info:
snrs = snr_tc(Y_acrossrun)
mis = mi_tc_bitrate(Y_acrossrun)

if not os.path.exists(os.path.join(path_data,"dataSimulations","figure3_DE.mat")):

    hdf5storage.savemat(os.path.join(path_data,"dataSimulations","figure3_DE.mat"),
                        {"proportions":proportions,
                         "Y_acrossruns":Y_acrossrun,
                         "sigmasonmeans":sigmasonmeans,
                         "proportion_true":proportion_true,
                         "sparse_id":sparse_id,
                         "snr":snrs,
                         "mutual_info":mis,
                         "dimensions_Y":np.array(["simulation repetition",
                                       "weight statistics","cells","HD"])})

    hdf5storage.savemat(os.path.join(path_data,"dataSimulations","figure3_DE_light.mat"),
                        {"proportions":proportions,
                         "Y_onerun":Y_acrossrun[0,...],
                         "sigmasonmeans":sigmasonmeans,
                         "proportion_true":proportion_true,
                         "sparse_id":sparse_id,
                         "snr":snrs,
                         "mutual_info":mis,
                         "dimensions_Y":np.array(["simulation repetition",
                                       "weight statistics","cells","HD"])})


## Supplementary figure 5a:
# No effect of the weight distribution
# We demonstrate that the previous result obtained with LogNormally distributed weights is
# in fact independent of the weight distribution. This fact is analytically proven
# in the supplementary texts of the paper (in section 3. Fourier spectrum simulations – theory)
means = 5
sigmas = np.logspace(0.2, 2, 40)
sigmasonmeans = 100 * sigmas / means

if not os.path.exists(os.path.join(path_data,"dataSimulations","supp_1_proportions.mat")):
    Y_distribs = {}
    for name,distrib in zip(["ln","normal","uniform"],[LogNormal2,np.random.normal,uniform2]):
        Y_acrossrun  =[]
        for i in tqdm.tqdm(range(10)):
            W2cell = np.stack([distrib(means / sparse_tc.shape[0], sigma / sparse_tc.shape[0], (1000, tc_x.shape[0]))
                           for sigma in sigmas], axis=0)
            Y = W2cell@tc_x
            Y_acrossrun+=[Y]
        Y_acrossrun = np.stack(Y_acrossrun,axis=0)
        Y_distribs[name] = Y_acrossrun
    for k in ["ln", "normal", "uniform"]:
        Y_distribs[k + "_proprotions"] = get_propotion(Y_distribs[k])
    hdf5storage.savemat(os.path.join(path_data, "dataSimulations", "supp_1_proportions.mat"),
                        Y_distribs)
else:
    Y_distribs = hdf5storage.loadmat(os.path.join(path_data,"dataSimulations","supp_1_proportions.mat"))

for k in ["ln","normal","uniform"]:
    fig = plot_proportions(Y_distribs[k+"_proprotions"],sigmasonmeans,proportion_true)
    fig.suptitle(k)
    fig.show()

## supplementary figure 5d:
## Increasing sparsity but allowing to sample from all possible inputs:
zg = zr.open_group(os.path.join(path_data,"dataSimulations","sparsity_effect.zarr"))
for sparse in tqdm.tqdm(np.arange(0.05,1,step=0.05)):
    if not str(sparse) in zg.keys():
        Y_acrossrun_fullinput  =[]
        for _ in range(10):
            W2 = np.stack([LogNormal2(means / tc_x.shape[0], sigma / tc_x.shape[0], (1000, tc_x.shape[0]))
                           for sigma in sigmas], axis=0)
            W2_sparse = np.copy(W2)
            nb_zero = int(np.floor(W2.shape[-1] * (1 - sparse)))
            for i in range(W2.shape[1]):
                W2_sparse[:, i, np.random.choice(W2.shape[-1], nb_zero, replace=False)] = 0
            Y = W2_sparse@tc_x
            Y_acrossrun_fullinput+=[Y]
        zg.array(str(sparse),np.stack(Y_acrossrun_fullinput,axis=0))
    if not str(sparse)+"prop" in zg.keys():
        prop = get_propotion(zg[sparse])
        zg.array(str(sparse)+"prop",prop)

proportions = [zg[str(sparse)+"prop"][:] for sparse in tqdm.tqdm(np.arange(0.05,1,step=0.05))]

fig,ax = plt.subplots()
for idp,p in enumerate(proportions):
    fig = _plot_proportions(fig, ax, p, sigmasonmeans,(idp+2)/(len(proportions)+2))
fig = _legend_proportions(fig, ax, sigmasonmeans, proportion_true)
plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,len(proportions)),
                                   cmap=shifted_cmap(plt.get_cmap("Oranges"),2,len(proportions),name="shift_blue")),
                                ax=ax,label="% of inputs")
plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,len(proportions)),
                                   cmap=shifted_cmap(plt.get_cmap("Greens"),2,len(proportions),name="shift_blue")),
                                ax=ax,label="% of inputs")
plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,len(proportions)),
                                   cmap=shifted_cmap(plt.get_cmap("Reds"),2,len(proportions),name="shift_blue")),
                                ax=ax,label="% of inputs")
fig.show()

if not os.path.exists(os.path.join(path_data,"dataSimulations","supp_2_allinputs_sparsity.mat")):
    snrs = []
    mi_bitrates = []
    for k in tqdm.tqdm(np.arange(0.05,1,step=0.05)):
        snrs+=[snr_tc(zg[k])]
        mi_bitrates+=[mi_tc_bitrate(zg[k])]

    ## The :-2 removes two outliers in fs-cells dataset which were badly classified as FS cells.
    snr_tcs = snr_tc(tcfs_pos)
    target_snr_std = np.std(snr_tcs[np.argsort(snr_tcs)][:-2])
    target_snr_mean = np.mean(snr_tcs[np.argsort(snr_tcs)][:-2])

    true_mitc_bitspike = mi_tc_bitrate(tcfs_pos)
    snr_tcs = snr_tc(tcfs_pos)
    target_mi_std = np.std(true_mitc_bitspike[np.argsort(snr_tcs)][:-2])
    target_mi_mean = np.mean(true_mitc_bitspike[np.argsort(snr_tcs)][:-2])

    hdf5storage.savemat(os.path.join(path_data,"dataSimulations","supp_2_allinputs_sparsity.mat"),
                        {"mutual_informations_bitperspike":np.stack(mi_bitrates,axis=0),
                         "snrs":np.stack(snrs,axis=0),
                         "proportions":np.stack(proportions,axis=0),
                         "mutual_info_data_bitperspike":true_mitc_bitspike,
                         "snr_data":snr_tcs,
                         "mean_snr_data_removeoutlier":target_snr_mean,
                         "mean_mi_data_removeoutlier":target_mi_mean,
                         "std_snr_data_removeoutlier": target_snr_std,
                         "std_mi_data_removeoutlier": target_mi_std,
                         "sigmasonmeans":sigmasonmeans})
else:
    hdmat = hdf5storage.loadmat(os.path.join(path_data,"dataSimulations","supp_2_allinputs_sparsity.mat"))
    mi_bitrates = hdmat["mutual_informations_bitperspike"]
    target_mi_std=  hdmat["std_mi_data_removeoutlier"]
    target_snr_mean = hdmat["mean_snr_data_removeoutlier"]
    snrs = hdmat["snrs"]
    target_mi_mean = hdmat["mean_mi_data_removeoutlier"]
    target_snr_std = hdmat["std_snr_data_removeoutlier"]

# Supplementary figure 5e:
# We plot the output cells mean mutual information as a function of the sparsity
# and of the weights coefficient of variation (std over mean)
fig,ax = plt.subplots()
for idp,p in enumerate(mi_bitrates):
    fig = _plot_mis(fig, ax, p, sigmasonmeans,(idp+5)/(len(mis)+5))
fig = _legend_mis(fig, ax,ylabel="mean mutual information (bit/spike)")
ax.set_yscale("log")
ax.hlines(target_mi_std,sigmasonmeans[0],sigmasonmeans[-1],color="black")
plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,len(proportions)),
                                   cmap=shifted_cmap(plt.get_cmap("Blues"),2,len(proportions),name="shift_blue")),
                                ax=ax,label="% of inputs")
fig.show()
# Same but for the standard deviation of the mutual information
fig,ax = plt.subplots()
for idp,p in enumerate(mi_bitrates):
    fig = _plot_std_mis2(fig, ax, p, sigmasonmeans,(idp+5)/(len(mis)+5))
fig = _legend_mis(fig, ax,ylabel="std mutual information (bit/spike)")
ax.set_yscale("log")
ax.hlines(target_mi_std,sigmasonmeans[0],sigmasonmeans[-1],color="black")
plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,len(proportions)),
                                   cmap=shifted_cmap(plt.get_cmap("Blues"),2,len(proportions),name="shift_blue")),
                                ax=ax,label="% of inputs")
fig.show()

#### Same plots than the mutual informations but for the SNR of the tuning curve
# we obtain very similar results
# (These plots were not included in the paper for the sake of simplicity)
fig,ax = plt.subplots()
for idp,p in enumerate(snrs):
    fig = _plot_mis(fig, ax, p, sigmasonmeans,(idp+5)/(len(mis)+5))
fig = _legend_mis(fig, ax,ylabel="mean SNR")
ax.hlines(target_snr_mean,sigmasonmeans[0],sigmasonmeans[-1],color="black")
plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,len(proportions)),
                                   cmap=shifted_cmap(plt.get_cmap("Blues"),2,len(proportions),name="shift_blue")),
                                ax=ax,label="% of inputs")
ax.set_yscale("log")
fig.show()
compose_mi("mean SNR",snrs,sigmasonmeans,target_snr_mean,_plot_mis)

fig,ax = plt.subplots()
for k in tqdm.tqdm(np.arange(0.05, 1, step=0.05)):
    fig = _plot_std_mis2(fig, ax, p, sigmasonmeans,(idp+5)/(len(mis)+5))
fig = _legend_mis(fig, ax,ylabel="std SNR")
ax.hlines(target_snr_std,sigmasonmeans[0],sigmasonmeans[-1],color="black")
plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,len(proportions)),
                                   cmap=shifted_cmap(plt.get_cmap("Blues"),2,len(proportions),name="shift_blue")),
                                ax=ax,label="% of inputs")
ax.set_yscale("log")
fig.show()



if not os.path.exists(os.path.join(path_data,"dataSimulations","supp_2_allinputs_sparsity_kldiv.mat")):
    ## KL divergence of the SNR and Mutual Information:
    kl_mi_bitspike_nominal = []
    kl_mi_bitspike_stddev = []
    kl_snr_nominal = []
    kl_snr_stddev = []

    for k in tqdm.tqdm(np.arange(0.05,1,step=0.05)):
        x = snr_tc(zg[k])
        nominal,stdev =log_kl_uncertainty(np.mean(x,axis=-1),np.std(x,axis=-1),
                                    target_snr_mean,
                                    target_snr_std)
        kl_snr_nominal += [nominal]
        kl_snr_stddev += [stdev]

        x = mi_tc_bitrate(zg[k])
        nominal,stdev =log_kl_uncertainty(np.mean(x,axis=-1),np.std(x,axis=-1),
                                    target_mi_mean,
                                    target_mi_std)

        kl_mi_bitspike_nominal += [nominal]
        kl_mi_bitspike_stddev += [stdev]

    hdf5storage.savemat(os.path.join(path_data,"dataSimulations","supp_2_allinputs_sparsity_kldiv.mat"),
                        {
                         "kl_snr_mean":np.stack(kl_snr_nominal,axis=0),
                         "kl_snr_stddev":np.stack(kl_snr_stddev,axis=0),
                            "kl_mi_mean": np.stack(kl_mi_bitspike_nominal, axis=0),
                            "kl_mi_stddev": np.stack(kl_mi_bitspike_stddev, axis=0),
                            "sigmasonmeans":sigmasonmeans
                        })
else:
    hdmat = hdf5storage.loadmat(os.path.join(path_data,"dataSimulations","supp_2_allinputs_sparsity_kldiv.mat"))
    kl_snr_nominal = hdmat["kl_snr_mean"]
    kl_snr_stddev = hdmat["kl_snr_stddev"]
    kl_mi_bitspike_nominal = hdmat["kl_mi_mean"]
    kl_mi_bitspike_stddev = hdmat["kl_mi_stddev"]
    sigmasonmeans = hdmat["sigmasonmeans"]

fig,ax = plt.subplots()
for idp,tp in enumerate(zip(kl_snr_nominal,kl_snr_stddev)):
    fig = _plot_kls(fig, ax, tp[0],tp[1], sigmasonmeans,(idp+5)/(len(mis)+5))
fig = _legend_kls(fig, ax,ylabel="log kl of SNR distributions")
plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,100),
                                   cmap=shifted_cmap(plt.get_cmap("Blues"),20,100,name="shift_blue")),
                                ax=ax,label="% of inputs")
fig.show()

fig,ax = plt.subplots()
for idp,tp in enumerate(zip(kl_mi_bitspike_nominal,kl_mi_bitspike_stddev)):
    fig = _plot_kls(fig, ax, tp[0],tp[1], sigmasonmeans,(idp+5)/(len(mis)+5))
fig = _legend_kls(fig, ax,ylabel="log kl of mutual-information distributions")
plt.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,100),
                                   cmap=shifted_cmap(plt.get_cmap("Blues"),20,100,name="shift_blue")),
                                ax=ax,label="% of inputs")
fig.show()


##### Supplementary figure 5B
# Goal: get a clearer pictures of the variability due to using different set of inputs

# Bottom of the figure: we just shift the tuning curves:
W2 = np.stack([LogNormal2(means / tc_x.shape[0], sigma / tc_x.shape[0], (1000, tc_x.shape[0]))
               for sigma in sigmas], axis=0)
W2_sparse = np.copy(W2)
nb_zero = int(np.floor(W2.shape[-1] * (1 - sparse)))
for i in range(W2.shape[1]):
    W2_sparse[:, i, np.random.choice(W2.shape[-1], nb_zero, replace=False)] = 0
Yshift = []
for _ in tqdm.tqdm(range(30)):
    shift = np.random.randint(0,tc_x.shape[-1]-1,tc_x.shape[0])
    ids = np.mod(np.arange(tc_x.shape[-1])[None,:] + shift[:,None],tc_x.shape[-1])
    tc_s = np.stack([tc_x[i,ids[i,:]] for i in range(tc_x.shape[0])])
    Y = W2_sparse@tc_s
    Yshift+=[Y]
Yshift= np.stack(Yshift,axis=0)

if not os.path.exists(os.path.join(path_data,"dataSimulations","supp_3_inputsdependence.mat")):
    proportions = get_propotion(Yshift)
    colors = ["darkorange", "darkgreen", "darkred"]
    fig,ax = plt.subplots()
    for i in range(30):
        for j in range(3):
            ax.plot(sigmasonmeans,proportions[i,:,j],c=colors[j],alpha=0.5)
    _legend_proportions(fig,ax,sigmasonmeans,proportion_true)
    ax.set_title("Shift of the inputs tuning curve \n one random weight matrix")
    fig.show()

    # Top B: we perform different sampling of the tuning curves
    Yshift_bysparse = []
    for _ in tqdm.tqdm(range(30)):
        sparse = 0.3
        sparse_id = np.random.choice(tc_x.shape[0], int(np.floor(tc_x.shape[0] * sparse)), replace=False)
        sparse_tc = tc_x[sparse_id, :]
        Y = W2[:,:,sparse_id]@sparse_tc
        Yshift_bysparse+=[Y]
    Yshift_bysparse= np.stack(Yshift_bysparse,axis=0)
    proportions_shiftbysparse = get_propotion(Yshift_bysparse)

    hdf5storage.savemat(os.path.join(path_data,"dataSimulations","supp_3_inputsdependence.mat"),
                        {"sigmasonmeans":sigmasonmeans,
                         "Weight_matrix":W2,
                         "Yshift":Yshift,
                         "Ysparsity":Yshift_bysparse,
                         "proportions_shift":proportions,
                         "proportions_shiftbysparse":proportions_shiftbysparse})

    hdf5storage.savemat(os.path.join(path_data,"dataSimulations","supp_3_inputsdependence_light.mat"),
                        { "proportions_shift":proportions,
                         "proportions_shiftbysparse":proportions_shiftbysparse,
                          "sigmasonmeans":sigmasonmeans})
else:
    hdmat = hdf5storage.loadmat(os.path.join(path_data,"dataSimulations","supp_3_inputsdependence.mat"))
    proportions_shiftbysparse = hdmat["proportions_shiftbysparse"]
    proportions = hdmat["proportions"]
    sigmasonmeans = hdmat["sigmasonmeans"]

colors = ["darkorange", "darkgreen", "darkred"]
fig,ax = plt.subplots()
for i in range(30):
    for j in range(3):
        ax.plot(sigmasonmeans,proportions_shiftbysparse[i,:,j],c=colors[j],alpha=0.5)
_legend_proportions(fig,ax,sigmasonmeans,proportion_true)
ax.set_title("Different sampling of subset of inputs \n one random weight matrix")
fig.show()

# Supplementary Figure 5D
# A figure to recapitulate the variability:
fig,ax = plt.subplots()
for j in range(2):
    ax.plot(sigmasonmeans,np.std(proportions[..., j],axis=0), c=colors[j], alpha=0.5
            ,linestyle="--")
    ax.plot(sigmasonmeans, np.std(proportions_shiftbysparse[..., j],axis=0), c=colors[j], alpha=0.5)
ax.plot(sigmasonmeans,np.std(proportions[..., 2],axis=0), c=colors[2], alpha=0.5
            ,linestyle="--",label="circular shift of inputs")
ax.plot(sigmasonmeans, np.std(proportions_shiftbysparse[..., 2],axis=0),
        c=colors[2], alpha=0.5,label="different subset of inputs")
_legend_proportions_variability(fig,ax)
ax.set_title("Variability due to the set of inputs \n one random weight matrix")
fig.tight_layout()
fig.show()

## Figure 3 paper, panel G right:
colors = ["darkorange", "darkgreen", "darkred"]
fig,ax = plt.subplots()
for j in range(3):
    ax.plot(sigmasonmeans,np.std(proportions_shiftbysparse[...,j],axis=0),c=colors[j])
_legend_proportions_variability(fig,ax)
fig.tight_layout()
fig.show()
hdf5storage.savemat(os.path.join(path_data,"dataSimulations","figure3_H.mat"),
                    {"proportions":proportions_shiftbysparse,
                      "proportions_std":np.std(proportions_shiftbysparse,axis=0),
                     "sigmasonmeans":sigmasonmeans})