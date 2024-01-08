import numpy as np
import  matplotlib.pyplot as plt
from scripts.load_tc import *
from scripts.load_noise import *
from scripts.fourierbasis_hd import *
from itertools import  product
import tqdm
import uncertainties.unumpy as unp
from scipy.stats import lognorm
from utils import LogNormal2,e_LN,σ2_LN
import torch
import zarr as zr
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.use("TkAgg")
from scripts.NatNeuro.utils import snr_tc,mi_tc_bitrate
import numpy as np

## In this file we fit the mean and standard deviations of Ex AD -> FS Posub
# and EX Posub -> FS Posub
# that best fulfill the three optimisation objective describe in supplementary material.
# Crucially we don't optimize the whole connectivity matrix but simply the parameters
# of the random distribution from which it is sampled.
# Additionally, we let the sparsity as a free parameter that can be further tuned to enhance the fit
# with the data.

snr_tcs = snr_tc(tcfs_pos)
target_snr_std = np.std(snr_tcs[np.argsort(snr_tcs)][:-2])
target_snr_mean = np.mean(snr_tcs[np.argsort(snr_tcs)][:-2])
true_mitc_bitspike = mi_tc_bitrate(tcfs_pos)
target_mi_std = np.std(true_mitc_bitspike[np.argsort(snr_tcs)][:-2])
target_mi_mean = np.mean(true_mitc_bitspike[np.argsort(snr_tcs)][:-2])

# Parametrizes the  mean and standard deviation
# of the mutual-information....
def get_tc_shift(tcpyr_ad):
    shift = np.random.randint(0, tcpyr_ad.shape[-1] - 1, tcpyr_ad.shape[0])
    ids = np.mod(np.arange(tcpyr_ad.shape[-1])[None, :] + shift[:, None], tcpyr_ad.shape[-1])
    tc_shift = np.stack([tcpyr_ad[i, ids[i, :]] for i in range(tcpyr_ad.shape[0])])
    return tc_shift
#
tcpyr_ad_shift = np.concatenate([get_tc_shift(tcpyr_ad) for _ in range(10)])
tcpyr_pos_shift = np.concatenate([get_tc_shift(tcpyr_pos) for _ in range(1)])
# tcpyr_ad_shift = tcpyr_ad
# tcpyr_pos_shift = tcpyr_pos

### parameters for the two loss forcing similar firing rate statistics:
a = np.mean(np.mean(tcfs_pos,axis=-1))
b = np.mean(np.mean(tcpyr_ad_shift,axis=-1))
c = np.mean(np.mean(tcpyr_pos_shift,axis=-1))
d = np.mean(np.std(tcfs_pos,axis=-1)**2)
f = np.mean(np.std(tcpyr_pos_shift,axis=-1)**2)/tcpyr_pos_shift.shape[0]
e = np.mean(np.std(tcpyr_ad_shift,axis=-1)**2)/tcpyr_ad_shift.shape[0]

## Optimization to find the best set of parameters:
k_fourier = 40 # number of fourier vector used
mFourier_basis = Fourier_basis[:k_fourier,:]
fourier_coeff_pyrpos = tcpyr_pos_shift@np.transpose(mFourier_basis)
fourier_coeff_pyrad = tcpyr_ad_shift@np.transpose(mFourier_basis)
fourier_coeff_pyrfs = tcfs_pos@np.transpose(mFourier_basis)

sv_pospyr = np.nanstd(fourier_coeff_pyrpos,axis=0)
sv_adpyr = np.nanstd(fourier_coeff_pyrad,axis=0)
sv_fspyr = np.nanstd(fourier_coeff_pyrfs,axis=0)
sv_ex = np.stack([sv_adpyr, sv_pospyr], axis=1).transpose()

W = torch.tensor(np.array([0.2,1.0,4.0,5.0]),requires_grad=True)
sv_ex = torch.tensor(np.stack([sv_adpyr, sv_pospyr], axis=1)).transpose(0, 1)
y = torch.tensor(sv_fspyr)
SStot = torch.sum((y - torch.mean(y)) ** 2)
def loss(W):
    pred_spectral = W[:2]@sv_ex
    SSres = torch.sum((pred_spectral - y) ** 2)
    L1  = (SSres / SStot)
    L2 = torch.sqrt((d - (W[2]**2 + (W[0] ** 2) * tcpyr_ad_shift.shape[0]) * e
                    - ((W[3] ** 2) + (W[1] ** 2) * tcpyr_pos_shift.shape[0]) * f) ** 2)
    L3 = torch.sqrt((a - (W[2]*b+W[3]*c))**2)
    Ltot = L1+0.25*L2/d + 0.25*L3/a
    return Ltot,L1,L2/d,L3/a

import zarr as zr
zg = zr.open_group(os.path.join(path,"dataSimulations","loss_optim.zarr"),mode="a") #OriginalTC
if "losses" not in zg.keys():
    optimizer = torch.optim.Adam([W], lr=0.0001)
    loss_tot =[]
    for _ in tqdm.tqdm(range(200000)): #50000
        l,l1,l2,l3 = loss(W)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        loss_tot+=[[l.detach().cpu().item(),l1.detach().cpu().item(),l2.detach().cpu().item(),l3.detach().cpu().item()]]
    loss_tot = np.array(loss_tot)
    zg.array("losses",np.array(loss_tot))
    zg.array("Woptim",W.detach().cpu().numpy())
    zg.array("names",np.array(["sol_Spectral_AD","sol_Spectral_pos","sol_moment_ad"]))
Wbest = zg["Woptim"][:]
loss_tot = zg["losses"][:]

sigma_ad = Wbest[0]/np.sqrt(tcpyr_ad_shift.shape[0])
sigma_pos = Wbest[1]/np.sqrt(tcpyr_pos_shift.shape[0])
mu_ad = Wbest[2]/tcpyr_ad_shift.shape[0]
mu_pos = Wbest[3]/tcpyr_pos_shift.shape[0]

### Given the result of this optimization, we can display
# example of generated tuning curve and highlight the contributions of different inputs.
# In the following we use a sparsity of 0.5 but this parameter could be further tuned.
def get_True_fromEff(mu_eff,sigma_eff,p):
    # sigma_eff: the effective final standard deviation of the weight matrix after applying the sparsity.
    # my_eff: same but mean
    # p: the proportion of sparsity.
    # returns: mean,sigma : the value of mean and initialization to use in the new weight matrix.

    mu_init = mu_eff/p
    sigma_init = np.sqrt(sigma_eff**2 - p*(1-p)*(mu_eff/p)**2)
    return mu_init,sigma_init


min_p_pos =1/(1+(sigma_pos/mu_pos)**2)
p_pos = 0.5
mu_init_pos,sigma_init_pos = get_True_fromEff(mu_pos,sigma_pos,p_pos)

min_p_ad = 1/(1+(sigma_ad/mu_ad)**2)
p_ad = 1
mu_init_ad,sigma_init_ad = get_True_fromEff(mu_ad,sigma_ad,p_ad)

W1 = LogNormal2(mu_init_ad,sigma_init_ad, (5000, tcpyr_ad_shift.shape[0]))
nb_zero = int(W1.shape[-1]*(1-p_ad))
for k in range(W1.shape[0]):
    W1[k, np.random.choice(W1.shape[-1], nb_zero, replace=False)] = 0
W2 = LogNormal2(mu_init_pos, sigma_init_pos, (5000, tcpyr_pos_shift.shape[0]))
nb_zero = int(W2.shape[-1]*(1-p_pos))
for k in range(W2.shape[0]):
    W2[k, np.random.choice(W2.shape[-1], nb_zero, replace=False)] = 0
Y_off =  W1@tcpyr_ad_shift + W2@tcpyr_pos_shift
Ydelta =  W1@tcpyr_ad_shift
Ymult = W2@tcpyr_pos_shift

mis = mi_tc_bitrate(Y_off)

### Let's make sure the generated tunning curve have a nice shape:
_,power_Ysimul = get_Fourier_power(Y_off)
to_display = np.concatenate([np.argsort(power_Ysimul[:,i])[-100:-80][np.random.choice(range(20),3,replace=False)] for i in range(3)])

fig = plt.figure(figsize=(20,10))
# ax = fig.subplot_mosaic([[str(i)+"_tc" for i in range(9)],[0,0,0,1,1,1,2,2,2],["means","means","means","stds","stds","stds","power","power","power"]],
#                         per_subplot_kw = {str(i)+"_tc" : {"projection":"polar"} for i in range(9)},
#                         height_ratios=[1,2,2])
ax = fig.subplot_mosaic([[str(i)+"_tc" for i in range(9)],[0,0,0,1,1,1,2,2,2],["means","means","means","stds","stds","stds","power","power","power"]],
                        height_ratios=[1,2,2])
for i in range(9):
    ax[str(i)+"_tc"].plot(theta,Ymult[to_display[i],:],color="darkviolet")
    # ax[str(i) + "_tc"].set_xticks([0,360])
    ax[str(i) + "_tc"].set_xlabel("HD")
    ax[str(i)+"_tc"].plot(theta,Ydelta[to_display[i],:],color="violet")
    ax[str(i)+"_tc"].plot(theta,Ydelta[to_display[i], :]+Ymult[to_display[i],:],color="black")
    ax[str(i)+"_tc"].set_ylim(0,80)
    ax[str(i) + "_tc"].set_yticks([0,40,80])
    ax[str(i) + "_tc"].set_xticks([])
ax[str(0)+"_tc"].set_ylabel("Firing rate (Hz)")
# ax[str(0)+"_tc"].set_yticks([0,25,50,75])
ax[0].hist(np.mean(Ydelta,axis=-1),bins=np.arange(0,60,0.5),color="violet",histtype="step",label="AD inputs")
# ax[0].vlines(np.std(np.mean(tcpyr_ad_shift,axis=0)),0,100,color="black")
ax[0].hist(np.mean(Ymult,axis=-1),bins=np.arange(0,60,0.5),color="darkviolet",label="PoSub inputs")
ax[0].set_xlabel("mean of the contributions")
ax[1].hist(np.std(Ydelta,axis=-1),bins=np.arange(0,60,0.5),color="violet",histtype="step",label="pyr - AD")
# ax[0].vlines(np.std(np.mean(tcpyr_ad_shift,axis=0)),0,100,color="black")
ax[1].hist(np.std(Ymult,axis=-1),bins=np.arange(0,60,0.5),color="darkviolet",label="pyr - PoSub")
ax[1].set_xlabel("std of the contributions")
# ax[0].vlines(np.std(np.mean(tcpyr_pos_shift,axis=0)),0,100,color="blue")
ax[2].hist(np.std(Ydelta,axis=-1)/np.mean(Ydelta,axis=-1),bins=np.arange(0,2,0.05),color="violet",histtype="step")
ax[2].hist(np.std(Ymult,axis=-1)/np.mean(Ymult,axis=-1),bins=np.arange(0,2,0.05),color="darkviolet")
ax[2].set_xlabel("std/mean of the contributions")

ax["means"].hist(np.mean(tcfs_pos,axis=-1),bins=np.arange(0,60,0.5),color="black",histtype="step",label="Measured TCs",density=True)
ax["means"].hist(np.mean(Y_off,axis=-1),bins=np.arange(0,60,0.5),color="grey",label="Simulated TCs",density=True)
ax["means"].set_yticks([0,0.05,0.1,0.150])
ax["means"].set_ylabel("density")
ax["means"].set_xlabel("firing rate mean")

ax["stds"].hist(np.std(tcfs_pos,axis=-1),bins=np.arange(0,60,0.5),color="black",histtype="step",label="Measured TCs",density=True)
ax["stds"].hist(np.std(Y_off,axis=-1),bins=np.arange(0,60,0.5),color="grey",label="Simulated TCs",density=True)
ax["stds"].set_yticks([0,0.15,0.3])
ax["stds"].set_ylabel("density")
ax["stds"].set_xlabel("firing rate standard deviations")

ax["power"].errorbar(range(k_fourier),np.std(tcfs_pos@np.transpose(mFourier_basis),axis=0),yerr=noise_fspos["sigma_l"][0,:k_fourier],color="black",label="Measured TCs")
ax["power"].plot(np.std(Y_off@np.transpose(mFourier_basis),axis=0),color="grey",label="Simulated TCs")
ax["power"].set_xlabel("Fourier component index")
ax["power"].set_ylabel("standard deviation of the Fourier component \n across tuning curves")

ax[1].legend(loc="upper right")
ax[0].set_ylabel("number of simulated cells")
for i in range(3):
    ax[i].set_ylim(0,1000)
fig.tight_layout()
fig.show()
