import numpy as np
import scipy.signal

theta = np.linspace(0,2*np.pi,361)[0:360]
Fourier_basis = np.zeros((100,theta.shape[0]))
for i in range(50):
    Fourier_basis[2*i,:] = np.cos((i+1)*theta)
    Fourier_basis[2*i+1,:] = np.sin((i+1)*theta)
Fourier_basis = Fourier_basis/np.sqrt(360/2)

def get_Fourier_power(Y):
    tc_fourier = Y @ np.transpose(Fourier_basis)
    power_fs = np.stack([tc_fourier[..., i] ** 2 + tc_fourier[..., i + 1] ** 2 for i in
                               np.arange(0, tc_fourier.shape[-1], 2)], axis=-1)
    power_fs_variancefree = power_fs / np.sum(power_fs, axis=-1)[..., None]
    return power_fs,power_fs_variancefree

def get_acc(Y):
    zY = (Y - np.mean(Y, axis=-1)[..., None]) / (np.std(Y, axis=-1)[..., None])
    zY2 = np.concatenate([zY, zY], axis=-1)
    Yacc = scipy.signal.fftconvolve(zY2, zY[..., ::-1], mode="valid", axes=(-1)) / zY.shape[1]
    return Yacc

def get_propotion(Y):
    # ratio
    Yfourier = Y @ np.transpose(Fourier_basis)
    Ypower = Yfourier[..., ::2] ** 2 + Yfourier[..., 1::2] ** 2
    Ypower_max = np.argmax(Ypower, axis=-1)
    Yproportion = np.concatenate([np.sum(np.equal(Ypower_max, i),axis=-1)[...,None] for i in range(4)],axis=-1)
    return Yproportion/Y.shape[-2]

def mi_tc(Y):
    # mutual information by assuming a uniform probability of the head-direction.
    return np.mean(Y*np.log2(Y/(np.mean(Y,axis=-1)[...,None])),axis=-1)

def mi_tc_bitrate(Y):
    # mutual information by assuming a uniform probability of the head-direction.
    return np.mean(Y/(np.mean(Y,axis=-1)[...,None])*np.log2(Y/(np.mean(Y,axis=-1)[...,None])),axis=-1)

def snr_tc(Y):
    return np.std(Y,axis=-1)/np.mean(Y,axis=-1)


def e_LN(ew,σ2w):
    return np.log(ew**2/(np.sqrt(σ2w+ew**2)))
def σ2_LN(ew,σ2w):
    return np.log(σ2w/(ew**2)+1)
#Log normal from the target μ and σ:
def LogNormal2(μ,σ,n):
    return np.random.lognormal(e_LN(μ,σ**2),np.sqrt(σ2_LN(μ,σ**2)),n)

def uniform2(μ,σ,n):
    return  np.random.uniform(μ-σ*np.sqrt(3),μ+σ*np.sqrt(3),n)

def symscore(Y,nb_comp=3):
    _,power_fs_variancefree =get_Fourier_power(Y)
    s1 = nb_comp*np.max(power_fs_variancefree[...,:nb_comp],axis=-1)
    return (s1-1)/(nb_comp-1)

import uncertainties
from uncertainties import unumpy

def log_kl_uncertainty(X_mean,X_stds,target_mean,target_std):
    # X_mean: 1D array the mean of the tuning curve mutual information for each repetition
    # X_stds: 1D array the stds of the tuning curve mutual information for each repetition

    usnr_mean = unumpy.uarray(np.mean(X_mean,axis=0),np.std(X_mean,axis=0))
    usnr_std = unumpy.uarray(np.mean(X_stds,axis=0),np.std(X_stds,axis=0))
    # kl = KL_div(usnr_mean, target_mean, usnr_std, target_std)
    klu = unumpy.log(target_std/usnr_std)+(usnr_std**2+(usnr_mean-target_mean)**2)/(2*target_std**2)-1/2
    logklu = unumpy.log(klu)
    nominal = unumpy.nominal_values(logklu)
    stdev = unumpy.std_devs(logklu)
    return nominal,stdev
