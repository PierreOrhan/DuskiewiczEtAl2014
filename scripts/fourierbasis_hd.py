# Code to generate a discretized Fourier basis
import numpy as np

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
