from load_tc import *
from fourierbasis_hd import *
import tqdm
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import scipy.optimize

path = os.path.join(os.path.abspath(__file__),"..","..","data")


## evolution of the std of the error with the number of subsampled cells:
def stdpvFourier_bootstrap(fourier_coeff,max_nbkept):
    stds_error_pyr = []
    keep_array = np.arange(20,max_nbkept)
    for nb_kept in tqdm.tqdm(keep_array):
        ids = np.random.randint(0, fourier_coeff.shape[0], (1000, nb_kept))
        x = np.nanstd(fourier_coeff[ids, :], axis=1)
        stds_error_pyr += [np.nanstd(np.log(x), axis=0)]
    stds_error_pyr = np.stack(stds_error_pyr)
    return stds_error_pyr

fourier_coeff_pospyr = tcpyr_pos@np.transpose(Fourier_basis)
fourier_coeff_fspyr = tcfs_pos@np.transpose(Fourier_basis)
fourier_coeff_adpyr = tcpyr_ad@np.transpose(Fourier_basis)

stds_error_pyr = stdpvFourier_bootstrap(fourier_coeff_pospyr,fourier_coeff_fspyr.shape[0])
stds_error_fs = stdpvFourier_bootstrap(fourier_coeff_pospyr,fourier_coeff_fspyr.shape[0])
stds_error_pyrad = stdpvFourier_bootstrap(fourier_coeff_adpyr,fourier_coeff_adpyr.shape[0])

fourier_coeff_adpospyr = np.concatenate([fourier_coeff_adpyr,fourier_coeff_pospyr],axis=0)
stds_error_pyradpos = stdpvFourier_bootstrap(fourier_coeff_adpospyr,fourier_coeff_fspyr.shape[0])

def decay_curve(x,p):
    return p/np.sqrt(x)
keep_array = np.arange(20,fourier_coeff_fspyr.shape[0])
sigma_l_pyr_pos = np.concatenate([scipy.optimize.curve_fit(decay_curve,keep_array,stds_error_pyr[:,i])[0]
                          for i in range(stds_error_pyr.shape[-1])])
sigma_l_fs_pos = np.concatenate([scipy.optimize.curve_fit(decay_curve,keep_array,stds_error_fs[:,i])[0]
                          for i in range(stds_error_fs.shape[-1])])

keep_array = np.arange(20,fourier_coeff_fspyr.shape[0])
sigma_l_pyr_adpos = np.concatenate([scipy.optimize.curve_fit(decay_curve,keep_array,stds_error_pyradpos[:,i])[0]
                          for i in range(stds_error_fs.shape[-1])])

keep_array = np.arange(20,fourier_coeff_adpyr.shape[0])
sigma_l_pyr_ad = np.concatenate([scipy.optimize.curve_fit(decay_curve,keep_array,stds_error_pyrad[:,i])[0]
                          for i in range(stds_error_pyrad.shape[-1])])

path_result = os.path.join(path,"analysis_results")
scipy.io.savemat(os.path.join(path_result,"noise_pyrpos.mat"),{"sigma_l":sigma_l_pyr_pos,
                                                               "last_size":fourier_coeff_fspyr.shape[0],
                                                               "stds":stds_error_pyr})
scipy.io.savemat(os.path.join(path_result,"noise_fspos.mat"),{"sigma_l":sigma_l_fs_pos,
                                                              "last_size":fourier_coeff_fspyr.shape[0],
                                                               "stds":stds_error_fs})
scipy.io.savemat(os.path.join(path_result,"noise_pyrad.mat"),{"sigma_l":sigma_l_pyr_ad,
                                                                "last_size":fourier_coeff_adpyr.shape[0],
                                                               "stds":stds_error_pyrad})

scipy.io.savemat(os.path.join(path_result,"noise_pyrposad.mat"),{"sigma_l":sigma_l_pyr_adpos,
                                                                "last_size":fourier_coeff_fspyr.shape[0],
                                                               "stds":stds_error_pyradpos})