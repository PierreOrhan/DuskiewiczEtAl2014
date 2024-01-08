import scipy.io
import os

path = os.path.join(os.path.abspath(__file__),"..","..","data")
path_result = os.path.join(path,"analysis_results")
noise_pyrpos = scipy.io.loadmat(os.path.join(path_result,"noise_pyrpos.mat"))
noise_fspos = scipy.io.loadmat(os.path.join(path_result,"noise_fspos.mat"))
noise_pyrad = scipy.io.loadmat(os.path.join(path_result,"noise_pyrad.mat"))
noise_pyrposad = scipy.io.loadmat(os.path.join(path_result,"noise_pyrposad.mat"))