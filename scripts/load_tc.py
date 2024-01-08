import os
import scipy.io

path = os.path.join(os.path.abspath(__file__),"..","..","data")

# tcpyr_ad = pandas.read_csv(os.path.join(path,"TcHDadn_Smooth.csv"),header=None).values.transpose()
# scipy.io.savemat(os.path.join(path,"TcHDadn_Smooth.mat"),{"tc":tcpyr_ad})
# tcpyr_ad_raw = pandas.read_csv(os.path.join(path,"TcHDadn_NoSmooth.csv"),header=None).values.transpose()
# scipy.io.savemat(os.path.join(path,"TcHDadn_NoSmooth.mat"),{"tc":tcpyr_ad_raw})
# tcpyr_pos = pandas.read_csv(os.path.join(path,"TcHDpos_Smooth.csv"),header=None).values.transpose()
# scipy.io.savemat(os.path.join(path,"TcHDpos_Smooth.mat"),{"tc":tcpyr_pos})
# tcpyr_pos_raw = pandas.read_csv(os.path.join(path,"TcHDpos_NoSmooth.csv"),header=None).values.transpose()
# scipy.io.savemat(os.path.join(path,"TcHDpos_NoSmooth.mat"),{"tc":tcpyr_pos_raw})
# tcfs_pos = pandas.read_csv(os.path.join(path,"TcFS_Smooth.csv"),header=None).values.transpose()
# scipy.io.savemat(os.path.join(path,"TcFS_Smooth.mat"),{"tc":tcfs_pos})
# tcfs_pos_raw = pandas.read_csv(os.path.join(path,"TcFS_NoSmooth.csv"),header=None).values.transpose()
# scipy.io.savemat(os.path.join(path,"TcFS_NoSmooth.mat"),{"tc":tcfs_pos_raw})

tcpyr_ad = scipy.io.loadmat(os.path.join(path,"TcHDadn_Smooth.mat"))["tc"]
tcpyr_ad_raw = scipy.io.loadmat(os.path.join(path,"TcHDadn_NoSmooth.mat"))["tc"]
tcpyr_pos = scipy.io.loadmat(os.path.join(path,"TcHDpos_Smooth.mat"))["tc"]
tcpyr_pos_raw = scipy.io.loadmat(os.path.join(path,"TcHDpos_NoSmooth.mat"))["tc"]
tcfs_pos = scipy.io.loadmat(os.path.join(path,"TcFS_Smooth.mat"))["tc"]
tcfs_pos_raw = scipy.io.loadmat(os.path.join(path,"TcFS_NoSmooth.mat"))["tc"]