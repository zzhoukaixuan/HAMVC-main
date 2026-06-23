from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import h5py
from pathlib import Path

def normalize(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

import h5py
class Caltech256():
    def __init__(self, path):
        data = h5py.File(path + 'Caltech256_fea.mat', 'r')
        self.Y = np.squeeze(np.array(data['Y'])).astype(np.int32)
        self.V1 = np.array(np.transpose(data[data['X'][0][0]])).astype(np.float32)
        self.V2 = np.array(np.transpose(data[data['X'][1][0]])).astype(np.float32)
        self.V3 = np.array(np.transpose(data[data['X'][2][0]])).astype(np.float32)
    def __len__(self):
        return 30607
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class cifar_10():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class cifar_100():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)],self.Y[idx], torch.from_numpy(np.array(idx)).long()

class synthetic3d():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'synthetic3d.mat')
        self.Y = data['Y'].astype(np.int32).reshape(600,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 600
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
               self.Y[idx], torch.from_numpy(np.array(idx)).long()

class prokaryotic():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'prokaryotic.mat')
        self.Y = data['Y'].astype(np.int32).reshape(551,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 551
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
               self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class Hdigit():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][0][1].T.astype(np.float32)
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class STL10():
    def __init__(self, path):
        data = h5py.File(path + 'stl10_fea.mat', 'r')
        scaler = MinMaxScaler()
        self.Y = np.squeeze(np.array(data['Y'])).astype(np.int32)
        self.V1 = scaler.fit_transform(np.array(np.transpose(data[data['X'][0][0]])).astype(np.float32))
        self.V2 = scaler.fit_transform(np.array(np.transpose(data[data['X'][1][0]])).astype(np.float32))
        self.V3 = scaler.fit_transform(np.array(np.transpose(data[data['X'][2][0]])).astype(np.float32))

    def __len__(self):
        return 13000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy((x1)), torch.from_numpy((x2)), torch.from_numpy((x3))], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class ALOI():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'ALOI.mat')
        self.Y = np.squeeze(data['Y']).astype(np.int32)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][0][2].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][0][3].astype(np.float32))

    def __len__(self):
        return 1079
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        return [torch.from_numpy((x1)), torch.from_numpy((x2)), torch.from_numpy((x3)), torch.from_numpy((x4))], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class ALOI100():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'ALOI100.mat')
        self.Y = np.squeeze(data['Y']).astype(np.int32)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][0][2].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][0][3].astype(np.float32))
    def __len__(self):
        return 10800
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        return [torch.from_numpy((x1)), torch.from_numpy((x2)), torch.from_numpy((x3)), torch.from_numpy((x4))], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Leaves():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '100leaves.mat')
        self.Y = data['Y'].astype(np.int32).reshape(1600, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)

    def __len__(self):
        return 1600
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    
class BBC4view():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'BBC4view_685.mat')
        self.Y = data['Y'].astype(np.int32).reshape(685, )
        self.V1 = data['X'][0][0].A.astype(np.float32)
        self.V2 = data['X'][0][1].A.astype(np.float32)
        self.V3 = data['X'][0][2].A.astype(np.float32)
        self.V4 = data['X'][0][3].A.astype(np.float32)
    def __len__(self):
        return 685
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class HW2sources():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'HW2sources.mat')
        self.Y = data['Y'].astype(np.int32).reshape(2000, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
    def __len__(self):
        return 2000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class LandUse():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'LandUse-21.mat')
        self.Y = np.squeeze(data['Y']).astype(np.int32) 
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
    def __len__(self):
        return 2100
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    
class BDGP():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'BDGP.mat')
        self.Y = data['Y'].T.astype(np.int32).reshape(2500,)
        self.V1 = data['X1'].astype(np.float32)
        self.V2 = data['X2'].astype(np.float32)
    def __len__(self):
        return 2500
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    
class BBCSport():
    def __init__(self, path):
        data = h5py.File(path + 'BBCSport.mat')
        self.Y = np.squeeze(np.array(data['Y'])).astype(np.int32)
        self.V1 = np.array(data[data['X'][0][0]]).T.astype(np.float32)
        self.V2 = np.array(data[data['X'][0][1]]).T.astype(np.float32)
    def __len__(self):
        return 544
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class MSRCV1():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'MSRCV1.mat')
        self.Y = data['Y'].astype(np.int32).reshape(210, )
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(data['X'][0][0].astype('float32'))
        self.V2 = scaler.fit_transform(data['X'][0][1].astype('float32'))
        self.V3 = scaler.fit_transform(data['X'][0][2].astype('float32'))
        self.V4 = scaler.fit_transform(data['X'][0][3].astype('float32'))
        self.V5 = scaler.fit_transform(data['X'][0][4].astype(np.float32))
        self.V6 = scaler.fit_transform(data['X'][0][5].astype(np.float32))

    def __len__(self):
        return 210
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx] 
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy((x1)), torch.from_numpy((x2)), torch.from_numpy((x3)),
                torch.from_numpy((x4)), torch.from_numpy((x5)), torch.from_numpy((x6))], self.Y[idx], torch.tensor(idx, dtype=torch.long)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def add_noise(data, noise_level=0.9, seed=10):
    if noise_level == 0:
        return data

    np.random.seed(seed)
    noise = np.random.normal(loc=0.0, scale=noise_level, size=data.shape)
    noisy_data = data + noise
    return noisy_data

class HW():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'handwritten.mat')
        self.Y = data['Y'].astype(np.int32).reshape(2000, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)
        self.V5 = data['X'][4][0].astype(np.float32)
        self.V6 = data['X'][5][0].astype(np.float32)

    def __len__(self):
        return 2000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [add_noise(torch.from_numpy((x1))), add_noise(torch.from_numpy((x2))), add_noise(torch.from_numpy((x3))),
                add_noise(torch.from_numpy((x4))), add_noise(torch.from_numpy((x5))), add_noise(torch.from_numpy((x6)))], self.Y[idx], torch.tensor(idx, dtype=torch.long)
 
class UCIDigit():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'uci-digit.mat')
        scaler = MinMaxScaler()
        self.Y = data['Y'].reshape(-1).astype(np.int64)
        self.V1 = scaler.fit_transform(data['mfeat_fac'].astype(np.float32))
        self.V2 = data['mfeat_fou'].astype(np.float32)
        self.V3 = scaler.fit_transform(data['mfeat_kar'].astype(np.float32))

    def __len__(self):
        return 2000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [(torch.from_numpy((x1))), (torch.from_numpy((x2))), (torch.from_numpy((x3)))], self.Y[idx], torch.tensor(idx, dtype=torch.long)

class NoisyMNIST(Dataset):
    def __init__(self, path):
        mat = scipy.io.loadmat(path + 'NoisyMNIST.mat') 
        self.X1 = mat['X1'].astype(np.float32)
        self.X2 = mat['X2'].astype(np.float32)
        self.labels = np.squeeze(mat['trainLabel']).astype(np.int64)
        self.views = [self.X1, self.X2]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        x1_tensor = torch.from_numpy(self.views[0][idx])
        x2_tensor = torch.from_numpy(self.views[1][idx])
        label_tensor = torch.tensor(self.labels[idx]).long()
        idx_tensor = torch.tensor(idx).long()
        
        return [x1_tensor, x2_tensor], label_tensor, idx_tensor

class NoisyMNIST_15000(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'NoisyMNIST_15000.mat')
        self.Y = data['Y'].astype(np.int64).reshape(-1)
        self.V1 = data['X'][0][0].astype(np.float32) 
        self.V2 = data['X'][0][1].astype(np.float32)
    def __len__(self):
        return self.Y.shape[0]
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.tensor(idx, dtype=torch.long)

class Webkb():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'webkb.mat')
        self.Y = data['Y'].astype(np.int32).reshape(203, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
    def __len__(self):
        return 203
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.tensor(idx, dtype=torch.long)
    
class Mfeat():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Mfeat.mat')
        scaler = MinMaxScaler()
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(-1)
        self.V1 = scaler.fit_transform(data['data'][0][0].astype(np.float32).T)
        self.V2 = data['data'][0][1].astype(np.float32).T
        self.V3 = scaler.fit_transform(data['data'][0][2].astype(np.float32).T)
        self.V4 = scaler.fit_transform(data['data'][0][3].astype(np.float32).T)
        self.V5 = scaler.fit_transform(data['data'][0][4].astype(np.float32).T)
        self.V6 = scaler.fit_transform(data['data'][0][5].astype(np.float32).T)

    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), 
                torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.tensor(idx, dtype=torch.long)

class Cifar10():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000, )
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.tensor(idx, dtype=torch.long)
    
class Animal(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Animal.mat')
        scaler = MinMaxScaler()
        self.Y = np.squeeze(data['Y']).astype(np.int32)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = scaler.fit_transform(data['X'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][0][2].astype(np.float32))
        self.V4 = data['X'][0][3].astype(np.float32)

    def __len__(self):
        return 11673

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4)], self.Y[idx], torch.tensor(idx, dtype=torch.long)

class Scene15():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Scene15.mat')
        self.Y = data['Y'].astype(np.int32).reshape(4485, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 4485
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.tensor(idx, dtype=torch.long)

class handwritten(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'handwritten.mat')
        scaler = MinMaxScaler()
        self.Y = data['truth'].reshape(-1).astype(np.int64)
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = scaler.fit_transform(data['X'][0][2].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X'][0][3].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X'][0][4].astype(np.float32))
        self.V6 = scaler.fit_transform(data['X'][0][5].astype(np.float32))

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),
            torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.tensor(idx, dtype=torch.long)

class YoutubeFace(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'YoutubeFace_sel_fea.mat')
        self.Y = data['Y'].astype(np.int64).reshape(-1)
        
        self.V1 = data['X'][0, 0].astype(np.float32)
        self.V2 = data['X'][1, 0].astype(np.float32)
        self.V3 = data['X'][2, 0].astype(np.float32)
        self.V4 = data['X'][3, 0].astype(np.float32)
        self.V5 = data['X'][4, 0].astype(np.float32)
        

        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(self.V1)
        self.V2 = scaler.fit_transform(self.V2)
        self.V3 = scaler.fit_transform(self.V3)
        self.V4 = scaler.fit_transform(self.V4)
        self.V5 = scaler.fit_transform(self.V5)

    def __len__(self):
        return 101499

    def __getitem__(self, idx):
        return [
            torch.from_numpy(self.V1[idx]),
            torch.from_numpy(self.V2[idx]),
            torch.from_numpy(self.V3[idx]),
            torch.from_numpy(self.V4[idx]),
            torch.from_numpy(self.V5[idx])
        ], torch.tensor(self.Y[idx]).long(), torch.tensor(idx).long()

class fmnist(Dataset):
    def __init__(self, path):
        with h5py.File(path + 'fmnist_fea.mat', 'r') as data:
            self.Y = np.array(data['Y']).astype(np.int64).squeeze()
            if self.Y.min() == 1:
                self.Y = self.Y - 1
            self.V1 = np.array(data[data['X'][0][0]]).T.astype(np.float32)
            self.V2 = np.array(data[data['X'][0][1]]).T.astype(np.float32)
            self.V3 = np.array(data[data['X'][0][2]]).T.astype(np.float32)
        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(self.V1)
        self.V2 = scaler.fit_transform(self.V2)
        self.V3 = scaler.fit_transform(self.V3)

    def __len__(self):
        return 60000

    def __getitem__(self, idx):
        return [
            torch.from_numpy(self.V1[idx]),
            torch.from_numpy(self.V2[idx]),
            torch.from_numpy(self.V3[idx])
        ], torch.tensor(self.Y[idx]).long(), torch.tensor(idx).long()
    
class MNIST(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'MNIST_fea.mat')
        self.Y = data['Y'].astype(np.int64).squeeze()
        if self.Y.min() == 1:
            self.Y = self.Y - 1
            
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)

        scaler = MinMaxScaler()
        self.V1 = scaler.fit_transform(self.V1)
        self.V2 = scaler.fit_transform(self.V2)
        self.V3 = scaler.fit_transform(self.V3)

    def __len__(self):
        return 60000

    def __getitem__(self, idx):
        return [
            torch.from_numpy(self.V1[idx]),
            torch.from_numpy(self.V2[idx]),
            torch.from_numpy(self.V3[idx])
        ], torch.tensor(self.Y[idx]).long(), torch.tensor(idx).long()

def load_data(dataset, data_root="./datasets"):
    data_root = f"{Path(data_root).expanduser().resolve()}/"

    if dataset == "BDGP":
        dataset = BDGP(data_root)
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS(data_root)
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "Hdigit":
        dataset = Hdigit(data_root)
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    elif dataset == "NoisyMNIST":
        dataset = NoisyMNIST(data_root)
        dims = [784, 784]
        view = 2
        data_size = 50000
        class_num = 10
    elif dataset == "NoisyMNIST_15000":
        dataset = NoisyMNIST_15000(data_root)
        dims = [784, 784]
        view = 2
        data_size = 15000
        class_num = 10
    elif dataset == "CCV":
        dataset = CCV(data_root)
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Animal":
        dataset = Animal(data_root)
        dims = [2689, 2000, 2001, 2000]
        view = 4
        class_num = 20
        data_size = 11673
    elif dataset == "Fashion":
        dataset = Fashion(data_root)
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == 'Scene15':
        dataset = Scene15(data_root)
        dims = [20, 59, 40]
        view = 3
        data_size = 4485
        class_num = 15
    elif dataset == "Synthetic3d":
        dataset = synthetic3d(data_root)
        dims = [3,3,3]
        view = 3
        data_size = 600
        class_num = 3
    elif dataset == 'BBC4view':
        dataset = BBC4view(data_root)
        dims = [4659, 4633, 4665, 4684]
        view = 4
        data_size = 685
        class_num = 5
    elif dataset == "MSRCV1":
        dataset = MSRCV1(data_root)
        dims = [1302, 48, 512, 100, 256, 210]
        view = 6
        data_size = 210
        class_num = 7
    elif dataset == 'BBCSport':
        dataset = BBCSport(data_root)
        dims = [3183, 3203]
        view = 2
        data_size = 544
        class_num = 5
    elif dataset == "Prokaryotic":
        dataset = prokaryotic(data_root)
        dims = [438, 3, 393]
        view = 3
        data_size = 551
        class_num = 4
    elif dataset == 'Cifar10':
        dataset = Cifar10(data_root)
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10
    elif dataset == 'Caltech256':
        dataset = Caltech256(data_root)
        dims = [1024, 512, 2048]
        view = 3
        data_size = 30607
        class_num = 257
    elif dataset == 'UCIDigit':
        dataset = UCIDigit(data_root)
        dims = [216, 76, 64]
        view = 3
        data_size = 2000
        class_num = 10
    elif dataset == "handwritten":
        dataset = handwritten(data_root)
        dims = [240, 76, 216, 47, 64, 6]
        view = 6
        data_size = 2000
        class_num = 10
    elif dataset == 'Mfeat':
        dataset = Mfeat(data_root)
        dims = [216, 76, 64, 6, 240, 47]
        view = 6
        data_size = 2000
        class_num = 10
    elif dataset == "ALOI100":
        dataset = ALOI100(data_root)
        dims = [77,13,64,125]
        view = 4
        data_size = 10800
        class_num = 100
    elif dataset == 'STL10':
        dataset = STL10(data_root)
        dims = [1024, 512, 2048]
        view = 3
        data_size = 13000
        class_num = 10
    elif dataset == "ALOI":
        dataset = ALOI(data_root)
        dims = [64, 64, 77, 13]
        view = 4
        data_size = 1079
        class_num = 10
    elif dataset == "Leaves":
        dataset = Leaves(data_root)
        dims = [64, 64, 64]
        view = 3
        data_size = 1600
        class_num = 100
    elif dataset == "Caltech-2V":
        dataset = Caltech(data_root + 'Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech(data_root + 'Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech(data_root + 'Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech(data_root + 'Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "Cifar100":
        dataset = cifar_100(data_root)
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 100
    elif dataset == 'HW2sources':
        dataset = HW2sources(data_root)
        dims = [784, 256]
        view = 2
        data_size = 2000
        class_num = 10
    elif dataset == 'LandUse':
        dataset = LandUse(data_root)
        dims = [20, 59, 40]
        view = 3
        data_size = 2100
        class_num = 21
    elif dataset == 'Webkb':
        dataset = Webkb(data_root)
        dims = [1703, 230, 230]
        view = 3
        data_size = 203
        class_num = 4
    elif dataset == "YoutubeFace":
        dataset = YoutubeFace(data_root)
        dims = [64, 512, 64, 647, 838]
        view = 5
        data_size = 101499
        class_num = 31
    elif dataset == "fmnist":
        dataset = fmnist(data_root)
        dims = [512, 512, 1280]
        view = 3
        data_size = 60000
        class_num = 10
    elif dataset == "MNIST":
        dataset = MNIST(data_root)
        dims = [342, 1024, 64]
        view = 3
        data_size = 60000
        class_num = 10

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return dataset, dims, view, data_size, class_num
