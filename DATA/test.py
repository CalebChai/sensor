import numpy as np

class Data:
    def __init__(self,time,index,data):
        self.time =time
        self.index=index
        self.data = data
    
    def __repr__(self):
        return f"{self.time},{self.index},{self.data}"
    
class CalibData:
    def __init__(self, time, data):
        self.time = time
        self.data = data

    def __repr__(self):
        return f"{self.time}, {self.data}"

CALIB_AXIS = [0,1,2,3,4,5]

class _Dataset:
    def __init__(self, count):
        self.data_name = f"DATA/data_log{count}.csv"
        self.calib_name = f"DATA/calib{count}.txt"
        self.time_bias = 5.246235246235244
        # self.time_bias = 3.194953194953193
        self._raw_start_time = None
        self._calib_start_time = None
        sum_raw = 0  # 初始化sum_raw为
        sum_calib=0
        with open(self.data_name, 'r') as f:
            data = f.readlines()
            # remove the first line
            data = data[1:]

            self._raw_data = []
            for line in data:
                dd = line.strip().split(',')
                raw_data = np.array(dd[2:10],dtype=float)
                if self._raw_start_time is None:
                    self._raw_start_time = float(dd[0])
                self._raw_data.append(Data((float(dd[0])-self._raw_start_time)/ 1000000,dd[1],raw_data))
                sum_raw += np.abs(raw_data).sum()  # 累加raw_data的绝对值和
            print("raw:",sum_raw)
        
        with open(self.calib_name, 'r') as f:
            calib = f.readlines()
            self._calib_data = []
            for line in calib:
                dd = line.strip().split(' ')
                calib_data = np.array(dd[1:],dtype=float)/6
                if self._calib_start_time is None:
                    self._calib_start_time = float(dd[0])
                self._calib_data.append(CalibData(float(dd[0])-self._calib_start_time + self.time_bias,calib_data))
                sum_calib += np.abs(raw_data).sum()  # 累加raw_data的绝对值和
            print("calib:",sum_calib)
        
        # self._raw = np.array([d.data for d in self._raw_data])
        # self._raw_time = np.array([d.time for d in self._raw_data]) / 1000000
        # self._calib = np.array([d.data for d in self._calib_data])[::20,:] * 0.02
        # self._calib_time = np.array([d.time for d in self._calib_data],dtype=float)[::20] + self.time_bias

    def get_data(self):
        raw_data = []
        calib_data = []
        calib_current_index = 0

        for d in self._raw_data:
            while True:
                if calib_current_index+1 >= len(self._calib_data):
                    break
                time1, time2 = self._calib_data[calib_current_index].time, self._calib_data[calib_current_index+1].time
                if d.time < time1:
                    break
                elif d.time > time2:
                    calib_current_index += 1
                elif d.time >= time1 and d.time < time2:
                    sum1 = np.sum(np.abs(d.data))
                    sum2 = np.sum(np.abs(self._calib_data[calib_current_index].data))
                    if sum2 < 1:
                        break
                    raw_data.append(d.data)
                    calib_data.append(self._calib_data[calib_current_index].data[CALIB_AXIS])
                    break
        return np.array(raw_data), np.array(calib_data)*[2,2,1.0,5,5,5]+[0.5,0,1,0,0,0]

def draw(raw_data, calib_data, * , axis = None):
    import matplotlib.pyplot as plt

    
    if axis is not None:
        fig, axs = plt.subplots(1, len(axis))
        for i,index in enumerate(axis):
            # random a color
            axs[i].plot(raw_data[:,index], label=f"raw{index}")
            axs[i].plot(calib_data[:,index], label=f"calib{index}")
            axs[i].set_xlabel('Sample Index')
            axs[i].set_ylabel(f'Value {index}')
            axs[i].legend()
    else:
        fig, ax = plt.subplots()
        for i in range(raw_data.shape[1]):
            ax.plot(raw_data[:,2], label=f"raw{i}")
        for i in range(calib_data.shape[1]):
            ax.plot(calib_data[:,2], label=f"calib{i}")
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')
        ax.legend()

    plt.show()

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class SensorDataset(Dataset):
    def __init__(self):
        raw_data = np.empty((0,8))
        calib_data = np.empty((0,len(CALIB_AXIS)))
        for i in range(5):
            dataset = _Dataset(i+1)
            raw, calib = dataset.get_data()
            raw_data = np.vstack((raw_data, raw))
            calib_data = np.vstack((calib_data, calib))
            print(f"raw shape: {raw.shape}, calib shape: {calib.shape}")
        print(f"raw_data shape: {raw_data.shape}, calib_data shape: {calib_data.shape}")
        self.raw_data = torch.tensor(raw_data, dtype=torch.float32)
        self.calib_data = torch.tensor(calib_data*0.3, dtype=torch.float32)

        # draw(self.raw_data, self.calib_data)
    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        return self.raw_data[idx], self.calib_data[idx]

class SensorModel(nn.Module):
    def __init__(self):
        super(SensorModel, self).__init__()
        self.bias = nn.Parameter(torch.randn(8))
        # self.mx = nn.Parameter(torch.randn(8,len(CALIB_AXIS)))
        self.mx = nn.Parameter(torch.randn(8,3))
    
    def forward(self, x):
        y1 = x[:,:4] @ self.mx[:4,:]
        y2 = x[:,4:] @ self.mx[4:,:]
        return torch.cat((y1,y2),1)
#############################  
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(8,6)
    def forward(self, x):
        return self.linear(x)
    
model = LinearModel()
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

dataset = _Dataset(18)
raw_data, calib_data = dataset.get_data()
raw_data = torch.tensor(raw_data, dtype=torch.float32)
calib_data = torch.tensor(calib_data, dtype=torch.float32)
#x_data1 is 1st column of raw_data
val = [[-0.34742835, -0.2943973, -0.16368383, -0.35832116,  1.2746422,  -1.4914296,
        -0.81430334,  1.1594403],
       [ 2.1484637,   2.1213264,  1.7846724,   2.1020029,  -1.4098333,   1.4123662,
         0.57387257, -0.765478],
       [-0.21408577,  0.06317329, -0.02131992,  0.2611141,   0.10286304, -0.78217345,
         0.83403355, -0.24016705],
       [-1.9560765,  -1.9455622,  -1.8497026,  -1.934165,    1.4082885,  -1.5507714,
        -0.04700585,  0.33311838],
       [-0.15666199, -0.30654615, -0.01919471, -0.1398079,   1.1371785,  -1.62595,
        -0.37701395,  0.9703038],
       [-0.05281514, -0.03802285, -0.05535824, -0.00309375, -0.48457462, -0.22240411,
        -0.05040386,  0.4999259]]
val_tensor = torch.tensor(val, dtype=torch.float32)  # 将 val 转换为张量
print("raw_data shape:", raw_data.shape)
print("val_tensor shape:", val_tensor.shape)
val_tensor_transposed = val_tensor.T

# 进行矩阵乘法，结果的形状将是 [133, 6]
result = torch.matmul(raw_data, val_tensor_transposed)
addition_vector = torch.tensor([0.3399, 0.2625, 0.9781, -0.0413, 0.0100, 0.0782], dtype=torch.float32)

# 使用广播机制，将 addition_vector 添加到 result 的每一行

# x_data1 = raw_data[:,0]*100
# #x_data2 is 4th column of raw_data
# x_data2 = raw_data[:,3]*100
# #x_data is x_data1 and x_data2
# x_data = torch.stack((x_data1, x_data2), 1)
x_data = result*100+addition_vector

# #change x_data to 2D tensor
# x_data = x_data.view(-1,1)
#y_data is first column of calib_data
# y_data = calib_data[:,5]
# #change y_data to 2D tensor
# y_data = y_data.view(-1,1)

y_data = calib_data

draw(x_data, y_data)
# 
print(f"x_data shape: {x_data.shape}, y_data shape: {y_data.shape}")