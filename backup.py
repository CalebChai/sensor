import numpy as np

class Data:
    def __init__(self, time, index, data):
        self.time = time
        self.index = index
        self.data = data
    
    def __repr__(self):
        return f"{self.time}, {self.index}, {self.data}"

class CalibData:
    def __init__(self, time, data):
        self.time = time
        self.data = data

    def __repr__(self):
        return f"{self.time}, {self.data}"

TIME_BIAS=[7.525437525437525, 2.608872608872605, -2.503052503052505, -0.6796906796906796, 4.204314204314201]

CALIB_AXIS = [0,1,2,3,4,5]

class _Dataset:
    def __init__(self, count):
        self.data_name = f"DATA/data_log{count}.csv"
        self.calib_name = f"DATA/calib{count}.txt"
        self.time_bias =  5.246235246235244

        self._raw_start_time = None
        self._calib_start_time = None

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
        
        with open(self.calib_name, 'r') as f:
            calib = f.readlines()
            self._calib_data = []
            for line in calib:
                dd = line.strip().split(' ')
                calib_data = np.array(dd[1:],dtype=float)
                if self._calib_start_time is None:
                    self._calib_start_time = float(dd[0])
                self._calib_data.append(CalibData(float(dd[0])-self._calib_start_time + self.time_bias,calib_data))
        
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
        return np.array(raw_data), np.array(calib_data)


def draw(raw_data, calib_data, * , axis = None):
    import matplotlib.pyplot as plt

    
    if axis is not None:
        fig, axs = plt.subplots(1, len(axis))
        for i,index in enumerate(axis):
            # random a color
            axs[i].plot(raw_data[:,index], label=f"raw{index}")
            axs[i].plot(calib_data[:,index], label=f"calib{index}")
    else:
        fig, ax = plt.subplots()
        for i in range(raw_data.shape[1]):
            ax.plot(raw_data[:,i], 'r', label=f"raw{i}")
        for i in range(calib_data.shape[1]):
            ax.plot(calib_data[:,i], 'b', label=f"calib{i}")

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
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

dataset = _Dataset(17)
raw_data, calib_data = dataset.get_data()
raw_data = torch.tensor(raw_data, dtype=torch.float32)
calib_data = torch.tensor(calib_data, dtype=torch.float32)
#x_data1 is 1st column of raw_data

# x_data1 = raw_data[:,0]*100
# #x_data2 is 4th column of raw_data
# x_data2 = raw_data[:,3]*100
# #x_data is x_data1 and x_data2
# x_data = torch.stack((x_data1, x_data2), 1)
x_data = raw_data*100

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

for epoch in range(15000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"epoch: {epoch}, loss: {loss.item()}")

print(model.linear.weight.data)
print(model.linear.bias.data)

verify = model(x_data)
verify = verify.detach().numpy()
draw(y_data, verify, axis=[0,1,2,3,4,5])
##################################

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")
# dataset = SensorDataset()

# x = dataset.raw_data[:,5]
# y = -dataset.calib_data[:,5]*0.2

# print("dddddddddddddddd : ",x.mean(),y.mean(),(y/x).mean(),(y/x).std())

# def draw_single(x, y):
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     ax.plot(x, 'r', label="raw")
#     ax.plot(y, 'b', label="calib")
#     plt.show()

# draw_single(x,y)

# dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
# model = SensorModel().to(device)
# print(model)
# for name, param in model.named_parameters():
#     print(f"{name}: {param.shape}")

# loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# def train(dataloader, model, loss_fn, optimizer):
#     model.train()
#     size = len(dataloader.dataset)
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if batch % 30 == 0:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# def test(dataloader, model):
#     size = len(dataloader.dataset)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#     test_loss /= size
#     print(f"Avg loss: {test_loss:>8f}")

# epochs = 300
# for t in range(epochs):
#     print(f"Epoch {t+1}")
#     train(dataloader, model, loss_fn, optimizer)
#     test(dataloader, model)

# # plot model coefficients matrix
# def plot_matrix(matrix):
#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots()
#     cax = ax.matshow(matrix, cmap='PiYG')
#     fig.colorbar(cax)
#     plt.show()

# calib_data = model(dataset.raw_data.to(device)).detach().cpu().numpy()

# # print model bias parameters
# bias = model.bias.detach().cpu().numpy()
# mx = model.mx.detach().cpu().numpy()
# print(bias, "\n", mx)

# draw(dataset.calib_data, calib_data, axis=[0,1])

# plot_matrix(mx)