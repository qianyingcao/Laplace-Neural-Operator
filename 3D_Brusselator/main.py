'''
Module Description:
------
This module implements the Laplace Neural Operator for Brusselator reaction-diffusion system (Example 10 in LNO paper)
Author: 
------
Qianying Cao (qianying_cao@brown.edu)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import time
from timeit import default_timer
from utilities3 import *
from Adam import Adam
import time


# ====================================
#  Laplace layer: pole-residue operation is used to calculate the poles and residues of the output
# ====================================  

class PR3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(PR3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, dtype=torch.cfloat))
        self.weights_pole3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes3, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  self.modes2, self.modes3, dtype=torch.cfloat))

    def output_PR(self, lambda1, lambda2, lambda3, alpha, weights_pole1, weights_pole2, weights_pole3, weights_residue):
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],weights_residue.shape[4],lambda1.shape[0], lambda2.shape[0], lambda2.shape[3], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.einsum("pbix,qbik,rbio->pqrbixko",torch.sub(lambda1,weights_pole1),torch.sub(lambda2,weights_pole2),torch.sub(lambda3,weights_pole3)))
        Hw=torch.einsum("bixko,pqrbixko->pqrbixko",weights_residue,term1)
        output_residue1=torch.einsum("bioxs,oxsikpqr->bkoxs", alpha, Hw) 
        output_residue2=torch.einsum("bioxs,oxsikpqr->bkpqr", alpha, -Hw) 
        return output_residue1,output_residue2
    

    def forward(self, x):
        tt=T.cuda()
        tx=X.cuda()
        ty=Y.cuda()
        #Compute input poles and resudes by FFT
        dty=(ty[0,1]-ty[0,0]).item()  # location interval
        dtx=(tx[0,1]-tx[0,0]).item()  # location interval
        dtt=(tt[0,1]-tt[0,0]).item()  # time interval
        alpha = torch.fft.fftn(x, dim=[-3,-2,-1])
        omega1=torch.fft.fftfreq(tt.shape[1], dtt)*2*np.pi*1j   # time frequency
        omega2=torch.fft.fftfreq(tx.shape[1], dtx)*2*np.pi*1j   # location frequency
        omega3=torch.fft.fftfreq(ty.shape[1], dty)*2*np.pi*1j   # location frequency
        omega1=omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega2=omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega3=omega3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=omega1.cuda()
        lambda2=omega2.cuda()    
        lambda3=omega3.cuda()

        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2 = self.output_PR(lambda1, lambda2, lambda3, alpha, self.weights_pole1, self.weights_pole2, self.weights_pole3, self.weights_residue)
 
      
        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifftn(output_residue1, s=(x.size(-3),x.size(-2), x.size(-1)))
        x1 = torch.real(x1)
        term1=torch.einsum("bip,kz->bipz", self.weights_pole1, tt.type(torch.complex64).reshape(1,-1))
        term2=torch.einsum("biq,kx->biqx", self.weights_pole2, tx.type(torch.complex64).reshape(1,-1))
        term3=torch.einsum("bim,ky->bimy", self.weights_pole3, ty.type(torch.complex64).reshape(1,-1))
        term4=torch.einsum("bipz,biqx,bimy->bipqmzxy", torch.exp(term1),torch.exp(term2),torch.exp(term3))
        x2=torch.einsum("kbpqm,bipqmzxy->kizxy", output_residue2,term4)
        x2=torch.real(x2)
        x2=x2/x.size(-1)/x.size(-2)/x.size(-3)
        return x1+x2

class LNO3d(nn.Module):
    def __init__(self, width,modes1,modes2,modes3):
        super(LNO3d, self).__init__()

        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.fc0 = nn.Linear(4, self.width) 

        self.conv0 = PR3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm3d(self.width)

        self.fc1 = nn.Linear(self.width,64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self,x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.norm(self.conv0(self.norm(x)))
        x2 = self.w0(x)
        x = x1 +x2
        # x = F.relu(x)

        # x1 = self.norm(self.conv1(self.norm(x)))
        # x2 = self.w1(x)
        # x = x1 +x2

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_t, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = gridt.reshape(1, size_t, 1, 1, 1).repeat([batchsize, 1, size_x, size_y,1])
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1, 1).repeat([batchsize, size_t, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1,size_y, 1).repeat([batchsize, size_t, size_x, 1,1])
        return torch.cat((gridt,gridx, gridy), dim=-1).to(device)

# ====================================
#  Define parameters and Load data
# ====================================
file = np.load('Data/Brusselator_force_train.npz')

nt, nx, ny = 39, file['nx'], file['ny']
num_train = file['n_samples1']
num_test = file['n_samples2']
inputs_train = file['inputs_train'].reshape(num_train, nt)
inputs_test = file['inputs_test'].reshape(num_test, nt)
outputs_train = np.array((file['outputs_train'])).reshape(num_train, nt, nx, ny)
outputs_test = np.array((file['outputs_test'])).reshape(num_test, nt, nx, ny)
        
batch_size = 50
epochs = 300
learning_rate = 0.005
step_size = 100
gamma = 0.5

t = nt
orig_r = 28
r = 2
h = int(((orig_r - 1)/r) + 1)
s = h

modes1 = 4
modes2 = 4
modes3 = 4
width = 8
    
x = np.linspace(0, 1, orig_r)
y = np.linspace(0, 1, orig_r)
z = np.linspace(0, 1, t)
tt, xx, yy = np.meshgrid(z, x, y, indexing='ij')

T=torch.linspace(0,19,nt).reshape(1,nt)
X=torch.linspace(0,1,steps=orig_r).reshape(1,orig_r)[:,:s]
Y=torch.linspace(0,1,steps=orig_r).reshape(1,orig_r)[:,:s]


x_train = torch.tile(torch.tensor(inputs_train),(orig_r,orig_r,1,1)).permute(2,3,0,1)[:,:,::r,::r][:,:,:s,:s]
y_train = torch.tensor(outputs_train)[:,:,::r,::r][:,:,:s,:s]
grid_x_train = torch.tile(torch.tensor(tt),(num_train,1,1,1))[:,:,::r,::r][:,:,:s,:s]
grid_y_train = torch.tile(torch.tensor(xx),(num_train,1,1,1))[:,:,::r,::r][:,:,:s,:s]
grid_z_train = torch.tile(torch.tensor(yy),(num_train,1,1,1))[:,:,::r,::r][:,:,:s,:s]

x_test = torch.tile(torch.tensor(inputs_test),(orig_r,orig_r,1,1)).permute(2,3,0,1)[:,:,::r,::r][:,:,:s,:s]
y_test = torch.tensor(outputs_test)[:,:,::r,::r][:,:,:s,:s]
grid_x_test = torch.tile(torch.tensor(tt),(num_test,1,1,1))[:,:,::r,::r][:,:,:s,:s]
grid_y_test = torch.tile(torch.tensor(xx),(num_test,1,1,1))[:,:,::r,::r][:,:,:s,:s]
grid_z_test = torch.tile(torch.tensor(yy),(num_test,1,1,1))[:,:,::r,::r][:,:,:s,:s]


x_normalizer = RangeNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = RangeNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

grid_x_train = grid_x_train.reshape(num_train, t, s, s, 1)  
grid_x_train.requires_grad = True
grid_y_train = grid_y_train.reshape(num_train, t, s, s, 1)
grid_y_train.requires_grad = True
grid_z_train = grid_z_train.reshape(num_train, t, s, s, 1)
grid_z_train.requires_grad = True
x_train = x_train.reshape(num_train, t, s, s, 1)
x_train.requires_grad = True
x_train = torch.cat([x_train, grid_x_train, grid_y_train, grid_z_train], dim = -1)

grid_x_test = grid_x_test.reshape(num_test, t, s, s, 1)
grid_y_test = grid_y_test.reshape(num_test, t, s, s, 1)
grid_z_test = grid_z_test.reshape(num_test, t, s, s, 1)
x_test = x_test.reshape(num_test, t, s, s, 1)
x_test = torch.cat([x_test, grid_x_test, grid_y_test, grid_z_test], dim = -1)


y_train = y_train.reshape(num_train, t, s, s, 1)
y_test = y_test.reshape(num_test, t, s, s, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)
train_loaderL2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

device = torch.device('cuda') 
# model
model = LNO3d(width,modes1, modes2,modes3).cuda()


# ====================================
# Training 
# ====================================
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
start_time = time.time()
myloss = LpLoss(size_average=False)
x_normalizer.cuda()
y_normalizer.cuda()

train_error = np.zeros((epochs, 1))
train_loss = np.zeros((epochs, 1))
vali_error = np.zeros((epochs, 1))
vali_loss = np.zeros((epochs, 1))
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()     
        optimizer.zero_grad()
        out = model(x.float())

        out = out[:,:,:,:,0:1]            
        y = y[:,:,:,:,0:1]

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1)) 
        loss.backward()

        optimizer.step()
        train_mse += loss.item()  

    scheduler.step()
    model.eval()
    train_l2 = 0.0
    with torch.no_grad():
            for x, y in train_loaderL2:
                x, y = x.cuda(), y.cuda()
                out = model(x.float())
                
                out = y_normalizer.decode(out[:,:,:,:,0:1])               
                y = y_normalizer.decode(y[:,:,:,:,0:1])
                
                train_l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item() 

    test_l2 = 0.0
    with torch.no_grad():
        n_vali=0
        for x, y in vali_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x.float())              
            out = y_normalizer.decode(out[:,:,:,:,0:1])
            y = y[:,:,:,:,0:1]
            test_l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item() 
            
    train_mse /= len(train_loader)
    train_l2 /= num_train
    test_l2 /= num_test


    train_error[ep,0] = train_l2
    vali_error[ep,0] = test_l2
    t2 = default_timer()
    print("Epoch: %d, time: %.3f, Train l2: %.4f, Vali l2: %.4f" % (ep, t2-t1, train_l2, test_l2))
elapsed = time.time() - start_time
print("\n=============================")
print("Training done...")
print('Training time: %.3f'%(elapsed))
print("=============================\n")


# ====================================
# saving settings
# ====================================
current_directory = os.getcwd()
case = "Case_"
save_index = 1  
folder_index = str(save_index)

results_dir = "/" + case + folder_index +"/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

x = np.linspace(0, epochs-1, epochs)
np.savetxt(save_results_to+'/epoch.txt', x)
np.savetxt(save_results_to+'/train_error.txt', train_error)
np.savetxt(save_results_to+'/vali_error.txt', vali_error)    
save_models_to = save_results_to +"model/"
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)
    
torch.save(model, save_models_to+'Wave_states')

# ====================================
# Testing 
# ====================================
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
pred_u = torch.zeros(num_test,y_test.shape[1],y_test.shape[2],y_test.shape[3])
index = 0
test_l2 = 0.0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        out = model(x.float())
        out = y_normalizer.decode(out[:,:,:,:,0])    
        test_l2 += myloss(out, y[:,:,:,:,0]).item()
        pred_u[index,:,:,:] = out
        index = index + 1
test_l2 /= index
scipy.io.savemat(save_results_to+'wave_states_test.mat', 
                     mdict={ 'test_err': test_l2,
                            'T': T.numpy(),
                            'X': X.numpy(),
                            'Y': Y.numpy(),
                            'y_test': y_test.numpy(), 
                            'y_pred': pred_u.cpu().numpy(),
                            'Train_time':elapsed})  
    
    
print("\n=============================")
print('Testing error: %.3e'%(test_l2))
print("=============================\n")

