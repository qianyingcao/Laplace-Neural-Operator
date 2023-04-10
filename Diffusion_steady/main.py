# This script is for the prediction of the states in a wave
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import os
import time
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import sys
from Adam import Adam
import scipy
torch.manual_seed(0)
np.random.seed(0)
from torchsummary import summary

################################################################
#  2d fourier layer
################################################################
# ====================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  dtype=torch.cfloat))
        self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  self.modes2, dtype=torch.cfloat))
  # # Original code for Complex multiplication, which is slow
    # def compl_mul1d(self, lambda1, lambda2, alpha, weights_pole1, weights_pole2, weights_residue):
    #     Pk=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],lambda1.shape[0], lambda2.shape[0], device=alpha.device, dtype=torch.cfloat)
    #     Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],lambda1.shape[0], lambda2.shape[0], device=alpha.device, dtype=torch.cfloat)
    #     for le1 in range(lambda1.shape[0]):
    #         for le2 in range(lambda2.shape[0]):
    #             term1=torch.einsum("bix,bik->bixk",torch.sub(lambda1[le1],weights_pole1),torch.sub(lambda2[le2],weights_pole2))
    #             term2=torch.div(1,term1)
    #             Hw[:,:,:,:,le1,le2]=weights_residue*term2
    #             term3=torch.einsum("bix,bik->bixk",torch.sub(weights_pole1,lambda1[le1]),torch.sub(weights_pole2,lambda2[le2]))
    #             term4=torch.div(1,term3)
    #             Pk[:,:,:,:,le1,le2]=weights_residue*term4
    #     output_residue1=torch.einsum("biox,ikpqox->bkox", alpha, Hw) 
    #     output_residue2=torch.einsum("biox,ikpqox->bkpq", alpha, Pk) 
    #     return output_residue1,output_residue2
    # New code for Complex multiplication, which is fast
    def compl_mul1d(self, lambda1, lambda2, alpha, weights_pole1, weights_pole2, weights_residue):
        Pk=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],lambda1.shape[0], lambda2.shape[0], device=alpha.device, dtype=torch.cfloat)
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],lambda1.shape[0], lambda2.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.einsum("pbix,qbik->pqbixk",torch.sub(lambda1,weights_pole1),torch.sub(lambda2,weights_pole2)))
        Hw=torch.einsum("bixk,pqbixk->pqbixk",weights_residue,term1)
        # term2=torch.div(1,torch.einsum("pbix,qbik->pqbixk",torch.sub(weights_pole1,lambda1),torch.sub(weights_pole2,lambda2)))
        # Pk=torch.einsum("bixk,pqbixk->pqbixk",weights_residue,term2)
        Pk=Hw # for ode, Pk=-Hw; for 2d pde, Pk=Hw; for 3d pde, Pk=-Hw; 
        output_residue1=torch.einsum("biox,oxikpq->bkox", alpha, Hw) 
        output_residue2=torch.einsum("biox,oxikpq->bkpq", alpha, Pk) 
        return output_residue1,output_residue2
    

    def forward(self, x):
        #Compute input poles and resudes by FFT
        tx=T.cuda()
        ty=X.cuda()
        dty=(ty[0,1]-ty[0,0]).item()  # location interval
        dtx=(tx[0,1]-tx[0,0]).item()  # time interval
        alpha = torch.fft.fft2(x, dim=[-2,-1])
        omega1=torch.fft.fftfreq(ty.shape[1], dty)*2*np.pi*1j   # location frequency
        omega2=torch.fft.fftfreq(tx.shape[1], dtx)*2*np.pi*1j   # time frequency
        omega1=omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega2=omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=omega1.cuda()
        lambda2=omega2.cuda()    
        #start=time.time()
        # Obtain output poles and residues
        output_residue1,output_residue2 = self.compl_mul1d(lambda1, lambda2, alpha, self.weights_pole1, self.weights_pole2, self.weights_residue)
        # stop=time.time()
        # time_req=stop-start
        # print(time_req)
 
      
        #Return to physical space
        x1 = torch.fft.ifft2(output_residue1, s=(x.size(-2), x.size(-1)))
        x1 = torch.real(x1)
        #x2=torch.zeros(output_residue2.shape[0],output_residue2.shape[1],ty.shape[0], tx.shape[0], device=alpha.device, dtype=torch.cfloat)    
        term1=torch.einsum("bip,kz->bipz", self.weights_pole1, ty.type(torch.complex64).reshape(1,-1))
        term2=torch.einsum("biq,kx->biqx", self.weights_pole2, tx.type(torch.complex64).reshape(1,-1))
        term3=torch.einsum("bipz,biqx->bipqzx", torch.exp(term1),torch.exp(term2))
        x2=torch.einsum("kbpq,bipqzx->kizx", output_residue2,term3)
        x2=torch.real(x2)
        x2=x2/x.size(-1)/x.size(-2)
        return x1+x2

class FNO1d(nn.Module):
    def __init__(self, width,modes1,modes2):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self,x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.norm(self.conv0(self.norm(x)))
        x2 = self.w0(x)
        x = x1 +x2

        # x1 = self.norm(self.conv0(tx,ty,self.norm(x)))
        # x2 = self.w0(x)
        # x = x1 +x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.sin(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
#  configurations
################################################################
     
ntrain = 200
nvali = 50
ntest=130

batch_size_train = 50
batch_size_vali = 50
#batch_size_test = ntest

learning_rate = 0.002

epochs = 1000
step_size = 100
gamma = 0.5

modes1 = 4   # time coordinate
modes2 = 5   # location coordinate
width = 16

################################################################
# load data and data normalization
################################################################
reader = MatReader('Data/data.mat')
x_train = reader.read_field('f_train')
y_train = reader.read_field('u_train')
T = reader.read_field('t')
X = reader.read_field('x')

x_vali = reader.read_field('f_vali')
y_vali = reader.read_field('u_vali')


x_test = reader.read_field('f_test')
y_test = reader.read_field('u_test')

# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_vali = x_normalizer.encode(x_vali)
# x_test = x_normalizer.encode(x_test)

# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)
# y_vali = y_normalizer.encode(y_vali)
# y_test1 = y_normalizer.encode(y_test)


x_train = x_train.reshape(ntrain,x_train.shape[1],x_train.shape[2],1)
x_vali = x_vali.reshape(nvali,x_vali.shape[1],x_vali.shape[2],1)
x_test = x_test.reshape(ntest,x_test.shape[1],x_test.shape[2],1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=True)
# model
model = FNO1d(width,modes1, modes2).cuda()


################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
start_time = time.time()
myloss = LpLoss(size_average=True)
#y_normalizer.cuda()

train_error = np.zeros((epochs, 1))
train_loss = np.zeros((epochs, 1))
vali_error = np.zeros((epochs, 1))
vali_loss = np.zeros((epochs, 1))
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    n_train=0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)   

        mse = F.mse_loss(out.view(batch_size_train, -1), y.view(batch_size_train, -1), reduction='mean')
        l2 = myloss(out.view(-1,x_train.shape[1],x_train.shape[2]), y)
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()
        n_train += 1

    scheduler.step()
    model.eval()
    vali_mse = 0.0
    vali_l2 = 0.0
    with torch.no_grad():
        n_vali=0
        for x, y in vali_loader:
            x, y = x.cuda(), y.cuda()
            tx=T.cuda()
            ty=X.cuda()
            out = model(x)
            #out = y_normalizer.decode(out[:,:,:,0:1].reshape(-1,x_vali.shape[1],x_vali.shape[2]))
            mse=F.mse_loss(out.view(-1,x_vali.shape[1],x_vali.shape[2]), y, reduction='mean')
            vali_l2 += myloss(out.view(-1,x_vali.shape[1],x_vali.shape[2]), y).item()
            vali_mse += mse.item()
            n_vali += 1

    train_mse /= n_train
    vali_mse /= n_vali
    train_l2 /= n_train
    vali_l2 /= n_vali
    train_error[ep,0] = train_l2
    vali_error[ep,0] = vali_l2
    train_loss[ep,0] = train_mse
    vali_loss[ep,0] = vali_mse
    t2 = default_timer()
    print("Epoch: %d, time: %.3f, Train Loss: %.3e,Vali Loss: %.3e, Train l2: %.4f, Vali l2: %.4f" % (ep, t2-t1, train_mse, vali_mse,train_l2, vali_l2))
elapsed = time.time() - start_time
print("\n=============================")
print("Training done...")
print('Training time: %.3f'%(elapsed))
print("=============================\n")
summary(model,(80,25,1),device="cuda")
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
np.savetxt(save_results_to+'/train_loss.txt', train_loss)
np.savetxt(save_results_to+'/vali_loss.txt', vali_loss)
np.savetxt(save_results_to+'/train_error.txt', train_error)
np.savetxt(save_results_to+'/vali_error.txt', vali_error)    
save_models_to = save_results_to +"model/"
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)
    
torch.save(model, save_models_to+'Wave_states')

################################################################
# testing
################################################################
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
pred_u = torch.zeros(ntest,y_test.shape[1],y_test.shape[2])
index = 0
test_l2 = 0.0
#test_l2_n = 0.0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        out = model(x)
        # test_l2_n += myloss(out.view(-1,x_test.shape[1],x_test.shape[2]), y).item()
        # out = y_normalizer.decode(out[0,:,:,0:1].reshape(-1,x_test.shape[1],x_test.shape[2]))
        test_l2 += myloss(out.view(-1,x_test.shape[1],x_test.shape[2]), y).item()
        pred_u[index,:,:] = out.view(-1,x_test.shape[1],x_test.shape[2])
        index = index + 1
test_l2 /= index
#test_l2_n /= index
scipy.io.savemat(save_results_to+'wave_states_test.mat', 
                     mdict={ 'test_err': test_l2,
                            'T': T.numpy(),
                            'X': X.numpy(),
                            'y_test': y_test.numpy(), 
                            'y_pred': pred_u.cpu().numpy()})  
    
    
print("\n=============================")
print('Testing error: %.3e'%(test_l2))
print("=============================\n")


# Plotting the loss history
num_epoch = epochs
epoch = np.linspace(1, num_epoch, num_epoch)
fig = plt.figure(constrained_layout=False, figsize=(7, 7))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
ax.plot(epoch, train_loss[:,0], color='blue', label='Train Loss')
ax.plot(epoch, vali_loss[:,0], color='red', label='Validation Loss')
ax.set_yscale('log')
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')
ax.legend(loc='upper left')
fig.savefig(save_results_to+'loss_history.png')
plt.show()