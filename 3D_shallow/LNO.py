'''
Module Description:
------
This module implements the Laplace Neural Operator for shallow-water equation (Example 11 in LNO paper)
Author: 
------
Qianying Cao (qianying_cao@brown.edu)
Somdatta Goswami (somdatta_goswami@brown.edu)
'''

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
import random
torch.manual_seed(0)
np.random.seed(0)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import argparse


################################################################
#  Laplace layer
################################################################
# ====================================
class PR3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(PR3d, self).__init__()

        # Set the value of the modes to given values
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        # Calculate the scale value for the weights initialization
        self.scale = (1 / (in_channels*out_channels))
        # Initialize the system poles using random values scaled by the calculated scales
        self.weights_pole1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  dtype=torch.cfloat)) 
        self.weights_pole2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes2, dtype=torch.cfloat))
        self.weights_pole3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes3, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1,  self.modes2, self.modes3, dtype=torch.cfloat))

    def output_PR(self, lambda1, lambda2, lambda3, alpha, weights_pole1, weights_pole2, weights_pole3, weights_residue):
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],weights_residue.shape[3],weights_residue.shape[4],lambda1.shape[0], lambda2.shape[0], lambda2.shape[3], device=alpha.device, dtype=torch.cfloat)# Create arrays to store kernel
        term1=torch.div(1,torch.einsum("pbix,qbik,rbio->pqrbixko",torch.sub(lambda1,weights_pole1),torch.sub(lambda2,weights_pole2),torch.sub(lambda3,weights_pole3)))
        Hw=torch.einsum("bixko,pqrbixko->pqrbixko",weights_residue,term1) # Calculate the kernel in Eq.31
        output_residue1=torch.einsum("bioxs,oxsikpqr->bkoxs", alpha, Hw) # Calculate output residues corresponding to the excitation poles
        output_residue2=torch.einsum("bioxs,oxsikpqr->bkpqr", alpha, -Hw) # Calculate output residues corresponding to the system poles
        return output_residue1,output_residue2
    

    def forward(self, x):
        tt=T.cuda()  # Load time history
        tx=X.cuda()  # Load the first dimension
        ty=Y.cuda()  # Load the second dimension
        #Compute input poles and resudes by FFT
        dty=(ty[0,1]-ty[0,0]).item()  # Time interval
        dtx=(tx[0,1]-tx[0,0]).item()  # Interval of the first dimension
        dtt=(tt[0,1]-tt[0,0]).item()  # Interval of the second dimension
        alpha = torch.fft.fftn(x, dim=[-3,-2,-1]) # Perform 3D FFT of the input to obtain input residues
        omega1=torch.fft.fftfreq(tt.shape[1], dtt)*2*np.pi*1j   # Obtain input poles in time dimension
        omega2=torch.fft.fftfreq(tx.shape[1], dtx)*2*np.pi*1j   # Obtain input poles in the first dimension
        omega3=torch.fft.fftfreq(ty.shape[1], dty)*2*np.pi*1j   # Obtain input poles in the second dimension
        # Expand the dimensions of poles by unsqueezing it along the last four dimensions
        omega1=omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        omega2=omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega3=omega3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Move the poles to the GPU
        lambda1=omega1.cuda() 
        lambda2=omega2.cuda()    
        lambda3=omega3.cuda()

        # Obtain output poles and residues
        output_residue1,output_residue2 = self.output_PR(lambda1, lambda2, lambda3, alpha, self.weights_pole1, self.weights_pole2, self.weights_pole3, self.weights_residue)
 
        #Return to physical space
        x1 = torch.fft.ifftn(output_residue1, s=(x.size(-3),x.size(-2), x.size(-1))) # Obtain steady-state response by ifft, one also can obtain steady-state response by superposition
        x1 = torch.real(x1)  # Choose the real values
        # Calculate the product between system pole and the corresponding dimension
        term1=torch.einsum("bip,kz->bipz", self.weights_pole1, tt.type(torch.complex64).reshape(1,-1))  
        term2=torch.einsum("biq,kx->biqx", self.weights_pole2, tx.type(torch.complex64).reshape(1,-1))
        term3=torch.einsum("bim,ky->bimy", self.weights_pole3, ty.type(torch.complex64).reshape(1,-1))
        # Perform Einstein summation over indices to compute the tensor product of different terms 
        term4=torch.einsum("bipz,biqx,bimy->bipqmzxy", torch.exp(term1),torch.exp(term2),torch.exp(term3))
        x2=torch.einsum("kbpqm,bipqmzxy->kizxy", output_residue2,term4) # Obtain transient response
        x2=torch.real(x2) # Choose the real values
        x2=x2/x.size(-1)/x.size(-2)/x.size(-3) # Change the scale the transient response with same scale of the steady-state response
        return x1+x2 # Calculate the total response by adding A and B

class LNO3d(nn.Module):
    def __init__(self, width,modes1,modes2,modes3):
        super(LNO3d, self).__init__()
# Set the value of the witdth and modes to given values
        self.width = width
        self.modes1 = modes1 
        self.modes2 = modes2
        self.modes3 = modes3
        self.fc0 = nn.Linear(4, self.width) # Input channel is 2: (F(x), x), add the mesh grid
        # Perform pole-residue operation to calculate response
        self.conv0 = PR3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = PR3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = PR3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = PR3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        # Applies a 3D convolution over an input signal composed of several input planes.
        self.w0 = nn.Conv3d(self.width, self.width, 1) 
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm3d(self.width) # Applies Instance Normalization

        self.fc1 = nn.Linear(self.width,64) # Perform linear transformation
        self.fc2 = nn.Linear(64, 1) # Perform linear transformation

    def forward(self,x):
        grid = self.get_grid(x.shape, x.device) # Generate mesh grid
        x = torch.cat((x, grid), dim=-1) # Concatenates the given sequence in the last dimension.
        x = self.fc0(x) # The input is transformed into a higher dimensional representation using a lifting layer
        x = x.permute(0, 4, 1, 2, 3) 

        x1 = self.norm(self.conv0(self.norm(x))) # Response from the 1st Laplace layer
        x2 = self.w0(x) # Response form the linear transformation W
        x = x1 +x2   # Total response 
        x = F.relu(x)  # Activation function

        x1 = self.norm(self.conv1(self.norm(x))) # Response from the 2nd Laplace layer
        x2 = self.w1(x)# Response form the linear transformation W
        x = x1 +x2# Total response
        x = F.relu(x)# Activation function

        x1 = self.norm(self.conv2(self.norm(x))) # Response from the 3rd Laplace layer
        x2 = self.w2(x)# Response form the linear transformation W
        x = x1 +x2# Total response
        x = F.relu(x)# Activation function

        x1 = self.norm(self.conv3(self.norm(x)))# Response from the 4th Laplace layer
        x2 = self.w3(x)# Response form the linear transformation W
        x = x1 +x2# Total response

        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)# Linear transformation
        x = F.relu(x)# Activation function
        x = self.fc2(x) # The output is obtained by projecting through a linear transformation
        return x

    def get_grid(self, shape, device): 
        batchsize, size_t, size_x, size_y = shape[0], shape[1], shape[2], shape[3] # Obtain the size of batch and different dimensions
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float) # Create a tensor ranging from 0 to 1 with size_t
        gridt = gridt.reshape(1, size_t, 1, 1, 1).repeat([batchsize, 1, size_x, size_y,1]) 
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float) # Create a tensor ranging from 0 to 1 with size_x
        gridx = gridx.reshape(1, 1, size_x, 1, 1).repeat([batchsize, size_t, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)# Create a tensor ranging from 0 to 1 with size_y
        gridy = gridy.reshape(1, 1, 1,size_y, 1).repeat([batchsize, size_t, size_x, 1,1])
        return torch.cat((gridt,gridx, gridy), dim=-1).to(device)  # Generate mesh grid


def preprocess(x, y):
    # Gives one datapoint back 
    n = x.shape[0] # Get the number of samples in the input dat
    xx = np.tile(x,[nt,1,1]).transpose(1,0,2) # Tile the input data to match the time steps and reshape it
    X_func = xx.reshape(n, nt,ld_sqrt, ld_sqrt, 1) # Reshape the data to the desired shape (#, nt, sqrt(ld), sqrt(ld), 1)
# Create a tensor ranging from 0 to 1
    T = torch.linspace(0, 1, nt).reshape(1,nt)
    X = torch.linspace(0, 1, ld_sqrt).reshape(1,ld_sqrt)
    Y = torch.linspace(0, 1, ld_sqrt).reshape(1,ld_sqrt)
    y = y.reshape(-1, nt, ld_sqrt, ld_sqrt,1) # Reshape the target labels to the desired shape
    #print(y.shape)

    return X_func, T, X, Y, y

## Parser
parser = argparse.ArgumentParser(description='Running autoencoder models.')
parser.add_argument(
    '--method',
    default='MLAE',     # Determine the autoencoder method
    help='vanilla-AE | MLAE | CAE | WAE')
parser.add_argument(
    '--latent_dim',
    type=int,
    default=64,   # Determine latent dimensionality
    help='latent dimensionality (default: 64)')
parser.add_argument(
    '--n_samples',
    type=int,
    default=300,   # Determine number of generated samples
    help='number of generated samples (default: 800)')
parser.add_argument(
    '--n_epochs',
    type=int,
    default=1000,  # Determine number of epochs
    help='number of epochs (default: 800)')
parser.add_argument(
    '--bs',
    type=int,
    default=8,   # Determine batch size 
    help='batch size (default: 128)')

args, unknown = parser.parse_known_args()

#### Fix random see (for reproducibility of results)
seed_value = random.randint(1,1000)
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

# Load all data (original + reduced)
data_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/data/'
file = np.load(data_dir + 'data_d_{}.npz'.format(args.latent_dim))

nt, nx, ny = 72, 256, 256
ld = int(file['latent_dim']) # latent dimension
ld_sqrt = int(np.sqrt(ld)) # square of latent dimension (for branch net convolutions)

# Load data (original + reduced inputs (x) and outputs (y))
x_train_red = file['x_train_red'] # Load reduced input for training
x_test_red = file['x_test_red'] # Load reduced input for testing

y_train_red = file['y_train_red'] # Load reduced output for training
y_test_red = file['y_test_red'] # Load reduced output for testing

x_train = file['x_train'] # Load original input for training
x_test = file['x_test'] # Load original input for testing

y_train_og = file['y_train']  # Load original output for training
y_test_og = file['y_test'] # Load original output for testing

norm_max=file['norm_max']  # Load maximum value of the output
norm_min=file['norm_min']  # Load minimum value of the output

num_train = x_train.shape[0]  # number of training data
num_test = x_test.shape[0] # number of testing data

if ld != args.latent_dim:
    print('Warning! The latent dimension of the saved file and the one given on prompt do not match!')

if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_LNO/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_LNO/')     
if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_LNO/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_LNO/') 

# Load autoencoder class
class_AE_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_AE/'

class MLAE(nn.Module):
    def __init__(self, latent_dim):
        super(MLAE, self).__init__()
        self.latent_dim = latent_dim
        # Define the encoder network as a sequential module
        self.encoder = nn.Sequential(
            nn.Flatten(), # Flatten the input tensor
            nn.Linear(nx * ny, 256), # Linear layer with input size nx*ny and output size 256
            nn.ReLU(),   # ReLU activation function
            nn.Linear(256, 169), # Linear layer with input size 256 and output size 169
            nn.ReLU(),   # ReLU activation function
            nn.Linear(169, 121), # Linear layer with input size 169 and output size 121
            nn.ReLU(),   # ReLU activation function
            nn.Linear(121, latent_dim),# Linear layer with input size 121 and output size latent_dim
            nn.ReLU()    # ReLU activation function
        )
        # Define the decoder network as a sequential module 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 121),
            nn.ReLU(),
            nn.Linear(121, 169),
            nn.ReLU(),
            nn.Linear(169, 256),
            nn.ReLU(),
            nn.Linear(256, nx * ny),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)   # Perform encoder
        decoded = self.decoder(encoded)  # Perform decoder
        decoded = decoded.view(-1, nx, ny) # Reshape to the desired shape
        return decoded
    
autoencoder = torch.load(class_AE_dir + 'class_AE_{}'.format(args.latent_dim))  # load autoencoder   

autoencoder.eval()

Par = {}
class_LNO_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_LNO/'  # Adress of the file
Par['address'] = class_LNO_dir

X_func_train, T,X,Y, y_train = preprocess(x_train_red, y_train_red) # Perform preprocess to obtain the desired shape of training data
X_func_test, T,X,Y, y_test = preprocess(x_test_red, y_test_red) # Perform preprocess to obtain the desired shape of testing data

# Par['n_channels'] = X_func_train.shape[-1]
# Par['mean'] = np.mean(X_func_train) 
# Par['std'] =  np.std(X_func_train)


print('LNO training in progress...')

# Parameters of LNO
batch_size = args.bs      # Batch size
epochs = args.n_epochs    # Epochs for training process
learning_rate = 0.01      # Learning rate
step_size = 100           # Period of learning rate decay 
gamma = 0.5               # Multiplicative factor of learning rate decay
modes1 = 4                # Modes of the first dimension (time)
modes2 = 2                # Modes of the second dimension (longitude)
modes3 = 2                # Modes of the third dimension (latitude)
width = 64
    
################################################################
# load data
################################################################
x_train = torch.tensor(X_func_train)  # Input data for training
x_test = torch.tensor(X_func_test)    # Input data for testing
y_train = torch.tensor(y_train)       # Output data for training
y_test = torch.tensor(y_test)         # Output data for testing

# Create a data loader for training data, which loads data in batches and shuffles the data
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)  
# Create a data loader for validating data, which loads data in batches without shuffling
vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
device = torch.device('cuda')  # Set the device to be a GPU for faster computation
# model
model = LNO3d(width,modes1, modes2,modes3).cuda()  # Construct the LNO1d model with specified parameters and move it to the GPU

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Create an optimizer for model parameters using the Adam optimizer with specified learning rate and weight decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) # Create a learning rate scheduler for the optimizer, which adjusts the learning rate based on the step size and gamma
start_time = time.time()  # Record the start time for training
myloss = LpLoss(size_average=False) # Create LpLoss: relative L2 loss
# Create arrays to store training nad valicatin errors and losses for each epoch
train_error = np.zeros((epochs, 1))
train_loss = np.zeros((epochs, 1))
vali_error = np.zeros((epochs, 1))
vali_loss = np.zeros((epochs, 1))
for ep in range(epochs):
    model.train()  # Set the model in training mode
    t1 = default_timer() # Record the start time for training
    # Initialize variables to store training MSE and L2 loss
    train_mse = 0
    train_l2 = 0.0
    # Iterate over the training data batches
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()  # Move the input data and target labels to the GPU
        optimizer.zero_grad() # Clear the gradients of the optimizer
        out = model(x.float())  # Forward pass: compute the model's output
        #  Select a subset of the output and target labels      
        out = out[:,:,:,:,0:1]       
        y = y[:,:,:,:,0:1]

        loss = myloss(out, y) # Calculate the loss function between the output and target labels
        loss.backward() # Backward pass: compute gradients of the loss
        #print(loss)
        optimizer.step() # Update the model parameters based on the computed gradients
        train_l2 += loss.item()  

    scheduler.step() # Adjust the learning rate using the scheduler

    model.eval() # set the model in evaluation mode

    vali_l2 = 0.0  # Initialize variable to store the validation L2 loss
    with torch.no_grad():  # Disable gradient calculation since we are in validation mode
        n_vali=0
        for x, y in vali_loader:  # Interate over the validation data batches
            x, y = x.cuda(), y.cuda() # Move the input data and target labels to the GPU
            out = model(x.float())   # Forward pass: compute the model's output    
            #  Select a subset of the output and target labels       
            out = out[:,:,:,:,0:1]    
            y = y[:,:,:,:,0:1]
            vali_l2 += myloss(out, y).item() # Calculate the loss function between the output and target labels
    # Calculate the average training/Validation MSE and L2 loss     
    train_mse /= num_train
    train_l2 /= num_train
    vali_l2 /= num_test

# Store the training/validation L2 loss in the corresponding epoch index of array
    train_error[ep,0] = train_l2
    vali_error[ep,0] = vali_l2
    t2 = default_timer()
    print("Epoch: %d, time: %.3f, Train l2: %.4f, Vali l2: %.4f" % (ep, t2-t1, train_l2, vali_l2))
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

# Save epochs, training error and validation error
x = np.linspace(0, epochs-1, epochs)
np.savetxt(save_results_to+'/epoch.txt', x)
np.savetxt(save_results_to+'/train_error.txt', train_error)
np.savetxt(save_results_to+'/vali_error.txt', vali_error)    
save_models_to = save_results_to +"model/"
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)
    
torch.save(model, save_models_to+'Wave_states')

################################################################
# testing
################################################################
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False) # Create a data loader for testing data, which loads data in batches without shuffling
pred_u = torch.zeros(num_test,y_test.shape[1],y_test.shape[2],y_test.shape[3])# Create arrays to store prediction results
index = 0    # Initialize variable to count the number of samples
test_l2 = 0.0  # Initialize variable to store the test L2 loss
with torch.no_grad():
    for x, y in test_loader: # Interate over the testing data batches
        x, y = x.cuda(), y.cuda() # Move the input data and target labels to the GPU
        out = model(x.float()) # Forward pass: compute the model's output  
        #  Select a subset of the output and target labels 
        out = out[:,:,:,:,0]     
        test_l2 += myloss(out, y[:,:,:,:,0]).item()
        pred_u[index,:,:,:] = out # Store the output in the corresponding sample index of array
        index = index + 1
test_l2 /= index # Calculate the average testing L2 loss in latent space

print("\n=============================")
print('Latent LNO relative L2 error on test data: %.3e'%(test_l2))
print("=============================\n")



pred_u = pred_u.reshape(num_test,nt,args.latent_dim) # Reshape the predicted latent space output to match the shape of the test data
## decoder
outputs_pred0 = autoencoder.decoder(pred_u).detach().numpy() # Decode the predicted latent space output using the autoencoder's decoder
num_t=num_train+num_test # The total number of samples
outputs_pred = torch.tensor(outputs_pred0*(norm_max -norm_min)+norm_min) # Scale the decoded output to the original range


# Save data
outputs_pred_s=outputs_pred.reshape(num_test,nt,y_test_og.shape[2],y_test_og.shape[2])
scipy.io.savemat(save_results_to+'wave_states_test.mat', 
                    mdict={ 'T': T.numpy(),
                            'X': X.numpy(),
                            'Y': Y.numpy(),
                            'y_test': y_test_og, 
                            'y_pred': outputs_pred_s.cpu().numpy()})  

