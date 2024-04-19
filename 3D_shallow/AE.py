'''
This code is changed based on the Latent-deeponet
available at https://github.com/katiana22/latent-deeponet
'''

import torch
import torch.nn as nn
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import random
#device = torch.device('cpu')


## Parser
parser = argparse.ArgumentParser(description='Running autoencoder models.')
parser.add_argument(
    '--method',
    default='MLAE',  # Determine the autoencoder method
    help='vanilla-AE | MLAE | CAE | WAE')
parser.add_argument(
    '--latent_dim',
    type=int,
    default=64, # Determine latent dimensionality
    help='latent dimensionality (default: 64)')
parser.add_argument(
    '--n_samples',
    type=int,
    default=300,  # Determine number of generated samples
    help='number of generated samples (default: 800)')
parser.add_argument(
    '--n_epochs',
    type=int,
    default=300, # Determine number of epochs
    help='number of epochs (default: 800)')
parser.add_argument(
    '--bs',
    type=int,
    default=16, # Determine batch size 
    help='batch size (default: 128)')
parser.add_argument(
    '--ood',
    type=int,
    default=0,
    help='generate results for OOD data')
parser.add_argument(
    '--noise',
    type=int,
    default=0,
    help='generate results for noisy data')

args, unknown = parser.parse_known_args()

#### Fix random see (for reproducibility of results)
seed_value = random.randint(1,1000)
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

#### Load data
nx, ny, nt = 256, 256, 72 # size of the longitude, latitude and time
n = 300 # number of generated samples

file = np.load('Data/shallow-water-256x256x72_1.npz')
inputs_1, outputs_1 = file['inputs'], np.array((file['outputs']))

file = np.load('Data/shallow-water-256x256x72_2.npz')
inputs_2, outputs_2 = file['inputs'], np.array((file['outputs']))

inputs = np.concatenate((inputs_1, inputs_2), axis=0).reshape(n, nx*ny) # Concatenate input data 
outputs = np.concatenate((outputs_1, outputs_2), axis=0)  # Concatenate output data 

outputs = outputs[:,:nt,:,:]  

outputs_re = outputs.reshape(n*nt, nx*ny) # Reshape the data to the desired shape

n_samples = inputs.shape[0] # Number of samples
num_train = 230  # Number of training samples
num_test = n - num_train  # Number of testing samples
x_y_data = outputs_re # only outputs

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) # Normalize the data

# Normalize datasets
x_norm = NormalizeData(inputs) # Normalize te input data 
x_y_data_norm = NormalizeData(x_y_data).astype("float32")  # Normalize te output data
y_norm = x_y_data_norm

# Perform PCA for input data
pca = PCA(n_components=args.latent_dim) # Create an instance of the PCA class with desired number of components
x_red = pca.fit_transform(x_norm) # Fit the PCA model to the normalized input data and transform the data into the reduced dimensional space
if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/PCA/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/PCA/')

PCA_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/PCA/'
#plt.show()
#print('All data shape:', x_norm.shape, y_norm.shape)

# Split to train/eval autoencoder
n_train=nt*num_train
n_test=nt*num_test
x_y_train=x_y_data_norm[:n_train,:]
x_y_test=x_y_data_norm[n_train:,:]
# x_y_train, x_y_test = train_test_split(x_y_data_norm, test_size=70/300, random_state=42)
# n_train, n_test, n_all = x_y_train.shape[0], x_y_test.shape[0], x_y_data.shape[0]  # for autoencoder training

# Reshaping the train and test sets
x_y_train = torch.tensor(x_y_train.reshape(n_train, nx, ny))
r=torch.randperm(n_train)
x_y_train = x_y_train[r]
x_y_test = torch.tensor(x_y_test.reshape(n_test, nx, ny))

#### Create directories for results
if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/plots/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/plots/')
if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_AE/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_AE/')
if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/data/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/data/')
if not os.path.exists('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_AE/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_AE/')    
    
class_AE_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_AE/'

#### Run autoencoders
print('Seed number:', seed_value)
print('Autoencoder training in progress...')


 ## Multi-layer perception (MLP)/Fully-connected autoencoder    
class MLAE(nn.Module):
    def __init__(self, latent_dim):
        super(MLAE, self).__init__()
        self.latent_dim = latent_dim
        # Define the encoder network as a sequential module
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nx * ny, 256),
            nn.ReLU(),
            nn.Linear(256, 169),
            nn.ReLU(),
            nn.Linear(169, 121),
            nn.ReLU(),
            nn.Linear(121, latent_dim),
            nn.ReLU()
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
        encoded = self.encoder(x) # Perform encoder
        decoded = self.decoder(encoded)  # Perform decoder
        decoded = decoded.view(-1, nx, ny) # Reshape to the desired shape
        return decoded

# control which parameters are frozen / free for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

#### Autoencoder

autoencoder = MLAE(args.latent_dim)

optimizer = torch.optim.Adam(autoencoder.parameters()) # Construct an optimizer
criterion = nn.MSELoss() # Define MSE loss



# Train model
epochs = args.n_epochs   # epochs for training
batch_size = args.bs     # batch size

loss_history = []
val_loss_history = []

for epoch in range(epochs):
    running_loss = 0.0 # Initialize variables to store training loss
    for i in range(0, len(x_y_train), batch_size):  # Iterate over the training data batches
        inputs_sbatch = torch.Tensor(x_y_train[i:i+batch_size]) # batch

        optimizer.zero_grad() # Clear the gradients of the optimizer

        outputs_sbatch = autoencoder(inputs_sbatch) # Obtain output by autoencoder
        loss = criterion(outputs_sbatch, inputs_sbatch) # Calculate the MSE loss

        loss.backward() # Backward pass: compute gradients of the loss
        optimizer.step()  # Update the model parameters based on the computed gradients

        running_loss += loss.item() 

    loss_history.append(running_loss) # Loss history

    # Evaluate on validation set
    val_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(x_y_test), batch_size):
            inputs_sbatch = torch.Tensor(x_y_test[i:i+batch_size])
            outputs_sbatch = autoencoder(inputs_sbatch)
            val_loss += criterion(outputs_sbatch, inputs_sbatch).item()
    val_loss_history.append(val_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}")

print('Autoencoder training is completed.')

# Compute L2 error on test data
decoded_data = autoencoder(torch.Tensor(x_y_test)).reshape(n_test, nx*ny).detach().numpy() # Calculate the decoded data
reference_data = x_y_test.reshape(n_test, nx*ny).detach().numpy()  # Reshape the target to the desired shape

# L2 error
errors = np.abs(decoded_data - reference_data) # Calculate the error between the output and target labels
l2_rel_err = np.linalg.norm(errors, axis=0)/np.linalg.norm(reference_data, axis=0)  # Calculate relative L2 error
l2 = np.mean(l2_rel_err) # Calculate the mean value of the relative L2 error
print('Autoencoder relative L2 error: {}\n'.format(round(l2,4)))

# Mean squared error (MSE)
mse = mean_squared_error(reference_data, decoded_data) # Calculate the MSE error
print('Mean squared error (MSE): {}\n'.format(round(mse, 4)))

errordir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/errors_AE/'
np.savetxt(
    errordir + 'error_' + str(args.method) + '_seed_' + str(seed_value) + '.txt',
    np.expand_dims(np.array(mse), axis=0),
    fmt='%e'
)


#### Save all data
# Pass ALL data through the encoder and then split
x_y_red = autoencoder.encoder(torch.Tensor(x_y_data_norm.reshape(n*nt, nx, ny))).detach().numpy()  # encoder
norm_min= np.min(x_y_data) # Calculate the minimum value of the output
norm_max= np.max(x_y_data) # Calculate the maximum value of the output
# Train-test
# Split x and y data
y_red = x_y_red.reshape(n,nt*args.latent_dim) # Reshape the reduced output to the desired shape

# Split to train/test
x_train_red, x_test_red = x_red[:num_train], x_red[num_train:] # Split the reduced input to training input and testing input
y_train_red, y_test_red = y_red[:num_train].reshape(num_train, nt, args.latent_dim), y_red[num_train:].reshape(num_test, nt, args.latent_dim) #  Split the reduced output to training output and testing output
x_train, x_test = inputs[:num_train], inputs[num_train:]  # Split the input to training input and testing input
y_train, y_test = outputs[:num_train], outputs[num_train:] #  Split the output to training output and testing output


# Save reduced data (for DeepONet)
data_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/data/'

np.savez(data_dir + 'data_d_{}.npz'.format(args.latent_dim), latent_dim=args.latent_dim, x_train=x_train, x_test=x_test, 
                                                    y_train=y_train, y_test=y_test, 
                                                    x_train_red=x_train_red, x_test_red=x_test_red, 
                                                    y_train_red=y_train_red, y_test_red=y_test_red,norm_max=norm_max, norm_min=norm_min)  
    
# Save autoencoder class (to use decoder later)
class_AE_dir = 'results/d_' + str(args.latent_dim) + '/' + str(args.method) + '/class_AE/'
torch.save(autoencoder,class_AE_dir+'class_AE_{}'.format(args.latent_dim))