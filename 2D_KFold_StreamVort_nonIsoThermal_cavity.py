
# -*- coding: utf-8 -*-
"""

2D lamninar Stream Vortisity version
from scratch version - 11/02/2024


@author: Amirreza
"""
#stage0: Import Libaraies

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import optuna
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

# stage0-0: Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##############################################################################
# stage1: Model craetion
# for each parameter one model is created so each model has  2 layers as input and 1 layer as ouput.


class Swish(nn.Module):
        def __init__(self, inplace=True):
            super(Swish, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            if self.inplace:
                x.mul_(torch.sigmoid(x))
                return x
            else:
                return x * torch.sigmoid(x)


class PINN_psi(nn.Module):

        #The __init__ function stack the layers of the
        #network Sequentially
        def __init__(self):
            super(PINN_psi, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n_psi),
                Swish(),
                nn.Linear(h_n_psi,h_n_psi),
                Swish(),
                nn.Linear(h_n_psi,h_n_psi),
                Swish(),
                nn.Linear(h_n_psi,h_n_psi),
                Swish(),
                nn.Linear(h_n_psi,h_n_psi),
                Swish(),
                nn.Linear(h_n_psi,h_n_psi),
                Swish(),
                nn.Linear(h_n_psi,h_n_psi),
                Swish(),
                nn.Linear(h_n_psi,h_n_psi),
                Swish(),
                nn.Linear(h_n_psi,h_n_psi),
                Swish(),
                nn.Linear(h_n_psi,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        #def forward(self,x):
        def forward(self,x):
            output = self.main(x)
            return output

class PINN_omega(nn.Module):

        #The __init__ function stack the layers of the
        #network Sequentially
        def __init__(self):
            super(PINN_omega, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n_omega),
                Swish(),
                nn.Linear(h_n_omega,h_n_omega),
                Swish(),
                nn.Linear(h_n_omega,h_n_omega),
                Swish(),
                nn.Linear(h_n_omega,h_n_omega),
                Swish(),
                nn.Linear(h_n_omega,h_n_omega),
                Swish(),
                nn.Linear(h_n_omega,h_n_omega),
                Swish(),
                nn.Linear(h_n_omega,h_n_omega),
                Swish(),
                nn.Linear(h_n_omega,h_n_omega),
                Swish(),
                nn.Linear(h_n_omega,h_n_omega),
                Swish(),
                nn.Linear(h_n_omega,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        #def forward(self,x):
        def forward(self,x):
            output = self.main(x)
            return output

class PINN_T(nn.Module):

        #The __init__ function stack the layers of the
        #network Sequentially
        def __init__(self):
            super(PINN_T, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n_T),
                Swish(),
                nn.Linear(h_n_T,h_n_T),
                Swish(),
                nn.Linear(h_n_T,h_n_T),
                Swish(),
                nn.Linear(h_n_T,h_n_T),
                Swish(),
                nn.Linear(h_n_T,h_n_T),
                Swish(),
                nn.Linear(h_n_T,h_n_T),
                Swish(),
                nn.Linear(h_n_T,h_n_T),
                Swish(),
                nn.Linear(h_n_T,h_n_T),
                Swish(),
                nn.Linear(h_n_T,h_n_T),
                Swish(),
                nn.Linear(h_n_T,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        #def forward(self,x):
        def forward(self,x):
            output = self.main(x)
            return output

#Stage1-0: defin number on input layer & neurons in each layer
input_n = 2
h_n_psi = 40
h_n_omega = 40
h_n_T = 40

# stage1-1: PINN_psi is model(networks for each components)
PINN_psi = PINN_psi().to(device)
PINN_omega = PINN_omega().to(device)
PINN_T = PINN_T().to(device)

##############################################################################
#Stage2: Function definitions:

#stage 2-0 : function for normalize inputs
def normal_inputs(df): #df is a dataframe
    normal_df = (2 * (df - df.min()) / (df.max() - df.min() )) - 1
    return normal_df

#stage2-1 : Add gausian noise to data for improving and boosting the model
def add_gaussian_noise(tensor, mean=1, std_dev=1.01):
    noise = torch.normal(mean, std_dev, size=tensor.shape)
    return tensor + noise

#stage2-2: PDE residual loss calculation
#important: in this case input(X) is NOT normalized and it will be done inside the function
def residula_loss(PINN_psi , PINN_omega  , PINN_T , x , y ):

    x = (2 * (x - x.min()) / (x.max() - x.min())) - 1
    y = (2 * (y - y.min()) / (y.max() - y.min())) - 1

    # Momentum equations
    nu = 0.01 #kinematic viscosity
    g = 9.8
    alpha = 0.002
    T_surf = 313
    T_inf = 303
    betha  = 1/ T_inf
    length = 10
    u_ref = 1
    Pr = 1
    Re = ((g * betha) / (nu * alpha)) * (T_surf - T_inf) * (length ** 3)
    Ra = u_ref * length / nu

    psi = PINN_psi(torch.cat((x,y) , dim = 1))
    T = PINN_T(torch.cat((x,y) , dim = 1))
    omega = PINN_omega(torch.cat((x , y) , dim = 1))

    # Calculate gradients
    psi_x = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True )[0]
    psi_y = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True )[0]

    psi_xx = torch.autograd.grad(psi_x, x, grad_outputs=torch.ones_like(psi_x), create_graph=True )[0]
    psi_yy = torch.autograd.grad(psi_y, y, grad_outputs=torch.ones_like(psi_y), create_graph=True )[0]


    omega_x = torch.autograd.grad(omega, x, grad_outputs=torch.ones_like(omega), create_graph=True)[0]
    omega_y = torch.autograd.grad(omega, y, grad_outputs=torch.ones_like(omega), create_graph=True)[0]

    omega_xx = torch.autograd.grad(omega_x, x, grad_outputs=torch.ones_like(omega_x), create_graph=True)[0]
    omega_yy = torch.autograd.grad(omega_y, y, grad_outputs=torch.ones_like(omega_y), create_graph=True)[0]


    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]

    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True)[0]

    poisson_residual = omega + (psi_xx + psi_yy)
    vorticity_residual =  psi_y * omega_x - psi_x * omega_y - (1/Re) * (omega_xx + omega_yy) - Ra * T_x
    energy_residual =  psi_y * T_x - psi_x * T_y - (1/(Re * Pr)) * (T_xx + T_yy)

    loss_mse = nn.MSELoss()
    #Note our target is zero. It is residual so we use zeros_like
    loss_pde = loss_mse(poisson_residual,torch.zeros_like(poisson_residual)) + loss_mse(vorticity_residual,torch.zeros_like(vorticity_residual)) + loss_mse(energy_residual,torch.zeros_like(energy_residual))
    return loss_pde

#Stage2-3: Boundary loss (important: in this case input(X) must be normalized oustside of the function)
#inputs are x and y size are [m ,1] , output is [m ,(u ,v ,T)]
def boundary_condition_loss(PINN_psi , PINN_omega , PINN_T , x , y , output):

    psi_b_pred = PINN_psi(torch.cat((x , y) ,dim = 1))
    omega_b_pred = PINN_omega(torch.cat((x , y) ,dim = 1))
    T_b_pred = PINN_T(torch.cat((x , y) ,dim = 1))

    #u = d(psi) / d(y)
    #psi_b_pred.retain_grad()
    u_b_pred = torch.autograd.grad(psi_b_pred, y , grad_outputs=torch.ones_like(psi_b_pred) , create_graph=True)[0]
    #v = - d(psi) / d(x)
    v_b_pred = -torch.autograd.grad(psi_b_pred, x , grad_outputs=torch.ones_like(psi_b_pred), create_graph=True)[0]

    u_b = output[: , 0].reshape(-1 , 1)
    v_b = output[: , 1].reshape(-1 , 1)
    T_b = output[: , 2].reshape(-1 , 1)

    loss_mse = nn.MSELoss()
    loss_u_b = loss_mse(u_b_pred  , u_b)
    loss_v_b = loss_mse(v_b_pred  , v_b)
    loss_T_b = loss_mse(T_b_pred  , T_b)

    loss_bc = loss_T_b + loss_u_b + loss_v_b
    return loss_bc

#Stage2-4: Interior data loss (important: in this case input(X) must be normalized oustside of the function)
def data_loss(PINN_psi , PINN_omega  , PINN_T , x , y , output):


    psi_pred = PINN_psi(torch.cat((x , y) ,dim = 1))
    omega_pred = PINN_omega(torch.cat((x , y) ,dim = 1))
    T_d_pred = PINN_T(torch.cat((x , y) ,dim = 1))

    u_d = output[: , 0].reshape(-1 , 1)
    v_d = output[: , 1].reshape(-1 , 1)
    T_d = output[: , 2].reshape(-1 , 1)

    u_d_pred = torch.autograd.grad(psi_pred, y, grad_outputs=torch.ones_like(psi_pred), create_graph=True )[0]
    v_d_pred = -torch.autograd.grad(psi_pred, x , grad_outputs=torch.ones_like(psi_pred) , create_graph=True)[0]

    loss_mse = nn.MSELoss()
    loss_u_d = loss_mse(u_d_pred  , u_d)
    loss_v_d = loss_mse(v_d_pred  , v_d)
    loss_T_d = loss_mse(T_d_pred  , T_d)

    loss_data = loss_u_d + loss_v_d  + loss_T_d
    return loss_data

#stage2-5: Noisy data loss calculation
def noisy_data_loss(PINN_psi , PINN_omega  , PINN_T , x , y , output):


    x_d_noisy = add_gaussian_noise(x.reshape(-1 , 1))
    y_d_noisy = add_gaussian_noise(y.reshape(-1 , 1))
    u_d_noisy = add_gaussian_noise(output[: , 0])
    v_d_noisy = add_gaussian_noise(output[: , 1])
    T_d_noisy = add_gaussian_noise(output[: , 2])

    psi_noisy_pred = PINN_psi(torch.cat((x_d_noisy,y_d_noisy) , dim = 1))
    u_d_noisy_pred  = torch.autograd.grad(psi_noisy_pred, y_d_noisy, grad_outputs=torch.ones_like(psi_noisy_pred), create_graph=True )[0]
    v_d_noisy_pred  = -torch.autograd.grad(psi_noisy_pred, x_d_noisy, grad_outputs=torch.ones_like(psi_noisy_pred), create_graph=True)[0]
    T_d_noisy_pred = PINN_T(torch.cat((x_d_noisy,y_d_noisy) , dim = 1))

    loss_mse = nn.MSELoss()
    loss_u_d_noisy = loss_mse(u_d_noisy_pred  , u_d_noisy.reshape(-1 , 1))
    loss_v_d_noisy = loss_mse(v_d_noisy_pred  , v_d_noisy.reshape(-1 , 1))
    loss_T_d_noisy = loss_mse(T_d_noisy_pred  , T_d_noisy.reshape(-1 , 1))

    loss_noisy_data = loss_u_d_noisy + loss_v_d_noisy + loss_T_d_noisy
    return loss_noisy_data

#stage2-6: Total Loss calculation:
def total_loss(PINN_psi, PINN_omega , PINN_T  , x_c , y_c,
               x_b , y_b  ,output_b, x_d ,y_d , output_d):
    pde_loss = residula_loss(PINN_psi , PINN_omega  , PINN_T , x_c , y_c )
    bc_loss = boundary_condition_loss(PINN_psi , PINN_omega  , PINN_T , x_b , y_b , output_b)
    interior_loss = data_loss(PINN_psi , PINN_omega  , PINN_T , x_d , y_d , output_d)
    noisy_loss = noisy_data_loss(PINN_psi , PINN_omega  , PINN_T , x_d , y_d , output_d)
    loss = (pde_loss * lambda_pde) + (bc_loss * lambda_bc) + (interior_loss * lambda_interior) + (noisy_loss * w_noise)

    return loss

#stage2-7:  << plot >>
def plot_results(PINN_u , PINN_v , PINN_T ,  ff):
    PINN_psi.eval()
    PINN_omega.eval()
    PINN_T.eval()


    #df = normal_inputs(pd.read_csv(file_tset))
    df = pd.read_csv(ff)
    df = 2 * ((df - df.min()) / (df.max() - df.min())) - 1
    x = torch.tensor(df[['x']].values, dtype=torch.float32 , requires_grad = True)
    y = torch.tensor(df[['y']].values, dtype=torch.float32 , requires_grad = True)
    truth = torch.tensor(df[['u' ,'v' ,'T' ,'p']].values, dtype=torch.float32 , requires_grad = True)

    psi_pred = PINN_psi(torch.cat((x , y) , dim = 1))
    omega_pred = PINN_omega(torch.cat((x , y) , dim = 1))
    T_pred = PINN_T(torch.cat((x , y) , dim = 1))
    u_pred = torch.autograd.grad(psi_pred, y, grad_outputs=torch.ones_like(psi_pred), create_graph=True)[0]
    v_pred = -torch.autograd.grad(psi_pred,x ,grad_outputs=torch.ones_like(psi_pred), create_graph=True)[0]

    fig, ax= plt.subplots(nrows=3 , ncols=1 , figsize=(10, 14) , sharex = True)


    ax[0].set_title("u velocity")
    ax[0].plot(truth[: ,0].detach().numpy() , label = "Exact")
    ax[0].plot(u_pred.detach().numpy() , label= "PINN")
    ax[0].legend(loc="upper right")


    ax[1].set_title("v velocity")
    ax[1].plot(truth[: ,1].detach().numpy() , label = "Exact")
    ax[1].plot(v_pred.detach().numpy() , label= "PINN")
    ax[1].legend(loc="upper right")

    ax[2].set_title("T velocity")
    ax[2].plot(truth[: ,2].detach().numpy() , label = "Exact")
    ax[2].plot(T_pred.detach().numpy() , label= "PINN")
    ax[2].legend(loc="upper right")

    fig.suptitle(f'Comparison With Unseen Data : {ff}')
    #fig.savefig("Results/Comparison_plot" + time.strftime("%Y-%m-%d %H%M%S") + ".png")
    plt.show()


#stage3-2: LBGF-s
opt_T_lbfgs=torch.optim.LBFGS(PINN_T.parameters(),
  lr=0.01,  # or adjust based on your problem
  max_iter=5000,  # More iterations for better convergence
  max_eval=None,  # Default
  tolerance_grad=1e-7,  # Increase sensitivity to gradients
  tolerance_change=1e-9,  # Keep default unless facing early stops
  history_size=100,  # Use larger history for better approximations
  line_search_fn="strong_wolfe")  # Use strong Wolfe line search for better convergence

opt_psi_lbfgs=torch.optim.LBFGS(PINN_psi.parameters(),
  lr=0.01,  # or adjust based on your problem
  max_iter=5000,  # More iterations for better convergence
  max_eval=None,  # Default
  tolerance_grad=1e-7,  # Increase sensitivity to gradients
  tolerance_change=1e-9,  # Keep default unless facing early stops
  history_size=100,  # Use larger history for better approximations
  line_search_fn="strong_wolfe")  # Use strong Wolfe line search for better convergence

opt_omega_lbfgs=torch.optim.LBFGS(PINN_omega.parameters(),
  lr=0.01,  # or adjust based on your problem
  max_iter=5000,  # More iterations for better convergence
  max_eval=None,  # Default
  tolerance_grad=1e-7,  # Increase sensitivity to gradients
  tolerance_change=1e-9,  # Keep default unless facing early stops
  history_size=100,  # Use larger history for better approximations
  line_search_fn="strong_wolfe")  # Use strong Wolfe line search for better convergence
############################# finish OPTIMIZERS ##########################################################/


#Stage4: Training

# Dynamic learning rate and optimizer switch
def train_pinn(PINN_psi, PINN_omega , PINN_T , filename_data,
    filename_bc  ,epochs_adam , epoch_lbgfs , num_collocation_points  ):
    print("Train starting . . . ")
    # Simulate a long process
    for i in tqdm(range(1)):
        time.sleep(0.00005)  # Simulating work by sleeping

    tic = time.time()

    #Stage4-0: Collocation points definition

    x_min = 0
    y_min = 0

    x_max = 10
    y_max = 5

    # Define cube boundaries within the domain (example values)
    cube_x_min, cube_x_max = 4.5, 5.5  # x bounds of cube
    cube_y_min, cube_y_max = 0, 1  # y bounds of cube

    # Generate random collocation points within the domain
    np.random.seed(50)

    collocation_points = np.random.rand(num_collocation_points, 2)
    collocation_points[:, 0] = collocation_points[:, 0] * (x_max - x_min) + x_min  # Scale to x bounds
    collocation_points[:, 1] = collocation_points[:, 1] * (y_max - y_min) + y_min  # Scale to y bounds

    # Filter out points that fall within the cube's region
    filtered_points = collocation_points[
        ~(
        (collocation_points[:, 0] >= cube_x_min) &
        (collocation_points[:, 0] <= cube_x_max) &
        (collocation_points[:, 1] >= cube_y_min) &
        (collocation_points[:, 1] <= cube_y_max)
        )]


    collocation_points_tensor = torch.tensor(filtered_points, dtype=torch.float32 ,  requires_grad=True)
    X_c = collocation_points_tensor[: , 0].reshape(-1 , 1)
    Y_c = collocation_points_tensor[: , 1].reshape(-1 , 1)

    #TEMPORARY : using noisy collocation points

    X_c_noisy = add_gaussian_noise(collocation_points_tensor[: , 0]).reshape(-1 , 1)
    Y_c_noisy = add_gaussian_noise(collocation_points_tensor[: , 1]).reshape(-1 , 1)

    #stage4-1: import interior data & boundary data
    interior_data = normal_inputs(pd.read_csv(filename_data))
    boundary_data = normal_inputs(pd.read_csv(filename_bc))

    x_interior = torch.tensor(interior_data[['x']].values, dtype=torch.float32 , requires_grad= True )
    y_interior = torch.tensor(interior_data[['y']].values, dtype=torch.float32 , requires_grad= True )
    output_interior = torch.tensor(interior_data[['u' ,'v' ,'T' ]].values, requires_grad= True ,dtype=torch.float32 )

    x_boundary = torch.tensor(boundary_data[['x']].values, dtype=torch.float32 , requires_grad= True )
    y_boundary = torch.tensor(boundary_data[['y']].values, dtype=torch.float32 , requires_grad= True )
    output_boundary = torch.tensor(boundary_data[['u' ,'v' ,'T' ]].values, requires_grad= True , dtype=torch.float32 )

    #Stage4-2 : Hyperparameter tuning with Optuna
    def objective(trial):

        # Hyperparameters for tuning
        lambda_pde = trial.suggest_float("lambda_pde", 1, 10)
        lambda_interior = trial.suggest_float("lambda_interior", 0 ,10)
        lambda_bc = trial.suggest_float("lambda_bc", 0, 10)


        opt_psi_adam = optim.Adam(PINN_psi.parameters() , lr = 1e-3)
        opt_omega_adam = optim.Adam(PINN_omega.parameters() , lr = 1e-3)
        opt_T_adam = optim.Adam(PINN_T.parameters() , lr = 1e-3)

        num_epochs_trial = 500  #best value is 500
        for epoch in range(num_epochs_trial):

            opt_psi_adam.zero_grad()
            opt_omega_adam.zero_grad()
            opt_T_adam.zero_grad()

            pde_loss = residula_loss(PINN_psi , PINN_omega  , PINN_T , X_c , Y_c )
            bc_loss = boundary_condition_loss(PINN_psi , PINN_omega  , PINN_T , x_boundary , y_boundary , output_boundary)
            interior_loss = data_loss(PINN_psi , PINN_omega  , PINN_T , x_interior , y_interior , output_interior)
            #noisy_loss = noisy_data_loss(PINN_psi , PINN_omega  , PINN_T , x_interior , y_interior , output_interior)
            noisy_loss = residula_loss(PINN_psi , PINN_omega  , PINN_T , X_c_noisy , Y_c_noisy )
            loss = (pde_loss * lambda_pde) + (bc_loss * lambda_bc) + (interior_loss * lambda_interior) + (noisy_loss * w_noise)

            loss.backward()

            opt_psi_adam.step()
            opt_omega_adam.step()
            opt_T_adam.step()

            # Return the final loss for this trial
            return loss.item()

    num_trials= 50
    # Run the Optuna hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials = num_trials)  # Adjust n_trials for more thorough search

    os.system('cls')
    # Extract the best lambda values
    best_params = study.best_params
    print("Optimized lambda_pde:", best_params["lambda_pde"])
    print("Optimized lambda_interior:", best_params["lambda_interior"])
    print("optimized lambda_bc:", best_params["lambda_bc"])

    lambda_pde = best_params["lambda_pde"]
    lambda_interior = best_params["lambda_interior"]
    lambda_bc = best_params["lambda_bc"]


    #stage4-3: Cross Validation

    fold_loss = []  # Define fold_loss here to store the validation loss of each fold
    k = 3
    kf = KFold(n_splits=k)
    fold_results = []

    for fold, (train_index, val_index) in enumerate(kf.split(x_interior)):

        # Split interior data for training and validation
        #CAUTION_ revise these tensors with : sourceTEnsor.clone().detach().requires_grad_(True)
        #x_train_interior = torch.tensor(x_interior[train_index], dtype=torch.float32 , requires_grad = True)
        x_train_interior = x_interior[train_index].clone().detach().requires_grad_(True)
        y_train_interior = y_interior[train_index].clone().detach().requires_grad_(True)
        output_train_interior = output_interior[train_index].clone().detach().requires_grad_(True)

        x_train_boundary = x_boundary[train_index].clone().detach().requires_grad_(True)
        y_train_boundary = y_boundary[train_index].clone().detach().requires_grad_(True)
        output_train_boundary = output_boundary[train_index].clone().detach().requires_grad_(True)


        x_val_interior = x_interior[val_index].clone().detach().requires_grad_(True)
        y_val_interior = y_interior[val_index].clone().detach().requires_grad_(True)
        output_val_interior = output_interior[val_index].clone().detach().requires_grad_(True)
        x_val_boundary = x_boundary[val_index].clone().detach().requires_grad_(True)
        y_val_boundary = y_boundary[val_index].clone().detach().requires_grad_(True)
        output_val_boundary = output_boundary[val_index].clone().detach().requires_grad_(True)
        

        
        #stage3: OPTIMIZERS (ADAM & LBGF-s)
        #stage3-0: Adam

        optimizer_adam_T = optim.Adam(PINN_T.parameters(), lr=0.001)
        optimizer_adam_psi = optim.Adam(PINN_psi.parameters(), lr=0.001)
        optimizer_adam_omega = optim.Adam(PINN_omega.parameters(), lr=0.001)

        #stage3-1: Adam Reduce LR on PLateaue
        scheduler_T = ReduceLROnPlateau(optimizer_adam_T , factor = 0.5 , min_lr = 1e-2 , verbose=False )
        scheduler_psi = ReduceLROnPlateau(optimizer_adam_psi , factor = 0.5 , min_lr = 1e-2 , verbose=False )
        scheduler_omega = ReduceLROnPlateau(optimizer_adam_omega , factor = 0.5 , min_lr = 1e-2 , verbose=False )
            

        loss_history = []
        loss_val_history = []
        for epoch in range(epochs_adam):

            PINN_T.train()
            PINN_psi.train()
            PINN_omega.train()

            optimizer_adam_T.zero_grad()
            optimizer_adam_psi.zero_grad()
            optimizer_adam_omega.zero_grad()

            pde_loss = residula_loss(PINN_psi , PINN_omega  , PINN_T , X_c , Y_c )
            bc_loss = boundary_condition_loss(PINN_psi , PINN_omega  , PINN_T , x_train_boundary , y_train_boundary , output_train_boundary)
            interior_loss = data_loss(PINN_psi , PINN_omega  , PINN_T , x_train_interior , y_train_interior , output_train_interior)
            #noisy_loss = noisy_data_loss(PINN_psi , PINN_omega  , PINN_T , x_train_interior , y_train_interior , output_train_interior)
            noisy_loss = residula_loss(PINN_psi , PINN_omega  , PINN_T , X_c_noisy , Y_c_noisy )
            loss = (pde_loss * lambda_pde) + (bc_loss * lambda_bc) + (interior_loss * lambda_interior) + (noisy_loss * w_noise)

            # Backpropagation
            loss.backward()


            optimizer_adam_T.step()
            optimizer_adam_psi.step()
            optimizer_adam_omega.step()

            scheduler_T.step(loss)
            scheduler_psi.step(loss)
            scheduler_omega.step(loss)


            if epoch % 500 == 0:
                print(f'Epoch Adam {epoch}/{epochs_adam} [Fold:{fold}] [{100 * epoch/epochs_adam :.2f}%]  ,Total Loss: {loss.item():.6f}  ')
                print(f"Loss PDE: {pde_loss.item():.4f}  |   loss Data: {interior_loss.item():.4f}  |   BC loss: {bc_loss.item():.4f}")
                print(f"======================================================================")

            if loss.item() < 0.0001:
                print(f" loss values is {loss.item():.4f} so optimization switches to LBGF-S . . . ")
                break
            loss_history.append(loss.item())
            #stage5: Validation

            #PINN_T.eval()
            #PINN_omega.eval()
            #PINN_psi.eval()

            # Compute predictions and loss for LBGFS optimization
            pde_loss = residula_loss(PINN_psi , PINN_omega  , PINN_T , X_c , Y_c )
            bc_loss = boundary_condition_loss(PINN_psi , PINN_omega  , PINN_T , x_val_boundary , y_val_boundary , output_val_boundary)
            interior_loss = data_loss(PINN_psi , PINN_omega  , PINN_T , x_val_interior , y_val_interior , output_val_interior)
            noisy_loss = noisy_data_loss(PINN_psi , PINN_omega  , PINN_T , x_val_interior , y_val_interior , output_val_interior)
            loss_val = (pde_loss * lambda_pde) + (bc_loss * lambda_bc) + (interior_loss * lambda_interior) + (noisy_loss * w_noise)
            loss_val_history.append(loss_val.item())



    plt.plot(loss_history , label = "LOSS train")
    plt.plot(loss_val_history , label = "LOSS val")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss History ")
    plt.legend()
    plt.ylim(0 , 1)
    plt.show()


    toc_batch = time.time()
    duration_time = toc_batch - tic
    print(f"Batch_Duration :{duration_time :.2f} s")

    # Compute overall cross-validation loss
    cross_val_loss = sum(fold_loss) / (len(fold_loss) + 1e-10)
    print(f"[Fold{fold / k :.2f}] , overall cross-validation loss: {cross_val_loss:.5f}")


    toc = time.time()
    elapseTime = toc - tic
    print ("elapse time in parallel = ", str(round(elapseTime , 4)) + " s")


filename_data = r'Data_PINN_Cavity_hetaFlux.csv'
filename_bc =  r'BC_PINN_Cavity_hetaFlux.csv'
f_test = r'Test_PINN_Cavity_hetaFlux.csv'

w_noise = 0
epochs_adam = 3000
epoch_lbgfs = 1
num_collocation_points = 5000


train_pinn(PINN_psi, PINN_omega , PINN_T , filename_data,
    filename_bc  ,epochs_adam , epoch_lbgfs , num_collocation_points  )

"""
# Stage:
#save model

torch.save(PINN_psi , "pinn_model_psi_full.pth")
torch.save(PINN_omega , " pinn_model_omega_full.pth")
torch.save(PINN_T , "pinn_model_T_full.pth")
"""

plot_results(PINN_psi, PINN_omega , PINN_T, f_test)

plot_results(PINN_psi, PINN_omega , PINN_T, filename_data)



"""
        def closure_psi():

            #optimizer_adam_T.zero_grad()
            optimizer_adam_psi.zero_grad()
            #optimizer_adam_omega.zero_grad()

            # Compute predictions and loss for LBGFS optimization
            pde_loss = residula_loss(PINN_psi , PINN_omega  , PINN_T , X_c , Y_c )
            bc_loss = boundary_condition_loss(PINN_psi , PINN_omega  , PINN_T , x_train_boundary , y_train_boundary , output_train_boundary)
            interior_loss = data_loss(PINN_psi , PINN_omega  , PINN_T , x_train_interior , y_train_interior , output_train_interior)
            #noisy_loss = noisy_data_loss(PINN_psi , PINN_omega  , PINN_T , x_train_interior , y_train_interior , output_train_interior)
            noisy_loss = residula_loss(PINN_psi , PINN_omega  , PINN_T , X_c_noisy , Y_c_noisy )
            loss = (pde_loss * lambda_pde) + (bc_loss * lambda_bc) + (interior_loss * lambda_interior) + (noisy_loss * w_noise)

            loss.backward()

            # Log the loss to track over time
            loss_history.append(loss.item())

            return loss


        def closure_omega():

            optimizer_adam_omega.zero_grad()

            # Compute predictions and loss for LBGFS optimization
            pde_loss = residula_loss(PINN_psi , PINN_omega  , PINN_T , X_c , Y_c )
            bc_loss = boundary_condition_loss(PINN_psi , PINN_omega  , PINN_T , x_train_boundary , y_train_boundary , output_train_boundary)
            interior_loss = data_loss(PINN_psi , PINN_omega  , PINN_T , x_train_interior , y_train_interior , output_train_interior)
            #noisy_loss = noisy_data_loss(PINN_psi , PINN_omega  , PINN_T , x_train_interior , y_train_interior , output_train_interior)
            noisy_loss = residula_loss(PINN_psi , PINN_omega  , PINN_T , X_c_noisy , Y_c_noisy )
            loss = (pde_loss * lambda_pde) + (bc_loss * lambda_bc) + (interior_loss * lambda_interior) + (noisy_loss * w_noise)

            loss.backward()

            # Log the loss to track over time
            loss_history.append(loss.item())

            return loss

        def closure_T():
            optimizer_adam_T.zero_grad()

            # Compute predictions and loss for LBGFS optimization
            pde_loss = residula_loss(PINN_psi , PINN_omega  , PINN_T , X_c , Y_c )
            bc_loss = boundary_condition_loss(PINN_psi , PINN_omega  , PINN_T , x_train_boundary , y_train_boundary , output_train_boundary)
            interior_loss = data_loss(PINN_psi , PINN_omega  , PINN_T , x_train_interior , y_train_interior , output_train_interior)
            #noisy_loss = noisy_data_loss(PINN_psi , PINN_omega  , PINN_T , x_train_interior , y_train_interior , output_train_interior)
            noisy_loss = residula_loss(PINN_psi , PINN_omega  , PINN_T , X_c_noisy , Y_c_noisy )
            loss = (pde_loss * lambda_pde) + (bc_loss * lambda_bc) + (interior_loss * lambda_interior) + (noisy_loss * w_noise)

            loss.backward()

            # Log the loss to track over time
            loss_history.append(loss.item())




        # Configure L-BFGS optimizer and optimize

        for epoch in range(epoch_lbgfs):

            loss_T = opt_T_lbfgs.step(closure_T)
            opt_psi_lbfgs.step(closure_psi)
            opt_omega_lbfgs.step(closure_omega)

            if epoch % 10 == 0:
                print(f'Epoch LBGF-S {epoch}/{epoch_lbgfs} [Fold:{fold}] [{100 * epoch/epoch_lbgfs :.2f}%]  ,Total Loss: {loss.item():.6f}  ')

"""
