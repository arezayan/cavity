
# -*- coding: utf-8 -*-
"""

2D lamninar Stream Vortisity version
from scratch version - 11/02/2024
# cavity

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
from sklearn.model_selection import train_test_split


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


class PINN_u(nn.Module):

        #The __init__ function stack the layers of the
        #network Sequentially
        def __init__(self):
            super(PINN_u, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),


                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),


                Swish(),

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        #def forward(self,x):
        def forward(self,x):
            output = self.main(x)
            return output

class PINN_v(nn.Module):

        #The __init__ function stack the layers of the
        #network Sequentially
        def __init__(self):
            super(PINN_v, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),


                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        #def forward(self,x):
        def forward(self,x):
            output = self.main(x)
            return output

class PINN_w(nn.Module):

        #The __init__ function stack the layers of the
        #network Sequentially
        def __init__(self):
            super(PINN_w, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),


                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        #def forward(self,x):
        def forward(self,x):
            output = self.main(x)
            return output

class PINN_p(nn.Module):

        #The __init__ function stack the layers of the
        #network Sequentially
        def __init__(self):
            super(PINN_p, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
            return  output

class PINN_T(nn.Module):

        #The __init__ function stack the layers of the
        #network Sequentially
        def __init__(self):
            super(PINN_T, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.BatchNorm1d(h_n),
                Swish(),
                nn.Linear(h_n,h_n),
                #nn.Tanh(),
                #nn.Sigmoid(),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),
                nn.Linear(h_n,h_n),

                #nn.BatchNorm1d(h_n),

                Swish(),

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
            return  output
#Stage1-0: defin number on input layer & neurons in each layer
input_n = 2
h_n = 40
#h_n_psi = 40
#h_n_omega = 40
#h_n_T = 40

# stage1-1: PINN_psi is model(networks for each components)
PINN_u = PINN_u().to(device)
PINN_v = PINN_v().to(device)
PINN_p = PINN_p().to(device)
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
def residula_loss(PINN_u , PINN_v , PINN_p , PINN_T , x , y ):

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

    u = PINN_u(torch.cat((x,y) , dim = 1))
    v = PINN_v(torch.cat((x,y) , dim = 1))
    p = PINN_p(torch.cat((x,y) , dim = 1))
    T = PINN_T(torch.cat((x,y) , dim = 1))


    # Calculate gradients

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True )[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True )[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True )[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True )[0]


    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]

    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True)[0]


    # Continuity equation
    continuity_residual = u_x + v_y
    # Momentum Residuals
    momentum_u_residual = (u * u_x + v * u_y ) + p_x - nu * (u_xx + u_yy)
    momentum_v_residual = (u * v_x + v * v_y ) + p_y - nu * (v_xx + v_yy)
    energy_residual = u * T_x + v * T_y  - alpha * (T_xx + T_yy ) #- alpha_t_diff

    loss_mse = nn.MSELoss()
    #Note our target is zero. It is residual so we use zeros_like
    loss_pde = loss_mse(continuity_residual,torch.zeros_like(continuity_residual)) + loss_mse(momentum_u_residual,torch.zeros_like(momentum_u_residual)) + loss_mse(momentum_v_residual,torch.zeros_like(momentum_v_residual)) + loss_mse(energy_residual,torch.zeros_like(energy_residual))
    return loss_pde



#Stage2-4: Interior data loss (important: in this case input(X) must be normalized oustside of the function)
def data_loss(PINN_u , PINN_v , PINN_p , PINN_T , x , y , output):
    #outputfile [T u v p]

    u_pred = PINN_u(torch.cat((x , y) ,dim = 1))
    v_pred = PINN_v(torch.cat((x , y) ,dim = 1))
    p_pred = PINN_p(torch.cat((x , y) ,dim = 1))
    T_pred = PINN_T(torch.cat((x , y) ,dim = 1))

    u_d = output[: , 0].reshape(-1 , 1)
    v_d = output[: , 1].reshape(-1 , 1)
    p_d = output[: , 3].reshape(-1 , 1)
    T_d = output[: , 2].reshape(-1 , 1)

    loss_mse = nn.MSELoss()
    loss_u_d = loss_mse(u_pred  , u_d)
    loss_v_d = loss_mse(v_pred  , v_d)
    loss_p_d = loss_mse(p_pred  , p_d)
    loss_T_d = loss_mse(T_pred  , T_d)

    loss_data = loss_u_d + loss_v_d  + loss_p_d + loss_T_d
    return loss_data

#stage2-5: Noisy data loss calculation
def noisy_data_loss(PINN_u , PINN_v , PINN_p , PINN_T, x , y ):


    x_d_noisy = add_gaussian_noise(x.reshape(-1 , 1))
    y_d_noisy = add_gaussian_noise(y.reshape(-1 , 1))

    loss_noisy_PDE = residula_loss(PINN_u , PINN_v , PINN_p , PINN_T , x_d_noisy , y_d_noisy )

    return loss_noisy_PDE

#stage2-6: Total Loss calculation:
def total_loss(PINN_u , PINN_v , PINN_p , PINN_T  , x_c , y_c,
               x_d ,y_d , output_d , x_noisy , y_noisy):
    pde_loss = residula_loss(PINN_u , PINN_v , PINN_p , PINN_T , x_c , y_c )
    interior_loss = data_loss(PINN_u , PINN_v , PINN_p , PINN_T, x_d , y_d , output_d)
    noisy_loss = noisy_data_loss(PINN_u , PINN_v , PINN_p , PINN_T , x_noisy , y_noisy )
    loss = (pde_loss * lambda_pde) + (interior_loss * lambda_interior) + (noisy_loss * w_noise)

    return loss

def impose_boundary_conditions(PINN_u , PINN_v , PINN_T , PINN_p, x_coords, y_coords , boundary_values):
    # Directly set model predictions to boundary values (hard enforcement)
    with torch.no_grad():
        # Boundary conditions for u, v, T, and p
        PINN_u.eval()
        PINN_v.eval()
        PINN_T.eval()
        PINN_p.eval()

        PINN_u(torch.cat((x_coords , y_coords) , dim = 1)).copy_(boundary_values[: , 0].reshape(-1 , 1))
        PINN_v(torch.cat((x_coords , y_coords) , dim = 1)).copy_(boundary_values[: , 1].reshape(-1 , 1))
        PINN_T(torch.cat((x_coords , y_coords) , dim = 1)).copy_(boundary_values[: , 2].reshape(-1 , 1))
        PINN_p(torch.cat((x_coords , y_coords) , dim = 1)).copy_(boundary_values[: , 3].reshape(-1 , 1))

        PINN_u.train()
        PINN_v.train()
        PINN_T.train()
        PINN_p.train()

#stage2-7:  << plot >>
def plot_results(PINN_u , PINN_v , PINN_p , PINN_T,  ff , epoch):
    PINN_u.eval()
    PINN_v.eval()
    PINN_p.eval()
    PINN_T.eval()


    #df = normal_inputs(pd.read_csv(file_tset))
    df = pd.read_csv(ff)
    df = 2 * ((df - df.min()) / (df.max() - df.min())) - 1
    x = torch.tensor(df[['x']].values, dtype=torch.float32 , requires_grad = True)
    y = torch.tensor(df[['y']].values, dtype=torch.float32 , requires_grad = True)
    truth = torch.tensor(df[['u' ,'v' ,'T' , 'p']].values, dtype=torch.float32 , requires_grad = True)

    u_pred = PINN_u(torch.cat((x , y) , dim = 1))
    v_pred = PINN_v(torch.cat((x , y) , dim = 1))
    T_pred = PINN_T(torch.cat((x , y) , dim = 1))
    p_pred = PINN_p(torch.cat((x , y) , dim = 1))


    fig, ax= plt.subplots(nrows=4 , ncols=1  ,dpi = 100 , sharex = True)


    ax[0].set_title("u velocity")
    ax[0].plot(truth[: ,0].detach().numpy() , label = "Exact")
    ax[0].plot(u_pred.detach().numpy() , label= "PINN")
    ax[0].legend(loc="best")


    ax[1].set_title("v velocity")
    ax[1].plot(truth[: ,1].detach().numpy() , label = "Exact")
    ax[1].plot(v_pred.detach().numpy() , label= "PINN")
    ax[1].legend(loc="best")

    ax[2].set_title("T velocity")
    ax[2].plot(truth[: ,2].detach().numpy() , label = "Exact")
    ax[2].plot(T_pred.detach().numpy() , label= "PINN")
    ax[2].legend(loc="best")

    ax[3].set_title("p velocity")
    ax[3].plot(truth[: ,3].detach().numpy() , label = "Exact")
    ax[3].plot(p_pred.detach().numpy() , label= "PINN")
    ax[3].legend(loc="best")


    fig.suptitle(f'comp PLot : {ff}  | EpochNo:{epoch}')
    #fig.savefig("Results/Comparison_plot" + time.strftime("%Y-%m-%d %H%M%S") + ".png")
    plt.show()

def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)
def init_xavier(m):
    if isinstance(m, nn.Linear):  # Check if the layer is nn.Linear
        nn.init.xavier_normal_(m.weight)  # Xavier normal distribution
        if m.bias is not None:           # Initialize biases to zero if present
            nn.init.zeros_(m.bias)

#stage3: OPTIMIZERS (ADAM & LBGF-s)


#stage3-2: LBGF-s
opt_T_lbfgs=torch.optim.LBFGS(PINN_T.parameters(),
  lr=0.01,  # or adjust based on your problem
  max_iter=5000,  # More iterations for better convergence
  max_eval=None,  # Default
  tolerance_grad=1e-7,  # Increase sensitivity to gradients
  tolerance_change=1e-9,  # Keep default unless facing early stops
  history_size=100,  # Use larger history for better approximations
  line_search_fn="strong_wolfe")  # Use strong Wolfe line search for better convergence

opt_u_lbfgs=torch.optim.LBFGS(PINN_u.parameters(),
  lr=0.01,  # or adjust based on your problem
  max_iter=5000,  # More iterations for better convergence
  max_eval=None,  # Default
  tolerance_grad=1e-7,  # Increase sensitivity to gradients
  tolerance_change=1e-9,  # Keep default unless facing early stops
  history_size=100,  # Use larger history for better approximations
  line_search_fn="strong_wolfe")  # Use strong Wolfe line search for better convergence

opt_v_lbfgs=torch.optim.LBFGS(PINN_v.parameters(),
  lr=0.01,  # or adjust based on your problem
  max_iter=5000,  # More iterations for better convergence
  max_eval=None,  # Default
  tolerance_grad=1e-7,  # Increase sensitivity to gradients
  tolerance_change=1e-9,  # Keep default unless facing early stops
  history_size=100,  # Use larger history for better approximations
  line_search_fn="strong_wolfe")  # Use strong Wolfe line search for better convergence

opt_p_lbfgs=torch.optim.LBFGS(PINN_p.parameters(),
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
def train_pinn(PINN_u , PINN_v , PINN_p , PINN_T, filename_data,
    filename_bc  ,epochs_adam , epoch_lbgfs , num_collocation_points  ):
    print("Train starting . . . ")
    # Simulate a long process
    for i in tqdm(range(1)):
        time.sleep(0.00005)  # Simulating work by sleeping

    tic = time.time()

    #Stage4-0: Collocation points definition

    x_min = 0
    y_min = 0

    x_max = 1
    y_max = 0.1

    # Define cube boundaries within the domain (example values)
    cube_x_min, cube_x_max = 4.5, 5.5  # x bounds of cube
    cube_y_min, cube_y_max = 0, 1  # y bounds of cube

    # Generate random collocation points within the domain
    np.random.seed(50)

    collocation_points = np.random.rand(num_collocation_points, 2)
    collocation_points[:, 0] = collocation_points[:, 0] * (x_max - x_min) + x_min  # Scale to x bounds
    collocation_points[:, 1] = collocation_points[:, 1] * (y_max - y_min) + y_min  # Scale to y bounds


    X_c = torch.tensor(collocation_points[:, 0], dtype=torch.float32 ,  requires_grad=True).reshape(-1 , 1)
    Y_c = torch.tensor(collocation_points[:, 1], dtype=torch.float32 ,  requires_grad=True).reshape(-1 , 1)

    #TEMPORARY : using noisy collocation points

    X_c_noisy = add_gaussian_noise(X_c).reshape(-1 , 1)
    Y_c_noisy = add_gaussian_noise(Y_c).reshape(-1 , 1)

    #stage4-1: import interior data & boundary data
    interior_data = normal_inputs(pd.read_csv(filename_data))
    boundary_data = normal_inputs(pd.read_csv(filename_bc))


    x_interior = torch.tensor(interior_data[['x']].values, dtype=torch.float32 , requires_grad= True )
    y_interior = torch.tensor(interior_data[['y']].values, dtype=torch.float32 , requires_grad= True )
    output_interior = torch.tensor(interior_data[['u' ,'v' ,'T' , 'p' ]].values, requires_grad= True ,dtype=torch.float32 )

    x_boundary = torch.tensor(boundary_data[['x']].values, dtype=torch.float32 , requires_grad= True )
    y_boundary = torch.tensor(boundary_data[['y']].values, dtype=torch.float32 , requires_grad= True )
    Explicit_Values = torch.tensor(boundary_data[['u' ,'v' ,'T' , 'p' ]].values, requires_grad= True ,dtype=torch.float32 )


    #Stage4-2 : Hyperparameter tuning with Optuna
    def objective(trial):

        # Hyperparameters for tuning
        lambda_pde = trial.suggest_float("lambda_pde", 1, 10)
        lambda_interior = trial.suggest_float("lambda_interior", 0 ,10)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        w_noise = trial.suggest_float("w_noise", 0 ,10)


        opt_u_adam = optim.Adam(PINN_u.parameters() , lr = learning_rate)
        opt_v_adam = optim.Adam(PINN_v.parameters() , lr = learning_rate)
        opt_p_adam = optim.Adam(PINN_p.parameters() , lr = learning_rate)
        opt_T_adam = optim.Adam(PINN_T.parameters() , lr = learning_rate)

        num_epochs_trial = 500  #best value is 500
        for epoch in range(num_epochs_trial):

            opt_u_adam.zero_grad()
            opt_u_adam.zero_grad()
            opt_p_adam.zero_grad()
            opt_T_adam.zero_grad()

            pde_loss = residula_loss(PINN_u , PINN_v , PINN_p , PINN_T , X_c , Y_c )
            interior_loss = data_loss(PINN_u , PINN_v , PINN_p , PINN_T , x_interior , y_interior , output_interior)
            noisy_loss = noisy_data_loss(PINN_u , PINN_v , PINN_p , PINN_T , X_c , Y_c )
            loss = (pde_loss * lambda_pde)  + (interior_loss * lambda_interior) + (noisy_loss * w_noise)

            loss.backward(retain_graph  = True)

            opt_u_adam.step()
            opt_v_adam.step()
            opt_p_adam.step()
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
    print("Optimized Learning Rate:", best_params["learning_rate"])
    print("Optimized weight noisy data:", best_params["w_noise"])


    lambda_pde = best_params["lambda_pde"]
    lambda_interior = best_params["lambda_interior"]
    learning_rate = best_params["learning_rate"]
    w_noise = best_params["w_noise"]

    #stage4-3: Cross Validation

    #stage3-0: Adam

    opt_u_adam = optim.Adam(PINN_u.parameters() , lr = learning_rate)
    opt_v_adam = optim.Adam(PINN_v.parameters() , lr = learning_rate)
    opt_p_adam = optim.Adam(PINN_p.parameters() , lr = learning_rate)
    opt_T_adam = optim.Adam(PINN_T.parameters() , lr = learning_rate)

    #stage3-1: Adam Reduce LR on PLateaue
    scheduler_u = ReduceLROnPlateau(opt_u_adam , factor = 0.5 , min_lr = 0.01 * learning_rate , verbose=False )
    scheduler_v = ReduceLROnPlateau(opt_v_adam , factor = 0.5 , min_lr = 0.01 * learning_rate , verbose=False )
    scheduler_p = ReduceLROnPlateau(opt_p_adam , factor = 0.5 , min_lr = 0.01 * learning_rate , verbose=False )
    scheduler_T = ReduceLROnPlateau(opt_T_adam , factor = 0.5 , min_lr = 0.01 * learning_rate , verbose=False )


    loss_history = []
    loss_val_history = []

    for epoch in range(epochs_adam):

        PINN_T.train()
        PINN_u.train()
        PINN_v.train()
        PINN_p.train()

        opt_u_adam.zero_grad()
        opt_v_adam.zero_grad()
        opt_p_adam.zero_grad()
        opt_T_adam.zero_grad()

        pde_loss = residula_loss(PINN_u , PINN_v , PINN_p , PINN_T , X_c , Y_c )
        interior_loss = data_loss(PINN_u , PINN_v , PINN_p , PINN_T , x_interior , y_interior , output_interior)
        noisy_loss = noisy_data_loss(PINN_u , PINN_v , PINN_p , PINN_T , X_c , Y_c )



        impose_boundary_conditions(PINN_u , PINN_v , PINN_T , PINN_p, x_boundary , y_boundary , Explicit_Values)

        loss = (pde_loss * lambda_pde) +  (interior_loss * lambda_interior) + (noisy_loss * w_noise)

        # Backpropagation
        loss.backward(retain_graph  = True)


        opt_u_adam.step()
        opt_v_adam.step()
        opt_p_adam.step()
        opt_T_adam.step()


        #scheduler_u.step(loss)
        #scheduler_v.step(loss)
        #scheduler_p.step(loss)
        #scheduler_T.step(loss)

        loss_history.append(loss.item())
        #if epoch % 500 == 0:
            #print(f'Epoch Adam {epoch}/{epochs_adam} [{100 * epoch/epochs_adam :.2f}%]  ,Total Loss: {loss.item():.6f}  ')
            #print(f"Loss PDE: {pde_loss.item():.4f}  |   loss Data: {interior_loss.item():.4f}  |   BC loss: {bc_loss.item():.4f}")
            #print(f"======================================================================")


        loss_history.append(loss.item())

        current_LR = opt_u_adam.param_groups[0]['lr']
        # Print losses
        if epoch % 500 == 0:
            print(f"Epoch [{epoch+1}/{epochs_adam}] || Train Loss: {loss.item():.3e} || LearningRate U :{current_LR:.3e} ")
            print(f"Loss PDE: {pde_loss.item():.2e}  |   loss Data: {interior_loss.item():.2e}  |   loss Noisy: {noisy_loss.item():.2e}")
            print(f"======================================================================")
        if epoch % 5000 == 0 :
            plot_results(PINN_u, PINN_v , PINN_p ,  PINN_T, filename_data , epoch = epoch)



    plt.plot(loss_history , label = "LOSS train")
    plt.plot(loss_val_history , label = "LOSS val")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss History ")
    plt.legend()
    #plt.ylim(0 , 1)
    plt.show()


    toc_batch = time.time()
    duration_time = toc_batch - tic
    print(f"Batch_Duration :{duration_time :.2f} s")



    toc = time.time()
    elapseTime = toc - tic
    print ("elapse time in parallel = ", str(round(elapseTime , 4)) + " s")


filename_data = r'Data_PINN_Cavity_hetaFlux.csv'
filename_bc =  r'BC_PINN_Cavity_hetaFlux.csv'
f_test = r'Test_PINN_Cavity_hetaFlux.csv'

epochs_adam = 50000
epoch_lbgfs = 1
num_collocation_points = 1500


train_pinn(PINN_u , PINN_v , PINN_p , PINN_T, filename_data,
    filename_bc  ,epochs_adam , epoch_lbgfs , num_collocation_points  )

"""
# Stage:
#save model

torch.save(PINN_u , "pinn_model_u_full.pth")
torch.save(PINN_v , " pinn_model_v_full.pth")
torch.save(PINN_p , " pinn_model_p_full.pth")
torch.save(PINN_T , "pinn_model_T_full.pth")
"""

plot_results(PINN_u, PINN_v , PINN_p ,  PINN_T, f_test , epoch = epochs_adam  )

#plot_results(PINN_psi, PINN_omega , PINN_T, filename_data)
