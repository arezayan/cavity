
"""
Created for solving Stedy-State 2D inconpressible flow in a channel
Momentum and continuty eqyatuin is considered
x, y  : inputs
u,v,p, T : outputs
@author: Amirreza
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize  


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

# Define the PINN
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# Define the network architecture
layers = [2, 40, 40, 40, 40, 40, 40, 40, 4]  # Input: (x, y), Output: (u, v, p, T)
#layers = [2, 20, 20, 20, 4]  # Input: (x, y), Output: (u, v, p)
model = PINN(layers).to(device)

def PDE(model, x, y, mu=0.01,alpha = 1,  lambda_momentum=0.9, lambda_continuity=0.9):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    
    
    uvp = model(torch.cat((x, y), dim=1))
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    T = uvp[:, 2:3]
    p = uvp[:, 3:4]
    
    # Calculate gradients
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True, retain_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True, retain_graph=True)[0]
    
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
    
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)[0]
    T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True, retain_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x), create_graph=True, retain_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, grad_outputs=torch.ones_like(T_y), create_graph=True, retain_graph=True)[0]
    
    # Navier-Stokes equations
    f_u = u * u_x + v * u_y + p_x - mu * (u_xx + u_yy)
    f_v = u * v_x + v * v_y + p_y - mu * (v_xx + v_yy)
    
    # Continuity equation
    continuity = u_x + v_y
    
    # Energy equation
    
    Energy = (u * T_x) + (v * T_y ) - alpha*(T_xx + T_yy)
    
    # Loss calculation with balancing factors
    loss_f = (lambda_momentum * (torch.mean(f_u**2) + torch.mean(f_v**2)) +
              lambda_continuity * torch.mean(continuity**2))
    return loss_f

def data_loss(uvp_pred, u_exact, v_exact, p_exact=None, T_exact=None):
    u_pred = uvp_pred[:, 0:1]
    v_pred = uvp_pred[:, 1:2]
    T_pred = uvp_pred[:, 2:3] if p_exact is not None else None
    p_pred = uvp_pred[:, 3:4]
    
    loss_u = torch.mean((u_pred - u_exact) ** 2)
    loss_v = torch.mean((v_pred - v_exact) ** 2)
    loss_T = torch.mean((p_pred - p_exact) ** 2) if p_exact is not None else 0
    loss_p = torch.mean((T_pred - T_exact) ** 2) if T_exact is not None else 0
    
    return loss_u + loss_v + (loss_p if p_exact is not None else 0) + (loss_T if T_exact is not None else 0)

def total_loss(model, x, y, u_exact, v_exact, p_exact=None, T_exact=None, mu=0.01, alpha = 1,
               lambda_momentum=0.3, lambda_continuity=0.2, lambda_data=0.1):
    uvp_pred = model(torch.cat((x, y), dim=1))
    
    # Physics-informed loss
    loss_f = PDE(model, x, y, mu, lambda_momentum, lambda_continuity)
    
    # Data loss
    loss_data = data_loss(uvp_pred, u_exact, v_exact, p_exact, T_exact) * lambda_data
   
    return loss_f + loss_data

# Training parameters
epochs = 5000
mu = 0.01  # Dynamic viscosity
alpha = 0.002
lambda_momentum = 1.0
lambda_continuity = 1.0
lambda_data = 1.0



# Optimization using Adam optimizer
def train(model, optimizer, x_data, y_data, u_exact, v_exact, p_exact=None,T_exact=None, epochs=10000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = total_loss(model, x_data, y_data, u_exact, v_exact, p_exact,T_exact, mu,alpha,  
                          lambda_momentum, lambda_continuity, lambda_data)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')

# Load data from CSV
data = pd.read_csv('Data_PINN_Cavity_hetaFlux.csv')
x_data = torch.tensor(data[['x']].values, dtype=torch.float32).to(device)
y_data = torch.tensor(data[['y']].values, dtype=torch.float32).to(device)
u_exact = torch.tensor(data[['u']].values, dtype=torch.float32).to(device)
v_exact = torch.tensor(data[['v']].values, dtype=torch.float32).to(device)
p_exact = torch.tensor(data[['p']].values, dtype=torch.float32).to(device) if 'p' in data.columns else None
T_exact = torch.tensor(data[['T']].values, dtype=torch.float32).to(device) if 'T' in data.columns else None

x_data = torch.tensor(normalize(x_data, axis=0),dtype = torch.float32)
y_data = torch.tensor(normalize(y_data,axis=0),dtype = torch.float32)
u_exact = torch.tensor(normalize(u_exact, axis=0),dtype = torch.float32)
v_exact = torch.tensor(normalize(p_exact, axis=0),dtype = torch.float32)
p_exact = torch.tensor(normalize(p_exact, axis=0),dtype = torch.float32)
T_exact = torch.tensor(normalize(T_exact, axis=0),dtype = torch.float32)

"""
optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, history_size=10)
# Train the model
train(model, optimizer, x_data, y_data)


"""
# Define the model and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
#optim.Adamax(model.parameters(), lr=1e-2)

# Train the model
train(model, optimizer, x_data, y_data, u_exact, v_exact, p_exact, T_exact)

# Plotting predicted vs exact values
uvp_pred = model(torch.cat((x_data, y_data), dim=1)).detach().cpu().numpy()
u_pred = uvp_pred[:, 0]
v_pred = uvp_pred[:, 1]
T_pred = uvp_pred[:, 2] if p_exact is not None else None
p_pred = uvp_pred[:, 3] if T_exact is not None else None

u_exact = u_exact.cpu().numpy()
v_exact = v_exact.cpu().numpy()
p_exact = p_exact.cpu().numpy() if p_exact is not None else None
T_exact = T_exact.cpu().numpy() if T_exact is not None else None




plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.scatter(u_exact, u_pred, label='u')
plt.xlabel('Exact u')
plt.ylabel('Predicted u')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(v_exact, v_pred, label='v')
plt.xlabel('Exact v')
plt.ylabel('Predicted v')
plt.legend()

if p_exact is not None:
    plt.subplot(1, 3, 3)
    plt.scatter(p_exact, p_pred, label='Temperature')
    plt.xlabel('Exact T')
    plt.ylabel('Predicted T')
    plt.legend()

plt.figure(dpi = 150)
plt.plot(T_exact, label = "exact Temperature")
plt.plot(T_pred ,  label = "Predicted Temperature",)
plt.legend()
plt.tight_layout()
plt.show()