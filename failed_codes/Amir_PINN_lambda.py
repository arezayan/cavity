# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 00:24:26 2024

@author: Amirreza
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# Define the network architecture
layers = [2, 200, 200, 200, 200, 200, 200, 200, 3]  # Input: (x, y), Output: (u, v, p)
model = PINN(layers).to(device)


def navier_stokes_loss(model, x, y, mu, lambda_momentum=1.0, lambda_continuity=1.0):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    
    uvp = model(torch.cat((x, y), dim=1))
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]
    
    # Calculate gradients
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    
    # Navier-Stokes equations
    f_u = u*u_x + v*u_y + p_x - mu * (u_xx + u_yy)
    f_v = u*v_x + v*v_y + p_y - mu * (v_xx + v_yy)
    
    # Continuity equation
    continuity = u_x + v_y
    
    # Loss calculation with balancing factors
    loss_f = (lambda_momentum * (torch.mean(f_u**2) + torch.mean(f_v**2)) +
              lambda_continuity * torch.mean(continuity**2))
    return loss_f

def data_loss(uvp_pred, u_exact, v_exact, p_exact=None):
    u_pred = uvp_pred[:, 0:1]
    v_pred = uvp_pred[:, 1:2]
    p_pred = uvp_pred[:, 2:3] if p_exact is not None else None
    
    loss_u = torch.mean((u_pred - u_exact) ** 2)
    loss_v = torch.mean((v_pred - v_exact) ** 2)
    loss_p = torch.mean((p_pred - p_exact) ** 2) if p_exact is not None else 0
    
    return loss_u + loss_v + (loss_p if p_exact is not None else 0)    

def total_loss(model, x, y, u_exact, v_exact, mu, lambda_momentum=1.0, lambda_continuity=1.0, lambda_data=1.0, p_exact=None):
    uvp_pred = model(torch.cat((x, y), dim=1))
    
    # Physics-informed loss
    loss_f = navier_stokes_loss(model, x, y, mu, lambda_momentum, lambda_continuity)
    
    # Data loss
    loss_data = data_loss(uvp_pred, u_exact, v_exact, p_exact) * lambda_data
    
    return loss_f + loss_data




# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#optimizer = optim.LBFGS(model.parameters(), lr=0.001)


# Load data from CSV
data = pd.read_csv('Data_PINN_Cavity_hetaFlux.csv')
x_train = torch.tensor(data[['x']].values, dtype=torch.float32).to(device)
y_train = torch.tensor(data[['y']].values, dtype=torch.float32).to(device)
u_exact = torch.tensor(data[['u']].values, dtype=torch.float32).to(device)
v_exact = torch.tensor(data[['v']].values, dtype=torch.float32).to(device)
p_exact = torch.tensor(data[['p']].values, dtype=torch.float32).to(device) if 'p' in data.columns else None

# Training parameters
epochs = 25000
mu = 0.01  # Dynamic viscosity
lambda_momentum = 1.0
lambda_continuity = 1.0
lambda_data = 1.0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    loss = total_loss(model, x_train, y_train, u_exact, v_exact, mu, lambda_momentum, lambda_continuity, lambda_data, p_exact)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Total Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    uvp_pred = model(torch.cat((x_train, y_train), dim=1))
    u_pred = uvp_pred[:, 0:1]
    v_pred = uvp_pred[:, 1:2]
    p_pred = uvp_pred[:, 2:3]

# Convert tensors to numpy arrays
x_train_np = x_train.detach().numpy()
y_train_np = y_train.detach().numpy()
u_exact_np = u_exact.detach().numpy()
v_exact_np = v_exact.detach().numpy()
u_pred_np = u_pred.detach().numpy()
v_pred_np = v_pred.detach().numpy()
p_pred_np = p_pred.detach().numpy()

plt.figure(dpi = 100)
plt.plot(u_exact_np, label='Exact u', marker='o')
plt.plot(u_pred_np, label='Predicted u', marker='x')
plt.legend()
plt.title('Comparison of u component')


"""
# Plotting the results
plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.scatter(x_train_np, y_train_np, c=u_exact_np, label='Exact u', marker='o')
plt.scatter(x_train_np, y_train_np, c=u_pred_np, label='Predicted u', marker='x')
plt.legend()
plt.colorbar()
plt.title('Comparison of u component')

plt.subplot(1, 3, 2)
plt.scatter(x_train_np, y_train_np, c=v_exact_np, label='Exact v', marker='o')
plt.scatter(x_train_np, y_train_np, c=v_pred_np, label='Predicted v', marker='x')
plt.legend()
plt.colorbar()
plt.title('Comparison of v component')

plt.subplot(1, 3, 3)
plt.scatter(x_train_np, y_train_np, c=p_pred_np, label='Predicted p', marker='x')
plt.legend()
plt.colorbar()
plt.title('Predicted Pressure')
"""


plt.show()
