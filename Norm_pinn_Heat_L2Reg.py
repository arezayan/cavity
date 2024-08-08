# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:21:45 2024

@author: Amirreza
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
model = PINN(layers).to(device)

def PDE(model, x, y, mu=0.01, alpha=0.002, lambda_momentum=0.1, lambda_continuity=0.1):
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

def total_loss(model, x, y, u_exact, v_exact, p_exact=None, T_exact=None, mu=0.01, alpha=0.002,
               lambda_momentum=0.1, lambda_continuity=0.1, lambda_data=1, l2_lambda=0.001):
    uvp_pred = model(torch.cat((x, y), dim=1))

    # Physics-informed loss
    loss_f = PDE(model, x, y, mu, lambda_momentum, lambda_continuity)

    # Data loss
    loss_data = data_loss(uvp_pred, u_exact, v_exact, p_exact, T_exact) * lambda_data

    # L2 regularization
    l2_reg = sum(param.pow(2.0).sum() for param in model.parameters())
    loss = loss_f + loss_data + l2_lambda * l2_reg

    return loss

# Training parameters
mu = 0.01  # Dynamic viscosity
alpha = 0.002
lambda_momentum = 0.1
lambda_continuity = 0.1
lambda_data = 1.0
l2_lambda = 0.001  # L2 regularization weight
epochs = 5000

# Optimization using Adam optimizer with early stopping
def train(model, optimizer, x_data, y_data, u_exact, v_exact, p_exact=None, T_exact=None, patience=500):
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = total_loss(model, x_data, y_data, u_exact, v_exact, p_exact, T_exact, mu, alpha, 
                          lambda_momentum, lambda_continuity, lambda_data, l2_lambda)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            print(f'Early stopping at epoch {epoch}')
            break

        if epoch % 500 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')

# Load data from CSV
data = pd.read_csv('sparce_data.csv')
x_data = torch.tensor(data[['x']].values, dtype=torch.float32).to(device)
y_data = torch.tensor(data[['y']].values, dtype=torch.float32).to(device)
u_exact = torch.tensor(data[['u']].values, dtype=torch.float32).to(device)
v_exact = torch.tensor(data[['v']].values, dtype=torch.float32).to(device)
p_exact = torch.tensor(data[['p']].values, dtype=torch.float32).to(device) if 'p' in data.columns else None
T_exact = torch.tensor(data[['T']].values, dtype=torch.float32).to(device) if 'T' in data.columns else None

# Normalize data
x_data = torch.tensor(normalize(x_data, axis=0), dtype=torch.float32).to(device)
y_data = torch.tensor(normalize(y_data, axis=0), dtype=torch.float32).to(device)
u_exact = torch.tensor(normalize(u_exact, axis=0), dtype=torch.float32).to(device)
v_exact = torch.tensor(normalize(v_exact, axis=0), dtype=torch.float32).to(device)
p_exact = torch.tensor(normalize(p_exact, axis=0), dtype=torch.float32).to(device)
T_exact = torch.tensor(normalize(T_exact, axis=0), dtype=torch.float32).to(device)

# Define the model and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

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

plt.figure(dpi=150)
plt.plot(T_exact, label="Exact Temperature")
plt.plot(T_pred, label="Predicted Temperature")
plt.legend()
plt.tight_layout()
plt.show()



# Load and preprocess new data
new_data = pd.read_csv('test_sparce_data.csv')  # Replace 'new_data.csv' with your new data file
x_new = torch.tensor(new_data[['x']].values, dtype=torch.float32).to(device)
y_new = torch.tensor(new_data[['y']].values, dtype=torch.float32).to(device)
u_exact_new = torch.tensor(new_data[['u']].values, dtype=torch.float32).to(device)
v_exact_new = torch.tensor(new_data[['v']].values, dtype=torch.float32).to(device)
p_exact_new = torch.tensor(new_data[['p']].values, dtype=torch.float32).to(device) if 'p' in new_data.columns else None
T_exact_new = torch.tensor(new_data[['T']].values, dtype=torch.float32).to(device) if 'T' in new_data.columns else None

# Normalize new data
x_new = torch.tensor(normalize(x_new, axis=0), dtype=torch.float32).to(device)
y_new = torch.tensor(normalize(y_new, axis=0), dtype=torch.float32).to(device)
u_exact_new = torch.tensor(normalize(u_exact_new, axis=0), dtype=torch.float32).to(device)
v_exact_new = torch.tensor(normalize(v_exact_new, axis=0), dtype=torch.float32).to(device)
p_exact_new = torch.tensor(normalize(p_exact_new, axis=0), dtype=torch.float32).to(device)
T_exact_new = torch.tensor(normalize(T_exact_new, axis=0), dtype=torch.float32).to(device)

# Make predictions with the trained model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    uvp_pred_new = model(torch.cat((x_new, y_new), dim=1)).cpu().numpy()
    u_pred_new = uvp_pred_new[:, 0]
    v_pred_new = uvp_pred_new[:, 1]
    T_pred_new = uvp_pred_new[:, 2] if p_exact_new is not None else None
    p_pred_new = uvp_pred_new[:, 3] if T_exact_new is not None else None

# Convert exact values to numpy arrays
u_exact_new = u_exact_new.cpu().numpy()
v_exact_new = v_exact_new.cpu().numpy()
p_exact_new = p_exact_new.cpu().numpy() if p_exact_new is not None else None
T_exact_new = T_exact_new.cpu().numpy() if T_exact_new is not None else None

# Calculate error metrics
u_mse = mean_squared_error(u_exact_new, u_pred_new)
u_mae = mean_absolute_error(u_exact_new, u_pred_new)
u_r2 = r2_score(u_exact_new, u_pred_new)

v_mse = mean_squared_error(v_exact_new, v_pred_new)
v_mae = mean_absolute_error(v_exact_new, v_pred_new)
v_r2 = r2_score(v_exact_new, v_pred_new)

if p_exact_new is not None:
    T_mse = mean_squared_error(T_exact_new, T_pred_new)
    T_mae = mean_absolute_error(T_exact_new, T_pred_new)
    T_r2 = r2_score(T_exact_new, T_pred_new)

if T_exact_new is not None:
    p_mse = mean_squared_error(p_exact_new, p_pred_new)
    p_mae = mean_absolute_error(p_exact_new, p_pred_new)
    p_r2 = r2_score(p_exact_new, p_pred_new)

print(f'U - MSE: {u_mse}, MAE: {u_mae}, R²: {u_r2}')
print(f'V - MSE: {v_mse}, MAE: {v_mae}, R²: {v_r2}')
if p_exact_new is not None:
    print(f'T - MSE: {T_mse}, MAE: {T_mae}, R²: {T_r2}')
if T_exact_new is not None:
    print(f'P - MSE: {p_mse}, MAE: {p_mae}, R²: {p_r2}')

# Plotting predicted vs exact values
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.scatter(u_exact_new, u_pred_new, label='u')
plt.xlabel('Exact u')
plt.ylabel('Predicted u')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(v_exact_new, v_pred_new, label='v')
plt.xlabel('Exact v')
plt.ylabel('Predicted v')
plt.legend()

if p_exact_new is not None:
    plt.subplot(1, 3, 3)
    plt.scatter(p_exact_new, p_pred_new, label='Temperature')
    plt.xlabel('Exact T')
    plt.ylabel('Predicted T')
    plt.legend()

plt.figure(dpi=150)
plt.plot(T_exact_new, label="Exact Temperature")
plt.plot(T_pred_new, label="Predicted Temperature")
plt.legend()
plt.tight_layout()
plt.show()