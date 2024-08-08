# -*- coding: utf-8 -*-
"""
This code is initialized with 2D steady state Laminar flow
@author: Amirreza
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Neural Network definition
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 3)  # Outputs: u, v, p

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

# Physics-informed loss function
def physics_informed_loss(model, x, y, u, v, p, nu=0.01, verbose=True):
    # Forward pass
    outputs = model(torch.cat([x, y], dim=1))
    u, v, p = outputs[:, 0], outputs[:, 1], outputs[:, 2]
    
    # Compute residuals for the continuity equation
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    continuity_residual = u_x + v_y

    # Compute residuals for the x-momentum equation
    #u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    #u_residual = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy) + p_x
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    u_residual = u * u_x + v * u_y - nu * (u_xx + u_yy) + p_x

    # Compute residuals for the y-momentum equation
    #v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    #v_residual = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy) + p_y
    v_residual = u * v_x + v * v_y - nu * (v_xx + v_yy) + p_y

    # Compute the total loss as the sum of residuals
    loss = torch.mean(continuity_residual**2) + torch.mean(u_residual**2) + torch.mean(v_residual**2)
    return loss

# Training loop
def train(model, optimizer, epochs=25000):
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            loss = physics_informed_loss(model, x, y, u, v, p)
            loss.backward()
            optimizer.step(closure)
            return loss
            optimizer.step(closure=closure)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            plt.scatter(epoch,loss.detach().numpy())
            plt.show()



# Sample data points
N = 1000
data = pd.read_csv("Data_PINN_Cavity_hetaFlux.csv")

# Sample data points


x = (torch.tensor(data["x"], dtype=torch.float32,requires_grad=True)).reshape(-1,1)
y = (torch.tensor(data["y"], dtype=torch.float32, requires_grad=True)).reshape(-1,1)
u = (torch.tensor(data["u"], dtype=torch.float32)).reshape(-1,1)
v = (torch.tensor(data["v"], dtype=torch.float32)).reshape(-1,1)
p = (torch.tensor(data["p"], dtype=torch.float32)).reshape(-1,1)
#t = torch.rand((N, 1), requires_grad=True)

# Initialize model and optimizer
model = PINN()
#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.LBFGS(model.parameters(), lr=0.001)


# Train the model
train(model, optimizer)


plt.figure(dpi=150)
plt.grid(True, which="both", ls=":")
plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
plt.xscale("log")
plt.legend()


# Predicting on new data points
data_test = pd.read_csv("Test_for_Valid_Data_PINN_Cavity_hetaFlux.csv")
x_test = (torch.tensor(data_test["x"], dtype=torch.float32)).reshape(-1,1)
y_test = (torch.tensor(data_test["y"], dtype=torch.float32)).reshape(-1,1)
u_exact = (torch.tensor(data_test["u"], dtype=torch.float32)).reshape(-1,1)

# Assuming x_test and y_test are within the domain used during training
test_inputs = torch.cat([x_test, y_test], dim=1)
predicted = model(test_inputs)
u_pred = predicted[:, 0].detach().numpy()
v_pred = predicted[:, 1].detach().numpy()
p_pred = predicted[:, 2].detach().numpy()


# Compare and plot u component
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_test.detach().numpy(), y_test.detach().numpy(), c=u_exact.detach().numpy(), cmap='viridis', label='Exact u')
plt.colorbar()
plt.title('Exact u')

plt.subplot(1, 2, 2)
plt.scatter(x_test.detach().numpy(), y_test.detach().numpy(), c=u_pred, cmap='viridis', label='Predicted u')
plt.colorbar()
plt.title('Predicted u')

plt.show()


plt.figure(dpi = 150)
plt.plot(u_exact.detach().numpy(), u_pred)
plt.title("exact  vs pinn predicted solution")
plt.show()
# Similarly, you can plot for v and p components