# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 22:43:04 2024

@author: Amirreza
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load the data


data = pd.read_csv('Data_PINN_Cavity_hetaFlux.csv')
x_data = data['x'].values
y_data = data['y'].values
u_data = data['u'].values
v_data = data['v'].values
p_data = data['p'].values

# Convert to TensorFlow constants
x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
y_data = tf.convert_to_tensor(y_data, dtype=tf.float32)
u_data = tf.convert_to_tensor(u_data, dtype=tf.float32)
v_data = tf.convert_to_tensor(v_data, dtype=tf.float32)
p_data = tf.convert_to_tensor(p_data, dtype=tf.float32)

# Define the PINN model
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(3, activation=None)
        self.dense2 = tf.keras.layers.Dense(10, activation='relu')
        self.dense3 = tf.keras.layers.Dense(20, activation='relu')
        self.dense4 = tf.keras.layers.Dense(20, activation='relu')
        self.dense5 = tf.keras.layers.Dense(20, activation='relu')
        self.dense6 = tf.keras.layers.Dense(20, activation='relu')
        self.dense7 = tf.keras.layers.Dense(10, activation='relu')
        self.out = tf.keras.layers.Dense(3, activation=None)

    def call(self, x, y):
        inputs = tf.stack([x, y], axis=1)
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        out = self.out(x)
        return out

# Initialize the model
pinn = PINN()

# Loss function
def loss_fn(x, y, u, v, p):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, y])
        u_v_p = pinn(x, y)
        u_pred, v_pred, p_pred = u_v_p[:, 0], u_v_p[:, 1], u_v_p[:, 2]

        u_x = tape.gradient(u_pred, x)
        u_y = tape.gradient(u_pred, y)
        v_x = tape.gradient(v_pred, x)
        v_y = tape.gradient(v_pred, y)
        p_x = tape.gradient(p_pred, x)
        p_y = tape.gradient(p_pred, y)

    continuity = u_x + v_y
    momentum_x = u_pred * u_x + v_pred * u_y + p_x
    momentum_y = u_pred * v_x + v_pred * v_y + p_y

    data_loss = tf.reduce_mean(tf.square(u_pred - u)) + tf.reduce_mean(tf.square(v_pred - v)) + tf.reduce_mean(tf.square(p_pred - p))
    physics_loss = tf.reduce_mean(tf.square(continuity)) + tf.reduce_mean(tf.square(momentum_x)) + tf.reduce_mean(tf.square(momentum_y))

    return data_loss + physics_loss

# Training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#optimizer = minimize(loss_fn(x,y,u,v,p),x0 = [0.1,0.2,0.4,0.5],method='L-BFGS-B')

@tf.function
def train_step(x, y, u, v, p):
    with tf.GradientTape() as tape:
        loss = loss_fn(x, y, u, v, p)
    gradients = tape.gradient(loss, pinn.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, pinn.trainable_variables))
    return loss

# Training loop
epochs = 100000
for epoch in range(epochs):
    loss = train_step(x_data, y_data, u_data, v_data, p_data)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss.numpy: {loss}')

# Save the model
#pinn.save('pinn_model')
"""
# Make predictions
predictions = pinn(x_data, y_data)
u_pred, v_pred, p_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2]

# Compare with exact values
error_u = tf.reduce_mean(tf.abs(u_pred - u_data))
error_v = tf.reduce_mean(tf.abs(v_pred - v_data))
error_p = tf.reduce_mean(tf.abs(p_pred - p_data))

print(f'Mean Absolute Error in u: {error_u.numpy()}')
print(f'Mean Absolute Error in v: {error_v.numpy()}')
print(f'Mean Absolute Error in p: {error_p.numpy()}')

# Plot results


plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(x_data, y_data, c=u_pred, cmap='viridis')



plt.colorbar()
plt.title('Predicted u')

plt.subplot(1, 3, 2)
plt.scatter(x_data, y_data, c=v_pred, cmap='viridis')
plt.colorbar()
plt.title('Predicted v')

plt.subplot(1, 3, 3)
plt.scatter(x_data, y_data, c=p_pred, cmap='viridis')
plt.colorbar()
plt.title('Predicted p')

plt.show()
"""

#Test new data
v_data = pd.read_csv('Test_for_Valid_Data_PINN_Cavity_hetaFlux.csv')

x_exact = v_data['x']
y_exact = v_data['y']
u_exact = v_data['u']
v_exact = v_data['v']
p_exact = v_data['p']

# Make predictions
predictions = pinn(x_exact, y_exact)
u_pred, v_pred, p_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2]

plt.plot(x_exact, v_pred, ls = "--")
plt.plot(x_exact,  v_exact)
plt.show()
