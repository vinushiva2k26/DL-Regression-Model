# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:

### Register Number:

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generate Input Data (X) and Output Data (y)
torch.manual_seed(71)

X = torch.arange(1, 51, dtype=torch.float32).reshape(-1, 1)
noise = torch.randn(X.shape) * 5
y = 2 * X + 1 + noise

# Plot the original data
plt.scatter(X, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()

# Define the Linear Model Class
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):

        return self.linear(x)

# Initialize the Model
torch.manual_seed(59)
model = Model(1, 1)

# Print Initial Weights and Bias
initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()

print("\nName: S Jagandrav vinu siva ")
print("Register No: 212224043001")
print(f'Initial Weight: {initial_weight:.8f}, Initial Bias: {initial_bias:.8f}\n')

# Define Loss Function & Optimizer
loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Train the Model
epochs = 100
losses = []

for epoch in range(1, epochs + 1):

    y_pred = model(X)

    loss = loss_function(y_pred, y)

    losses.append(loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.linear.weight.item():10.8f}  '
          f'bias: {model.linear.bias.item():10.8f}')

# Plot Loss Curve
plt.plot(range(epochs), losses, color='blue')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Curve')
plt.show()

# Final Weights & Bias
final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()

print("\nName: S Jagandrav vinu siva ")
print("Register No: 212224043001")
print(f'\nFinal Weight: {final_weight:.8f}, Final Bias: {final_bias:.8f}')

# Best-Fit Line Calculation
x1 = torch.tensor([X.min().item(), X.max().item()])

y1 = x1 * final_weight + final_bias

# Plot Original Data & Best-Fit Line
plt.scatter(X, y, label="Original Data")
plt.plot(x1, y1, 'r', label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()

# Prediction for x = 120
x_new = torch.tensor([[120.0]])

y_new_pred = model(x_new).item()

print("\nName: S Jagandrav vinu siva ")
print("Register No: 212224043001")
print(f"\nPrediction for x = 120: {y_new_pred:.8f}")

```



# Initialize the Model, Loss Function, and Optimizer


### Dataset Information
Include screenshot of the generated data
![Output](<Screenshot 2026-06-01 133134.png>)
### OUTPUT

![Output](<Screenshot 2026-06-01 132852.png>)
![Output](<Screenshot 2026-06-01 132837.png>)
![Output](<Screenshot 2026-06-01 132759.png>)

### New Sample Data Prediction

![alt text](<Screenshot 2026-06-01 133252.png>)

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
