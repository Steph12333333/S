import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt




X=torch.tensor( [[0.2, 1.5, -0.3],  
    [1.0, -1.0, 0.5],   
    [-0.5, 2.0, 1.0]])

y=torch.tensor([[0.0],[1.0],[0.0]])

activations = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'Sigmoid': nn.Sigmoid(),
    
}

for name,activation in activations.items():
    print(f"\nTraining with {name} activation...")
    model=nn.Sequential(
    nn.Linear(3,4),
    activation,
    nn.Linear(4,1),
    nn.Sigmoid()
    )
    param=[]
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        param.append(loss.item())

        if epoch % 100 == 0:
              print(f"Epoch {epoch}: loss = {loss.item():.4f}")
    plt.plot(param, label=name)
    print(f"{name} activation prediction {y_pred}")

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Activation Function Comparison')
plt.legend()

plt.show()





        