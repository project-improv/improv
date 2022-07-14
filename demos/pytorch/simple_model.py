import torch
from torch.nn import Linear, Module, ReLU, Softmax
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss as CEL
from torch.optim import SGD

class MLP(Module):
    
    def __init__(self):
        super(MLP, self).__init__()

        self.layer = Linear(n_inputs, 1)
        self.activation = Softmax()

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)

        return x

class SimpleModel(Module):

    def __init__(self):
        super(SimpleModel, self).__init__()

        self.linear1 = Linear(n_inputs, 200)
        self.activation = ReLU()
        self.linear2 = Linear(200, 10)
        self.softmax = Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

def train_model(train_dl, model):
    model.train()
    train_loss = list()
    # Optimizationn
    criterion = CEL()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    # Epochs
    for epoch in range(n_epochs):
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), i * len(inputs)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # train_loss.append(loss.item())
    torch.save(network.state_dict(), '/results/model.pth')
    torch.save(optimizer.state_dict(), 'results/optimizer.pth')

def test_model(test_dl, model):
    model.eval()
    test_loss, correct = 0, 0
    predict, true = list(), list()
    with torch.no_grad()
        for inputs, targets in enumerate(test_dl):
            yhat = model(inputs)
            test_loss += CEL(yhat, targets)
            correct += (yhat.argmax(1) == targets).type(torch.float).sum().item()
    # test_loss /= len()
    # correct /= len(test_dl)
    # real = targets.numpy()
    # real = real.reshape((len(real), 1))
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    torch.save(model.state_dict()), "model.pth
    print("Saved PyTorch Model to model.pth")

