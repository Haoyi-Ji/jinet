from jinet.tensor import Tensor
from jinet.nn import NeuralNet
from jinet.loss import Loss, MSE
from jinet.optimizer import Optimizer, SGD
from jinet.data import DataIterator, BatchIterator


def train(net: NeuralNet, 
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs , targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print('Epoch', epoch, ' loss =', epoch_loss)

