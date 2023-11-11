from _context import vugrad

import numpy as np

# for running from the command line
from argparse import ArgumentParser

import vugrad as vg
from vugrad.core import Op

class ReLU(Op):
    """
    Op for element-wise application of ReLU function
    """

    @staticmethod
    def forward(context, input):
        ind = input < 0
        context['ind'] = ind
        input[ind] = 0
        return input

    @staticmethod
    def backward(context, goutput):
        ind = context['ind']
        goutput[ind] = 0
        return goutput

def relu(x):
    return ReLU.do_forward(x)

# Parse command line arguments
parser = ArgumentParser()

parser.add_argument('-D', '--dataset',
                dest='data',
                help='Which dataset to use. [synth, mnist]',
                default='mnist', type=str)

parser.add_argument('-b', '--batch-size',
                dest='batch_size',
                help='The batch size (how many instances to use for a single forward/backward pass).',
                default=128, type=int)

parser.add_argument('-e', '--epochs',
                dest='epochs',
                help='The number of epochs (complete passes over the complete training data).',
                default=20, type=int)

parser.add_argument('-l', '--learning-rate',
                dest='lr',
                help='The learning rate. That is, a scalar that determines the size of the steps taken by the '
                     'gradient descent algorithm. 0.1 works well for synth, 0.0001 works well for MNIST.',
                default=3e-6, type=float)

args = parser.parse_args()

## Load the data
if args.data == 'synth':
    (xtrain, ytrain), (xval, yval), num_classes = vg.load_synth()
elif args.data == 'mnist':
    (xtrain, ytrain), (xval, yval), num_classes = vg.load_mnist(final=False, flatten=True)
else:
    raise Exception(f'Dataset {args.data} not recognized.')

print(f'## loaded data:')
print(f'         number of instances: {xtrain.shape[0]} in training, {xval.shape[0]} in validation')
print(f' training class distribution: {np.bincount(ytrain)}')
print(f'     val. class distribution: {np.bincount(yval)}')

num_instances, num_features = xtrain.shape

# Create a simple neural network.
# This is a `Module` consisting of other modules representing linear layers, provided by the vugrad library.
class MLP(vg.Module):
    """
    A simple MLP with one hidden layer, and a sigmoid non-linearity on the hidden layer and a softmax on the
    output.
    """

    def __init__(self, input_size, output_size, hidden_sizes=[1568, 784, 392]):
        """
        :param input_size:
        :param output_size:
        :param hidden_sizes: the number of neurons in each hidden layer
        """
        super().__init__()

        if len(hidden_sizes) > 0:
            self.initialize_layers(input_size, output_size, hidden_sizes)
        else: self.layers = [vg.Linear(input_size, output_size), vg.logsoftmax]

    def initialize_layers(self, input_size, output_size, hidden_sizes):
        self.layers = [vg.Linear(input_size, hidden_sizes[0]), relu]
        for i in range(len(hidden_sizes[:-1])):
            self.layers.append(vg.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(relu)
        self.layers.extend([vg.Linear(hidden_sizes[-1], output_size), vg.logsoftmax])

    def forward(self, x):
        assert len(x.size()) == 2

        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for l in self.layers if hasattr(l, 'parameters') for p in l.parameters()]

## Instantiate the model
mlp = MLP(input_size=num_features, output_size=num_classes)

n, m = xtrain.shape
b = args.batch_size

gamma = 0.9
momentum = [np.zeros_like(p.value) for p in mlp.parameters()]

print('\n## Starting training')
for epoch in range(args.epochs):

    print(f'epoch {epoch:03}')

    ## Compute validation accuracy
    o = mlp(vg.TensorNode(xval))
    oval = o.value

    predictions = np.argmax(oval, axis=1)
    num_correct = (predictions == yval).sum()
    acc = num_correct / yval.shape[0]

    o.clear() # gc the computation graph
    print(f'       accuracy: {acc:.4}')

    cl = 0.0 # running sum of the training loss

    # We loop over the data in batches of size `b`
    for fr in range(0, n, b):

        # The end index of the batch
        to = min(fr + b, n)

        # Slice out the batch and its corresponding target values
        batch, targets = xtrain[fr:to, :], ytrain[fr:to]

        # Wrap the inputs in a Node
        batch = vg.TensorNode(value=batch)

        outputs = mlp(batch)
        loss = vg.logceloss(outputs, targets)
        # -- The computation graph is now complete. It consists of the MLP, together with the computation of
        #    the scalar loss.
        # -- The variable `loss` is the TensorNode at the very top of our computation graph. This means we can call
        #    it to perform operations on the computation graph, like clearing the gradients, starting the backpropgation
        #    and clearing the graph.
        # -- Note that we set the MLP up to produce log probabilties, so we should compute the CE loss for these.

        cl += loss.value
        # -- We must be careful here to extract the _raw_ value for the running loss. What would happen if we kept
        #    a running sum using the TensorNode?

        # Start the backpropagation
        loss.backward()

        # pply gradient descent
        for i, parm in enumerate(mlp.parameters()):
            momentum[i] = gamma * momentum[i] + parm.grad
            parm.value -= args.lr * momentum[i]
            # -- Note that we are directly manipulating the members of the parm TensorNode. This means that for this
            #    part, we are not building up a computation graph.

        # -- In Pytorch, the gradient descent is abstracted away into an Optimizer. This allows us to build slightly more
        #    complexoptimizers than plain graident descent.

        # Finally, we need to reset the gradients to zero ...
        loss.zero_grad()
        # ... and delete the parts of the computation graph we don't need to remember.
        loss.clear()

    print(f'   running loss: {cl/n:.4}')
