
# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#

import torch
from torch import nn
from torch.optim import Adam

from smoe import sMoE



def train(x, y, model, loss_fn, optim):
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    y_hat, aux_loss = model(x.float())
    # calculate prediction loss
    loss = loss_fn(y_hat, y)
    # combine losses
    total_loss = loss + aux_loss
    optim.zero_grad()
    total_loss.backward()
    optim.step()
    print("  Training Results - loss: {:.4f}, aux_loss: {:.4f}".format(loss.item(), aux_loss.item()))
    return model

def eval(x, y, model, loss_fn):
    model.eval()
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    y_hat, aux_loss = model(x.float())
    moe_loss = loss_fn(y_hat, y) # MoE Predict and Label
    # smoe loss
    total_loss = moe_loss + aux_loss
    print("Evaluation Results - moe_loss: {:.4f}, aux_loss: {:.4f}, total_loss: {:.4f}".format(moe_loss.item(), aux_loss.item(), total_loss.item()))

def dummy_data(batch_size, input_size, num_classes):
    # dummy input
    x = torch.rand(batch_size, input_size)
    # dummy target
    y = torch.randint(num_classes, (batch_size, 1)).squeeze(1)
    return x, y

# arguments
input_size = 1000
num_classes = 20
num_experts = 8
hidden_size = 64
batch_size = 5
topk = 2

# determine device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# instantiate the MoE layer
model = sMoE(input_size, num_classes, num_experts, hidden_size, topk=topk, noisy_gating=True)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optim = Adam(model.parameters())

x, y = dummy_data(batch_size, input_size, num_classes)

# train
model = train(x.to(device), y.to(device), model, loss_fn, optim)
# evaluate
x, y = dummy_data(batch_size, input_size, num_classes)
eval(x.to(device), y.to(device), model, loss_fn)
