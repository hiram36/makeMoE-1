
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


# arguments
input_size = 1000
num_classes = 20
num_experts = 10
hidden_size = 64
batch_size = 5
topk = 4


# instantiate the MoE layer
model = sMoE(input_size, num_classes, num_experts, hidden_size, topk=topk, noisy_gating=True)

def demo1():
    # 6 experts
    experts_seq1 = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.01, 0.02]) 
    experts_seq2 = torch.tensor([0.5, 0.4, 0.3, 0.6, 0.055, 0.44]) 

    print('expert seq1 CV:',model.cv_squared(experts_seq1)) # no balancing
    print('expert seq1 CV^2:',model.cv_squared(experts_seq1)**2) # no balancing

    print('expert seq2 CV:',model.cv_squared(experts_seq2)) # balancing
    print('expert seq2 CV^2:',model.cv_squared(experts_seq2)**2) # balancing


def demo2():
    gate1 = torch.tensor([[0.0, 0.9, 0.1, 0.0],
                        [0.0, 0.7, 0.3, 0.0],
                        [0.0, 0.8, 0.2, 0.0],
                        [0.0, 0.95, 0.05, 0.0]])


    gate2 = torch.tensor([[0.9, 0.0, 0.1, 0.0],
                        [0.0, 0.7, 0.0, 0.3],
                        [0.0, 0.2, 0.8, 0.0],
                        [0.1, 0.0, 0.0, 0.9]])


    gate1_sum = gate1.sum(0)
    importance1 = gate1_sum
    print(importance1)                        #tensor([0.0000, 3.3500, 0.6500, 0.0000])
    print(model.cv_squared(importance1))            #tensor(2.5483)

    gate2_sum = gate2.sum(0)
    importance2 = gate2_sum
    print(importance2)                        #tensor([1.0000, 0.9000, 0.9000, 1.2000])
    print(model.cv_squared(importance2))            #tensor(0.0200)

    print(model._gates_to_load(gates=gate1))
    print(model._gates_to_load(gates=gate2))



if __name__ == '__main__':
    demo1()
    demo2()