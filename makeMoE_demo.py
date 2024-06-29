
import torch
import torch.nn as nn

from makeMoE import SparseMoELanguageModel, kaiming_init_weights, get_batch

device = torch.device('mps')

def demo():
    model = SparseMoELanguageModel()
    model.apply(kaiming_init_weights)
    model = model.to(device)
    # sample a batch of data
    xb, yb = get_batch('train')
    x = xb[0, :].unsqueeze(0)
    y = yb[0, :].unsqueeze(0)
    x.to(device)
    y.to(device)

    logits, loss = model(x, y)
    
    print(f'logits.shape===> {logits.shape}')
    print(f'logits===> {logits}')
    print(f'loss.shape===> {loss.shape}')
    print(f'loss===> {loss}')


if __name__ == "__main__":
    demo()