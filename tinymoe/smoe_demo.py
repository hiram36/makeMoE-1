

from modeling_mixtral import *
from mixtral_config import MixtralConfig


mixstral_config = MixtralConfig(
    vocab_size=1200,
    hidden_size=2048,
    intermediate_size=14336//2
)

def main():

    model = MixtralSparseMoeBlock(config=mixstral_config)

    batch_size, seq_len, hidden_dim = 10, 50, 2048

    hidden_status = torch.randn(size=(batch_size, seq_len, hidden_dim))

    final_hidden_status, roter_logits = model(hidden_status)

    print(f'final_hidden_status.shape===> {final_hidden_status.shape}')
    print(f'final_hidden_status===> {final_hidden_status}')
    print(f'roter_logits.shape===> {roter_logits.shape}')
    print(f'roter_logits===> {roter_logits}')

if __name__ == '__main__':
    main()