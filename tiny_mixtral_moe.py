
import torch

from torch.distributions.normal import Normal
from torch import nn
from torch.nn import functional as F
from torch.nn import init

# from transformers import MixtralConfig

run_device = torch.device('mps')

batch_size = 16 
block_size = 32
n_embed = 128
dropout = 0.1
n_head = 8
top_k = 2
n_layer = 6
learning_rate = 1e-3
max_iters = 450
eval_interval = 100
eval_iters = 400
train_ratio = 0.9

model_file_path = 'model/makemoe_gatingnetwork_9M_v1.pth'

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(train_ratio * len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(run_device), y.to(run_device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def save_model(model, model_file_path):
    torch.save(model, model_file_path)


class MixtralConfig2():
    ffn_dim = 256
    hidden_dim = 128
    num_experts_per_tok = 2
    num_local_experts = 8
    intermediate_size = 128
    hidden_size = 128
    num_experts = 8


class MixtralBLockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig2):
        super().__init__()
        self.ffn_dim = config.ffn_dim
        self.hidden_dim = config.hidden_dim

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = nn.SiLU()

    # Forward 是 SwiGLU
    def forward(self, x):
        hidden_states = x
        y = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        y = self.w2(y)
        return y

class GatingNetwork(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, noisy_gating=True, noise_epsilon=1e-2):
        super(GatingNetwork, self).__init__()
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.nosie_epsilon = noise_epsilon
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts, bias=False)
        self.noise_linear = nn.Linear(n_embed, num_experts, bias=False)

    def forward(self, x):
        hidden_states = x
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # 每层都会产生router_logits, 将用于最后作 load balance loss
        # hidden_states is the output tensor from multihead self attention block
        router_logits = self.topkroute_linear(hidden_states)
        #print(f'experts.gate output router logits : \n {router_logits}')
        sum_logits = router_logits

        if self.noisy_gating:
            #Noise logits
            noise_logits = self.noise_linear(hidden_states)
            noise_stddev = F.softplus(noise_logits) + self.nosie_epsilon
            #Adding scaled unit gaussian noise to the logits
            noise = torch.randn_like(router_logits) * noise_stddev
            sum_logits = router_logits + noise

        #print(f'sum_logits===> {sum_logits}')
        top_logits, top_indices = torch.topk(sum_logits, min(self.top_k+1, self.num_experts), dim=-1)
        #print(f'top_logits===> {top_logits}')
        #print(f'top_indices===> {top_indices}')
        # 直接赋值？
        routing_weights = top_logits[:, :self.top_k]
        selected_experts = top_indices[:, :self.top_k]
        #print(f'expert select : \n {selected_experts}')
        #print(f'topk : \n {routing_weights}')

        # 计算 TopK 的 专家 logits 和 Top2 专家的位置
        routing_weights = F.softmax(routing_weights, dim=1, dtype=torch.float).cpu().clone()
        #print(f'softmax weight  : {routing_weights.shape}')

        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        #print(f'topk归一化 : \n {routing_weights}')

        routing_weights = routing_weights.to(hidden_states.dtype)

        ## One Hot 编码
        expert_mask = torch.nn.functional.one_hot(selected_experts, \
                                                num_classes=self.num_experts).permute(2, 1, 0)
        
        zeros = torch.zeros_like(sum_logits, requires_grad=True)
        selected_experts = selected_experts.to(run_device)
        routing_weights = routing_weights.to(run_device)
        gates = zeros.scatter(1, selected_experts, routing_weights)
        #for i in range(self.seq_len):
        #    print(f'【token_{i}】\n', expert_mask[:,:,i])
        router_logits = router_logits.to(run_device)
        sum_logits = sum_logits.to(run_device)
        noise_stddev = noise_stddev.to(run_device)
        top_logits = top_logits.to(run_device)
        if self.noisy_gating and self.top_k < self.num_experts:
            load = load_loss_v2(router_logits, sum_logits, noise_stddev, top_logits)
            #load = None
        else:
            load = gates_to_load(gates=gates)

        return routing_weights, selected_experts, expert_mask, load

    def forward_bak(self, x):
        hidden_states = x
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # 每层都会产生router_logits, 将用于最后作 load balance loss
        # hidden_states is the output tensor from multihead self attention block
        router_logits = self.topkroute_linear(hidden_states)
        #print(f'experts.gate output router logits : \n {router_logits}')
        sum_logits = router_logits

        if self.noisy_gating:
            #Noise logits
            noise_logits = self.noise_linear(hidden_states)
            noise_stddev = F.softplus(noise_logits) + self.nosie_epsilon
            #Adding scaled unit gaussian noise to the logits
            noise = torch.randn_like(router_logits) * noise_stddev
            sum_logits = router_logits + noise

        #print(f'sum_logits===> {sum_logits}')
        routing_weights, selected_experts  = torch.topk(sum_logits, self.top_k, dim=-1)
        #print(f'expert select : \n {selected_experts}')
        #print(f'topk : \n {routing_weights}')

        # 计算 TopK 的 专家 logits 和 Top2 专家的位置
        routing_weights = F.softmax(routing_weights, dim=1, dtype=torch.float).cpu().clone()
        #print(f'softmax weight  : \n {routing_weights}')

        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        #print(f'topk归一化 : \n {routing_weights}')

        routing_weights = routing_weights.to(hidden_states.dtype)

        ## One Hot 编码
        expert_mask = torch.nn.functional.one_hot(selected_experts, \
                                                num_classes=self.num_experts).permute(2, 1, 0)
        
        zeros = torch.zeros_like(sum_logits, requires_grad=True)
        gates = zeros.scatter(1, selected_experts, routing_weights)
        #for i in range(self.seq_len):
        #    print(f'【token_{i}】\n', expert_mask[:,:,i])
        if self.noisy_gating and self.top_k < self.num_experts:
            #load = load_loss_v2(routing_weights, sum_logits, noise_stddev, routing_weights)
            load = load_loss(self.num_experts, self.top_k, batch_size, sequence_length)
        else:
            load = gates_to_load(gates=gates)

        return routing_weights, selected_experts, expert_mask, load
    
def gates_to_load(gates):
    """Compute the true load per expert, given the gates.
    The load is the number of examples for which the corresponding gate is >0.
    Args:
    gates: a `Tensor` of shape [batch_size, n]
    Returns:
    a float32 `Tensor` of shape [n]
    """
    return (gates > 0).sum(0)
    
def load_loss_v2(clean_values, noisy_values, noise_stddev, noisy_top_values, topk=2):
        clean_values = clean_values.to(run_device)
        noisy_values = noisy_values.to(run_device)
        noise_stddev = noise_stddev.to(run_device)
        noisy_top_values = noisy_top_values.to(run_device)
        batch_size = clean_values.size(0)
        m = noisy_top_values.size(1)
        #print(f'noisy_top_values.shape===> {noisy_top_values.shape}')
        top_values_flat = noisy_top_values.flatten()
        # top-k时会把无关的expert的gating置为0， 这时要填补一些随机值，使得参数是可导的
        threshold_positions_if_in = torch.arange(batch_size, device=clean_values.device) * m + topk
        #print(f'threshold_positions_if_in===> {threshold_positions_if_in}')
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        threshold_if_in = threshold_if_in.to(run_device)
        # Xi noisy gating > Kth excluding 说明Noisy 无用???
        is_in = torch.gt(noisy_values, threshold_if_in).to(run_device)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1).to(run_device)
        # is each value currently in the top k.
        # 这里计算每个专家的估计负载值
        mean = torch.tensor([0.0]).to(run_device)
        std = torch.tensor([1.0]).to(run_device)
        normal = Normal(mean, std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev).to(run_device)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev).to(run_device)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

def load_loss(num_experts, top_k, batch_size, seq_length):
    #print(f'sMoE num_experts:{num_experts} top_k:{top_k} batch:{batch_size} seq_length:{seq_length}')

    router_logits_1 = torch.randn(batch_size, seq_length, num_experts).view(-1,num_experts) # layer 1
    router_logits_2 = torch.randn(batch_size, seq_length, num_experts).view(-1,num_experts) # layer 2
    router_logits = [router_logits_1, router_logits_2] 

    concatenated_gate_logits = torch.cat(router_logits, dim = 0)
    #print('单层gating的路由logits:', router_logits_1.shape) 
    #print('两层gating的路由logits:', concatenated_gate_logits.shape)

    #print('根据logits top-k 计算one-hot编码')
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1).cpu().clone()
    #print(f'routing_weights.shape ===> {routing_weights.shape}')

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    #print(expert_mask.shape)

    tokens_sum_expert = torch.sum(expert_mask.float(), dim=0)
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    #print(f'top1 每个专家平均处理的token   :', tokens_sum_expert[0])
    #print(f'top2 每个专家平均处理的token fi:', tokens_per_expert[1])
    #print(f'top1与top2水平合计', tokens_per_expert.sum(dim=1))

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    #print('router_prob_per_expert Pi: ' , router_prob_per_expert)
    #print( '每个专家的负载：',  tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    #print('final loss:', overall_loss)
    return overall_loss

class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config: MixtralConfig2):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        #self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.gating_network = GatingNetwork(
            self.hidden_dim, 
            self.num_experts, 
            self.top_k
        )

        # 多个 SwiGLU MLP 层组成混合专家
        self.experts = nn.ModuleList(
            [MixtralBLockSparseTop2MLP(config) for _ in range(self.num_experts)]
        )

    def forward(self, x):
        hidden_states = x
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        #hidden_states = hidden_states.view(-1, hidden_dim)
        #print(f'x.shape===> {hidden_states.shape}')
        #print(f'type of x===> {type(hidden_states)}')
        routing_weights, selected_experts, expert_mask, load_loss = self.gating_network(hidden_states)
        #print(f'routing_weights.shape ===> {routing_weights.shape}')
        #print(f'selected_experts ===> {selected_experts}')
        #print(f'expert_mask ===> {expert_mask}')
        #print(f'load_loss ===> {load_loss}')

        hidden_states = hidden_states.view(-1, hidden_dim)
        ## 最终结果
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), \
                dtype=hidden_states.dtype, device=hidden_states.device
        )
        #print(f'final moe result shape for each token: {final_hidden_states.shape}')

        # 每个专家收集需要计算token
        for expert_idx in range(self.num_experts):
            #print(f'--------expert {expert_idx} ---------')
            expert_layer = self.experts[expert_idx]
            #print(expert_mask[expert_idx])
            idx, top_x = torch.where(expert_mask[expert_idx])
            #print(f'专家 {expert_idx} 计算的样本编号:',top_x.tolist()) # select x_idx for expert top1
            #print(f'专家 {expert_idx} top1:0, top2:1 ',idx.tolist()) # 0 is top1 ,1 is top2
            #print(f'有 {len(top_x)} / {x.shape[1]} token 选到专家 {expert_idx}')
            
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)

            expert_output = expert_layer(current_state)
            routing_output = routing_weights[top_x_list, idx_list, None]
            # expert_0(x) * routing_weights
            current_hidden_states =  expert_output * routing_output 

            # 将计算的单个专家结果填入到结果表里
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            #print(f'current_state.shape===> {current_state.shape}') 
            #print(f'routing_out===> {routing_output.shape}')
            #print(f'current_hidden_states.shape===> {current_hidden_states.shape}')
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, -1)
        #print(f'final_hidden_states.shape===> {final_hidden_states.shape}')
        #print(f'final_hidden_states===> {final_hidden_states}')
        return final_hidden_states

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1).cpu().clone()   # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        wei = wei.to(run_device)
        v = v.to(run_device)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    """ Mixture of Experts Transformer block: communication followed by computation (multi-head self attention + SparseMoE) """

    def __init__(self, n_embed, n_head, num_experts, top_k):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        config = MixtralConfig2()
        self.smoe = MixtralSparseMoeBlock(config=config)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.smoe(self.ln2(x))
        return x


class SparseMoELanguageModel(nn.Module):

    def __init__(self, config: MixtralConfig2):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head, num_experts=config.num_experts,top_k=top_k) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=run_device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1).cpu().clone() # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def one_expert_demo():
    x = torch.randn(1, 64, 128)
    config = MixtralConfig2()
    expert = MixtralBLockSparseTop2MLP(config)
    print('单个专家为原LLaMA的MLP层')
    print(expert)
    g = expert(x)
    print(f'单个专家输入: {x.shape}')
    print(f'单个专家输出结果：{g.shape}')

def mutli_experts_demo():
    x = torch.randn(1, 64, 128)
    config = MixtralConfig2()
    experts = MixtralSparseMoeBlock(config)
    print('多个专家混合专家')
    print(experts)

def sMoe_demo():
    x = torch.randn((1, 64, 128), device=run_device)
    config = MixtralConfig2()
    experts = MixtralSparseMoeBlock(config).to(device=run_device)
    print('sMoE')
    print(experts)
    final_hidden_states = experts(x)
    print(f'final_hidden_states.shape===> {final_hidden_states.shape}')
    print(f'final_hidden_states===> {final_hidden_states}')

def kaiming_init_weights(m):
    if isinstance (m, (nn.Linear)):
        init.kaiming_normal_(m.weight)

def get_model_parameters_numel(model):
    total_params = 0
    for name, param in model.named_parameters():
        each_param_numel = param.numel()
        total_params = total_params + each_param_numel
        print(f'{name} {each_param_numel}')
    return total_params

def run_model():
    config = MixtralConfig2()
    model = SparseMoELanguageModel(config=config)
    model.apply(kaiming_init_weights)
    model = model.to(run_device)
    total_params = get_model_parameters_numel(model=model)
    print(f'model parameters: {total_params}')
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    #m = model.to(run_device)
    # print the number of parameters in the model
    #print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')
        xb = xb.to(device=run_device)
        yb = yb.to(device=run_device)
        # evaluate the loss
        logits, loss = model(xb, yb)
        with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    #save_model(model=model, model_file_path=model_file_path)


if __name__ == '__main__':
    #one_expert_demo()
    #print('#'*30)
    #mutli_experts_demo()
    #print('#'*30)
    #sMoe_demo()

    run_model()
