
import tiktoken
import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out)) #Careful here, it will autofill, must be torch.rand(d_in, d_out)
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        self.d_out = d_out

    def forward(self, x):

        keys = x @ self.W_key
        values = x @ self.W_value
        queries = x @ self.W_query

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vecs = attn_weights @ values
        return context_vecs

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) #Careful here, it will autofill, must be torch.rand(d_in, d_out)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.d_out = d_out

    def forward(self, x):

        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        attn_scores = queries @ keys.T

        context_length = attn_scores.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)

        context_vecs = attn_weights @ values

        return context_vecs


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) #Careful here, it will autofill, must be torch.rand(d_in, d_out)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.d_out = d_out
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):

        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        # With batch, we are transposing the keys within each batch, so we transpose dimensions (1, 2) or y,z dimensions in (x, y, z)
        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)
        context_vecs = attn_weights @ values

        return context_vecs

def main():
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]
    ])

    torch.manual_seed(789)
    batch = torch.stack([inputs, inputs], dim=0)
    print(batch.shape)

    context_length = batch.shape[1]
    d_in = batch.shape[2]
    d_out = 2
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("context_vec.shape:", context_vecs.shape)



    # sa1 = SelfAttention_v1(d_in = inputs.shape[1], d_out = 2)
    #
    # sa1.W_value.data = sa.W_value.weight.T
    # sa1.W_key.data = sa.W_key.weight.T
    # sa1.W_query.data = sa.W_query.weight.T


if __name__ == '__main__':
    main()



