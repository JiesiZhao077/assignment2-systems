
import torch
import math
import einops
import torch.nn as nn

class FlashAttention(nn.Module):
    def __init__(self, d_model, num_heads, q_b=64, k_b=64):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.q_block_size = q_b
        self.k_block_size = k_b

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, is_causal: bool=True) -> torch.Tensor:
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)


        Q = einops.rearrange(Q, "B T (num_heads d_k) -> B num_heads T d_k", num_heads=self.num_heads)
        K = einops.rearrange(K, "B T (num_heads d_k) -> B num_heads T d_k", num_heads=self.num_heads)
        V = einops.rearrange(V, "B T (num_heads d_k) -> B num_heads T d_k", num_heads=self.num_heads)

        q_blocks = math.ceil(Q.shape[-2] / self.q_block_size)
        k_blocks = math.ceil(K.shape[-2] / self.k_block_size)


        O = torch.zeros(Q.shape, device=Q.device)

        for i in range(q_blocks):
            Q_i = Q[:, :, i * self.q_block_size : (i+1) * self.q_block_size, :]  # slice by seq_len dimension
            # output
            B, N, q_len, d_k = Q.shape
            prev_out_shape = (B, N, self.q_block_size, d_k)
            prev_out = torch.zeros(prev_out_shape, device=Q.device)
            # init holders
            max_shape = (B, N, self.q_block_size, 1)
            prev_max = torch.full(max_shape, fill_value=float("-inf"), device=Q.device)
            prev_sum = torch.zeros(max_shape, device=Q.device)


            for j in range(k_blocks):
                K_j = K[:, :, j * self.k_block_size : (j+1) * self.k_block_size, :]
                V_j = V[:, :, j * self.k_block_size : (j+1) * self.k_block_size, :]
                prev_max, prev_sum, prev_out = block_scaled_dot_product(
                    Q_i, K_j, V_j, is_causal, prev_max, prev_sum, prev_out, i, self.q_block_size, j, self.k_block_size,
                    )

            O[:, :, i * self.q_block_size : (i+1) * self.q_block_size, :] = prev_out / prev_sum
    

        O = einops.rearrange(O, "B N T D -> B T (N D)")
        return self.o_proj(O)

def block_scaled_dot_product(
    Q, 
    K, 
    V, 
    is_causal, 
    prev_max: torch.Tensor, 
    prev_sum: torch.Tensor, 
    prev_out: torch.Tensor,
    Q_idx,
    Q_block_size,
    K_idx,
    K_block_size,
    ):
    d_k = Q.shape[-1]
    q_len = Q.shape[-2]
    k_len = K.shape[-2]
    attn_score = einops.einsum(Q, K, "B seq_q D, B seq_k D -> B seq_q seq_k")
    scaled_score = attn_score / math.sqrt(d_k)

    if is_causal:
        q_indices = Q_idx * Q_block_size + torch.arange(q_len)
        k_indices = K_idx * K_block_size + torch.arange(k_len)
        q_indices = q_indices.unsqueeze(1)  # q_len, 1
        k_indices = k_indices.unsqueeze(0)  # 1, k_len

        masked = k_indices > q_indices  # q_len, k_len\

        scaled_score = scaled_score.masked_fill(masked, float("-inf"))

    
    curr_max = torch.max(scaled_score, dim=-1, keepdim=True).values # B N seq 1
    cat_max = torch.cat((curr_max, prev_max), -1)
    new_max = torch.max(cat_max, dim=-1, keepdim=True).values # B N seq 2->1

    exp_score = torch.exp(scaled_score - new_max)  # nominator
    correction = torch.exp(prev_max - new_max)
    new_sum = prev_sum * correction + torch.sum(exp_score, dim=-1, keepdim=True) # denom


    curr_out =  einops.einsum(exp_score, V, "B q_len k_len, B k_len D -> B q_len D")
    new_out = prev_out * correction + curr_out
    return new_max, new_sum, new_out

class FlashAttentionAutogradFnSingleHead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal):
        q_block_size = 16
        k_block_size = 16
        # num_heads = 1
        # Q = einops.rearrange(Q, "B T (num_heads d_k) -> B num_heads T d_k", num_heads=num_heads)
        # K = einops.rearrange(K, "B T (num_heads d_k) -> B num_heads T d_k", num_heads=num_heads)
        # V = einops.rearrange(V, "B T (num_heads d_k) -> B num_heads T d_k", num_heads=num_heads)

        q_blocks = math.ceil(Q.shape[-2] / q_block_size)
        k_blocks = math.ceil(K.shape[-2] / k_block_size)


        O = torch.zeros(Q.shape, device=Q.device)
        L = torch.zeros(Q.shape[:-1], device=Q.device).unsqueeze(-1)

        for i in range(q_blocks):
            Q_i = Q[:, i * q_block_size : (i+1) * q_block_size, :]  # slice by seq_len dimension
            # output
            B, q_len, d_k = Q.shape
            prev_out_shape = (B, q_block_size, d_k)
            prev_out = torch.zeros(prev_out_shape, device=Q.device)
            # init holders
            max_shape = (B, q_block_size, 1)
            prev_max = torch.full(max_shape, fill_value=float("-inf"), device=Q.device)
            prev_sum = torch.zeros(max_shape, device=Q.device)


            for j in range(k_blocks):
                K_j = K[:, j * k_block_size : (j+1) * k_block_size, :]
                V_j = V[:, j * k_block_size : (j+1) * k_block_size, :]
                prev_max, prev_sum, prev_out = block_scaled_dot_product(
                    Q_i, K_j, V_j, is_causal, prev_max, prev_sum, prev_out, i, q_block_size, j, k_block_size,
                    )

            O[:, i * q_block_size : (i+1) * q_block_size, :] = prev_out / prev_sum
            L[:, i * q_block_size : (i+1) * q_block_size, :] = prev_max + torch.log(prev_sum)
        
        L = L.squeeze(-1)
        ctx.save_for_backward(L, Q, K, V, O)

        # attn_out = einops.rearrange(output, "B N T D -> B T (N D)")
        return O

