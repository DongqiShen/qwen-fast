import torch
import torch.nn as nn
from safetensors import safe_open

# model_path = "Qwen-18B-Chat/model-00001-of-00002.safetensors"
# tensors = {}
# with safe_open(model_path, framework="pt", device=0) as f:
#     for k in f.keys():
#         tensors[k] = f.get_tensor(k)

# print("Hello World!")

def _split_heads(tensor, num_heads, attn_head_size):
    new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
    tensor = tensor.view(new_shape)
    return tensor

dim = 2048

x = torch.rand(1, 24, 2048, dtype=torch.float32)

wqkv = nn.Linear(dim, dim * 3)
nn.init.uniform_(wqkv.weight, a=0.0, b=1.0)

wkv = wqkv(x)

q, k, v = wkv.split([dim, dim, dim], dim=-1)
q = q.view(1, 24, 16, 128)

qq, kk, vv = wkv.split(dim, dim=2)
qq = _split_heads(qq, 16, 128)

is_equal_q = torch.equal(q, qq)
is_equal_k = torch.equal(k, kk)
is_equal_v = torch.equal(v, vv)


print("Hello World!")