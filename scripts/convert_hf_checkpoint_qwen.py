# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import sys
from pathlib import Path
from typing import Optional
from safetensors import safe_open

import torch
import re

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model_qwen import ModelArgs

@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("Qwen-1_8B"),
    model_name: Optional[str] = None,
) -> None:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json = checkpoint_dir / "model.safetensors.index.json"

    assert model_map_json.is_file()

    with open(model_map_json) as json_map:
        bin_index = json.load(json_map)

    # weight_map = {
    #     "model.embed_tokens.weight": "tok_embeddings.weight",
    #     "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    #     "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    #     "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    #     "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    #     'model.layers.{}.self_attn.rotary_emb.inv_freq': None,
    #     'model.layers.{}.mlp.gate_proj.weight': 'layers.{}.feed_forward.w1.weight',
    #     "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    #     "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    #     "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    #     "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    #     "model.norm.weight": "norm.weight",
    #     "lm_head.weight": "output.weight",
    # }

    weight_map = {
        "transformer.wte.weight": "tok_embeddings.weight",
        "transformer.h.{}.attn.c_attn.bias": "layers.{}.attention.wqkv.bias",
        "transformer.h.{}.attn.c_attn.weight": "layers.{}.attention.wqkv.weight",
        "transformer.h.{}.attn.c_proj.weight": "layers.{}.attention.wo.weight",
        "transformer.h.{}.attention_norm.weight": "layers.{}.attention_norm.weight",
        "transformer.h.{}.ffn_norm.weight": "layers.{}.ffn_norm.weight",
        "transformer.h.{}.mlp.c_proj.weight": "layers.{}.feed_forward.w2.weight",
        "transformer.h.{}.mlp.wup.weight": "layers.{}.feed_forward.w1.weight",
        "transformer.h.{}.mlp.wdown.weight": "layers.{}.feed_forward.w3.weight",
        "transformer.ln_f.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    
    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}

    def permute(w, n_head):
        dim = config.dim
        return (
            w.view(n_head, 2, config.head_dim // 2, dim)
            .transpose(1, 2)
            .reshape(config.head_dim * n_head, dim)
        )

    merged_result = {}
    # for file in sorted(bin_files):
    #     state_dict = torch.load(str(file), map_location="cpu", mmap=False, weights_only=True)
    #     merged_result.update(state_dict)
    # tensors = {}
    for file in sorted(bin_files):
        with safe_open(file, framework="pt", device="cpu") as f:  
            for k in f.keys():
                merged_result[k] = f.get_tensor(k).to("cpu").to(torch.float16)

    final_result = {}
    for key, value in merged_result.items():
        if ".h." in key:
            key = key.replace("ln_1", "attention_norm")
            key = key.replace("ln_2", "ffn_norm")
            key = key.replace("w1", "wup")
            key = key.replace("w2", "wdown")
            abstract_key = re.sub(r'(\d+)', '{}', key)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert HuggingFace checkpoint.')
    parser.add_argument('--checkpoint_dir', type=Path, default=Path("Qwen-1_8B"))
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
