from __future__ import annotations

import torch

from c3rnt2.model.core_transformer import CoreTransformer


def test_step_matches_forward():
    settings = {
        "core": {"hidden_size": 64, "layers": 2, "heads": 2, "vocab_size": 256},
        "tokenizer": {"block_size": 64, "vortex_model_path": "data/runs/vortex_tok.pt"},
        "vortex_model": {"window_size": 16, "latent_slots": 8, "lava_top_k": 2, "local_mixer_kernel": 3, "ssm_state_size": 32, "gated_mlp_ratio": 2},
    }
    model = CoreTransformer.from_settings(settings)
    ids = [1, 2, 3, 4]
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits_full = model.forward(input_ids)
        last_full = logits_full[:, -1, :]
        state = model.init_state(batch=1)
        last_logits = None
        for tok in ids:
            last_logits, state = model.step(tok, state, write_memory=False)
        assert last_logits is not None
        atol = 1e-2 if last_full.dtype in (torch.float16, torch.bfloat16) else 1e-4
        assert torch.allclose(last_logits, last_full, atol=atol)


def test_chat_uses_cuda_if_available():
    if not torch.cuda.is_available():
        return
    settings = {
        "core": {"hidden_size": 64, "layers": 2, "heads": 2, "vocab_size": 256},
        "tokenizer": {"block_size": 64, "vortex_model_path": "data/runs/vortex_tok.pt"},
        "vortex_model": {"window_size": 16, "latent_slots": 8, "lava_top_k": 2, "local_mixer_kernel": 3, "ssm_state_size": 32, "gated_mlp_ratio": 2},
    }
    model = CoreTransformer.from_settings(settings)
    device = next(model.parameters()).device
    assert device.type == "cuda"
