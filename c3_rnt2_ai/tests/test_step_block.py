from __future__ import annotations
import pytest

torch = pytest.importorskip("torch")

from c3rnt2.model.core_transformer import CoreTransformer


def test_step_block_matches_step():
    settings = {
        "core": {"hidden_size": 64, "layers": 2, "heads": 2, "vocab_size": 256},
        "tokenizer": {"block_size": 64, "vortex_model_path": "data/runs/vortex_tok.pt"},
        "vortex_model": {
            "window_size": 16,
            "latent_slots": 8,
            "lava_top_k": 2,
            "local_mixer_kernel": 3,
            "ssm_state_size": 32,
            "gated_mlp_ratio": 2,
            "lava_read_every": 1,
            "lava_write_every": 1,
        },
    }
    model = CoreTransformer.from_settings(settings)
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        state = model.init_state(batch=1, write_memory=False)
        logits_block, _state_block = model.step_block(tokens, state, write_memory=False)
        state_seq = model.init_state(batch=1, write_memory=False)
        logits_seq = []
        for tok in tokens[0].tolist():
            logits_t, state_seq = model.step(tok, state_seq, write_memory=False)
            logits_seq.append(logits_t)
        logits_seq = torch.stack(logits_seq, dim=1)
    if logits_block.dtype in (torch.float16, torch.bfloat16):
        atol = 2e-2
        rtol = 2e-2
    else:
        atol = 1e-4
        rtol = 1e-4
    assert torch.allclose(logits_block, logits_seq, atol=atol, rtol=rtol)


def test_step_block_matches_cuda_graph():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for graph test")
    from c3rnt2.runtime.graph_runner import build_graph_step_block

    settings = {
        "core": {"hidden_size": 64, "layers": 2, "heads": 2, "vocab_size": 256, "cuda_graphs": True},
        "tokenizer": {"block_size": 64, "vortex_model_path": "data/runs/vortex_tok.pt"},
        "vortex_model": {
            "window_size": 16,
            "latent_slots": 8,
            "lava_top_k": 2,
            "local_mixer_kernel": 3,
            "ssm_state_size": 32,
            "gated_mlp_ratio": 2,
            "lava_read_every": 1,
            "lava_write_every": 1,
        },
    }
    model = CoreTransformer.from_settings(settings)
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        state = model.init_state(batch=1, write_memory=False)
        graph = build_graph_step_block(model, state, token_shape=(1, tokens.size(1)))
        if graph is None:
            pytest.skip("CUDA graph unavailable")
        logits_graph, _ = graph(tokens, state)
        logits_block, _ = model.step_block(tokens, state, write_memory=False)
    atol = 2e-2 if logits_block.dtype in (torch.float16, torch.bfloat16) else 1e-4
    rtol = 2e-2 if logits_block.dtype in (torch.float16, torch.bfloat16) else 1e-4
    assert torch.allclose(logits_graph, logits_block, atol=atol, rtol=rtol)
