import pytest
torch = pytest.importorskip("torch")

from c3rnt2.model.core_transformer import CoreTransformer, save_checkpoint


def test_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(1234)
    settings = {
        "core": {"hidden_size": 64, "layers": 2, "heads": 2, "vocab_size": 256, "checkpoint_path": str(tmp_path / "ckpt.pt")},
        "tokenizer": {"block_size": 16, "vortex_model_path": str(tmp_path / "vortex_tok.pt")},
        "vortex_model": {"window_size": 16, "latent_slots": 8, "lava_top_k": 2, "local_mixer_kernel": 3, "ssm_state_size": 32, "gated_mlp_ratio": 2},
    }
    model = CoreTransformer.from_settings(settings)
    model.eval()
    model.to("cpu", dtype=torch.float32)
    model.device = torch.device("cpu")
    save_checkpoint(model, settings["core"]["checkpoint_path"], settings)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=torch.device("cpu"))
    with torch.inference_mode():
        logits_a = model.forward(input_ids)
    model2 = CoreTransformer.from_settings(settings)
    model2.eval()
    model2.to("cpu", dtype=torch.float32)
    model2.device = torch.device("cpu")
    with torch.inference_mode():
        logits_b = model2.forward(input_ids)
    assert torch.allclose(logits_a, logits_b)
