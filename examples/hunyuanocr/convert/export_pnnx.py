"""Export HunyuanOCR (tencent/HunyuanOCR) to ncnn format via pnnx.

Steps:
    pip install "huggingface_hub[cli]" pnnx torch torchvision transformers
    huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR-hf
    python export_pnnx.py --model ./HunyuanOCR-hf --output ./hunyuan_ocr
"""

import argparse
import os
import torch

def _load_model(model_dir: str):
    from transformers import AutoProcessor, AutoModelForVision2Seq  # type: ignore[import-untyped]
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_dir, torch_dtype=torch.float16, trust_remote_code=True
    ).eval()
    return proc, model


def export_vision(model, out_dir: str, patch_size: int) -> None:
    """Export ViT visual encoder."""
    import pnnx  # type: ignore[import-untyped]

    vit = model.visual
    # Representative input: one 14x14 patch grid (196 patches) at fp16
    h = w = 14 * patch_size
    dummy_pixels = torch.zeros(1, 3, h, w, dtype=torch.float16)
    dummy_grid_thw = torch.tensor([[1, h // patch_size, w // patch_size]], dtype=torch.int32)

    pt_path = os.path.join(out_dir, "vision.pt")
    out_param = os.path.join(out_dir, "vision.ncnn.param")
    out_bin = os.path.join(out_dir, "vision.ncnn.bin")

    with torch.no_grad():
        pnnx.export(vit, pt_path, inputs=(dummy_pixels, dummy_grid_thw), fp16=True)
    print(f"[vision] written {out_param}  {out_bin}")


def export_text_embed(model, out_dir: str) -> None:
    """Export token embedding table."""
    import pnnx  # type: ignore[import-untyped]

    embed = model.language_model.model.embed_tokens
    dummy_ids = torch.zeros(1, 16, dtype=torch.long)

    pt_path = os.path.join(out_dir, "text_embed.pt")
    out_param = os.path.join(out_dir, "text_embed.ncnn.param")
    out_bin = os.path.join(out_dir, "text_embed.ncnn.bin")

    with torch.no_grad():
        pnnx.export(embed, pt_path, inputs=(dummy_ids,), fp16=True)
    print(f"[text_embed] written {out_param}  {out_bin}")


def export_text_decoder(model, out_dir: str, hidden_size: int) -> None:
    """Export one decoder forward pass (without KV cache; add_kvcache.py adds it)."""
    import pnnx  # type: ignore[import-untyped]

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.model = lm.model
            self.layers = lm.model.layers

        def forward(self, hidden_states, attention_mask, position_ids):
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
            return self.model.norm(hidden_states)

    wrapper = DecoderWrapper(model.language_model).eval()
    seq = 64
    dummy_hs = torch.zeros(1, seq, hidden_size, dtype=torch.float16)
    dummy_mask = torch.zeros(1, 1, seq, seq, dtype=torch.float16)
    dummy_pos = torch.zeros(3, 1, seq, dtype=torch.long)

    pt_path = os.path.join(out_dir, "text_decoder.pt")
    out_param = os.path.join(out_dir, "text_decoder.ncnn.param")
    out_bin = os.path.join(out_dir, "text_decoder.ncnn.bin")

    with torch.no_grad():
        pnnx.export(
            wrapper, pt_path,
            inputs=(dummy_hs, dummy_mask, dummy_pos),
            fp16=True,
        )
    print(f"[text_decoder] written {out_param}  {out_bin}")


def export_lm_head(model, out_dir: str, hidden_size: int) -> None:
    """Export LM head (linear projection to vocab logits)."""
    import pnnx  # type: ignore[import-untyped]

    head = model.language_model.lm_head
    dummy_hs = torch.zeros(1, 1, hidden_size, dtype=torch.float16)

    pt_path = os.path.join(out_dir, "lm_head.pt")
    out_param = os.path.join(out_dir, "lm_head.ncnn.param")
    out_bin = os.path.join(out_dir, "lm_head.ncnn.bin")

    with torch.no_grad():
        pnnx.export(head, pt_path, inputs=(dummy_hs,), fp16=True)
    print(f"[lm_head] written {out_param}  {out_bin}")


def write_model_cfg(model, proc, out_dir: str) -> None:
    """Write model.cfg consumed by the hunyuanocr binary (key=value, no dependencies)."""
    cfg = model.config
    lm_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    vis_cfg = cfg.vision_config if hasattr(cfg, "vision_config") else cfg

    hidden_size = lm_cfg.hidden_size
    num_layers = lm_cfg.num_hidden_layers
    head_dim = hidden_size // lm_cfg.num_attention_heads
    vocab_size = lm_cfg.vocab_size
    patch_size = vis_cfg.patch_size if hasattr(vis_cfg, "patch_size") else 14
    spatial_merge_size = getattr(vis_cfg, "spatial_merge_size", 2)
    image_token_id = getattr(cfg, "image_token_id", 59280)
    vis_hidden = vis_cfg.hidden_size if hasattr(vis_cfg, "hidden_size") else 1152

    tok = proc.tokenizer
    eos_str = tok.special_tokens_map.get("eos_token", "")
    rope_theta = getattr(lm_cfg, "rope_theta", 10000.0)

    lines = [
        "# HunyuanOCR ncnn model config",
        f"attn_cnt = {num_layers}",
        f"hidden_size = {hidden_size}",
        f"head_dim = {head_dim}",
        f"vocab_size = {vocab_size}",
        f"image_token_id = {image_token_id}",
        f"bos_token_id = {tok.bos_token_id or 120000}",
        f"system_end_token_id = 120021",
        f"user_end_token_id = 120006",
        f"image_start_token_id = 120118",
        f"image_end_token_id = 120119",
        f"special_token_id_begin = 120000",
        f"eos_token = {eos_str}",
        f"rope_theta = {rope_theta}",
        f"rope_alpha = 1000.0",
        f"xdrope_section = 16, 24, 24",
        f"patch_size = {patch_size}",
        f"spatial_merge_size = {spatial_merge_size}",
        f"vision_hidden_size = {vis_hidden}",
        f"min_pixels = {getattr(vis_cfg, 'min_pixels', 12544)}",
        f"max_pixels = {getattr(vis_cfg, 'max_pixels', 9633792)}",
        "vocab_file = vocab.txt",
        "merges_file = merges.txt",
        "vision_param = vision.ncnn.param",
        "vision_bin = vision.ncnn.bin",
        "text_embed_param = text_embed.ncnn.param",
        "text_embed_bin = text_embed.ncnn.bin",
        "text_decoder_param = text_decoder.ncnn.param",
        "text_decoder_bin = text_decoder.ncnn.bin",
        "lm_head_param = lm_head.ncnn.param",
        "lm_head_bin = lm_head.ncnn.bin",
    ]

    path = os.path.join(out_dir, "model.cfg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[model.cfg] written {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HunyuanOCR HF model directory")
    ap.add_argument("--output", default="./hunyuan_ocr", help="Output directory for ncnn assets")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading model (fp16) …")
    proc, model = _load_model(args.model)

    cfg = model.config
    lm_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    vis_cfg = cfg.vision_config if hasattr(cfg, "vision_config") else cfg
    hidden_size = lm_cfg.hidden_size
    patch_size = vis_cfg.patch_size if hasattr(vis_cfg, "patch_size") else 14

    print(f"  hidden_size={hidden_size}, layers={lm_cfg.num_hidden_layers}, "
          f"vocab={lm_cfg.vocab_size}, patch={patch_size}")

    export_vision(model, args.output, patch_size)
    export_text_embed(model, args.output)
    export_text_decoder(model, args.output, hidden_size)
    export_lm_head(model, args.output, hidden_size)
    write_model_cfg(model, proc, args.output)

    print("\nDone. Next steps:")
    print("  1. python add_kvcache.py", os.path.join(args.output, "text_decoder.ncnn.param"))
    print("  2. python export_tokenizer.py --model", args.model, "--output", args.output)


if __name__ == "__main__":
    main()
