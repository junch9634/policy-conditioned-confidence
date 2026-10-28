from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .response_parser import parse_event_verdict as parse_base
ORDER = ("YES", "NO", "UNKNOWN")
LABEL_TEXT = {"YES": "네", "NO": "아니요", "UNKNOWN": "판단불가"}
LABEL_FIRST = {"YES": "네", "NO": "아", "UNKNOWN": "판"}


def strip_image_token(prompt: str) -> str:
    return re.sub(r"^\s*<image>\s*", "", prompt, flags=re.IGNORECASE).strip()


def parse_verdict(text: str) -> str | None:
    return {"yes": "YES", "no": "NO", "unknown": "UNKNOWN"}.get(parse_base(text))


def strict_verdict(text: str) -> str | None:
    match = re.match(r"^\s*(네|아니요|판단불가)\s*,", text or "")
    return None if not match else {"네": "YES", "아니요": "NO", "판단불가": "UNKNOWN"}[match.group(1)]


def softmax(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    array -= np.max(array)
    exp = np.exp(array)
    return (exp / exp.sum()).tolist()


def build_messages(image_source, prompt: str) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "image", "image": image_source},
        {"type": "text", "text": strip_image_token(prompt)},
    ]}]


def verify_candidate_interface(processor, first_record: dict, project_root: Path) -> dict:
    image_path = project_root / first_record["image_project_relative"]
    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")
    messages = build_messages(image_rgb, first_record["prompt"])
    prompt = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    prompt_ids = prompt["input_ids"][0].tolist()
    labels = {}
    for label in ORDER:
        full = processor.apply_chat_template(
            messages + [{"role": "assistant", "content": LABEL_TEXT[label]}],
            tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt",
        )
        full_ids = full["input_ids"][0].tolist()
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"Chat-template prefix mismatch for {label}")
        suffix = full_ids[len(prompt_ids):]
        standalone = processor.tokenizer.encode(LABEL_TEXT[label], add_special_tokens=False)
        if not suffix:
            raise RuntimeError(f"Missing assistant answer token for {label}")
        labels[label] = {
            "full_label": LABEL_TEXT[label],
            "first_token_text": LABEL_FIRST[label],
            "standalone_token_ids": [int(value) for value in standalone],
            "context_suffix_token_ids": [int(value) for value in suffix],
            "candidate_first_token_id": int(suffix[0]),
            "candidate_first_token_decoded": processor.tokenizer.decode([suffix[0]]),
        }
    candidate_ids = [labels[label]["candidate_first_token_id"] for label in ORDER]
    if len(set(candidate_ids)) != 3:
        raise RuntimeError(f"Candidate first-token IDs not distinct: {candidate_ids}")
    for label in ORDER:
        expected = processor.tokenizer.encode(LABEL_FIRST[label], add_special_tokens=False)
        if len(expected) != 1 or int(expected[0]) != labels[label]["candidate_first_token_id"]:
            raise RuntimeError(f"First-token contract mismatch for {label}")
    return {
        "verified": True,
        "prompt_length_first_input": len(prompt_ids),
        "labels": labels,
        "candidate_ids": {label: labels[label]["candidate_first_token_id"] for label in ORDER},
    }


def extract_step0_hidden(generated, inputs: dict, model, first_logits: torch.Tensor) -> tuple[np.ndarray, dict]:
    hidden_steps = getattr(generated, "hidden_states", None)
    if not hidden_steps:
        raise RuntimeError("Generation output has no hidden_states")
    step0 = hidden_steps[0]
    if not step0:
        raise RuntimeError("Generation step 0 has no layer hidden states")
    last_layer = step0[-1]
    if last_layer.ndim != 3 or last_layer.shape[0] != 1:
        raise RuntimeError(f"Unexpected last-layer shape: {tuple(last_layer.shape)}")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None or attention_mask.shape[0] != 1:
        raise RuntimeError("Missing or unexpected attention mask")
    nonzero = torch.nonzero(attention_mask[0], as_tuple=False).flatten()
    if nonzero.numel() == 0:
        raise RuntimeError("No non-padding prompt token")
    prompt_position = int(nonzero[-1].item())
    if prompt_position >= last_layer.shape[1]:
        raise RuntimeError("Prompt position outside step-0 hidden sequence")
    hidden = last_layer[0, prompt_position, :]
    with torch.inference_mode():
        projected = model.lm_head(hidden.unsqueeze(0))[0].detach().float()
    raw = first_logits.detach().float()
    if projected.shape != raw.shape:
        raise RuntimeError("LM-head projection and generation-logit shapes differ")
    diff = torch.abs(projected - raw)
    metadata = {
        "generation_step_index": 0,
        "hidden_layer_index": len(step0) - 1,
        "hidden_state_layer_count_including_embedding": len(step0),
        "prompt_token_index": prompt_position,
        "input_id_sequence_length": int(inputs["input_ids"].shape[1]),
        "attention_nonpadding_count": int(attention_mask[0].sum().item()),
        "step0_hidden_sequence_length": int(last_layer.shape[1]),
        "hidden_dimension": int(hidden.shape[0]),
        "extraction_dtype": str(hidden.dtype).replace("torch.", ""),
        "storage_dtype": "float32",
        "lm_head_full_vocab_max_abs_diff_vs_generation_raw_logits": float(diff.max().item()),
        "lm_head_full_vocab_mean_abs_diff_vs_generation_raw_logits": float(diff.mean().item()),
        "connection_check_finite": bool(torch.isfinite(diff).all().item()),
    }
    return hidden.detach().float().cpu().numpy(), metadata


