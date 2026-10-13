"""Pure-Python helpers shared by train/qlora_sft.py and nofc/serve.py (no torch import, unit-testable)."""
import json, os, re

BASE_MODELS = {"qwen3-8b": "Qwen/Qwen3-8B", "normistral-7b": "ltg/normistral-7b-warm"}

# Quantisation settings shared by training and serving; callers add bnb_4bit_compute_dtype=torch.bfloat16.
BNB_4BIT = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

DEFAULT_SAVE_TOTAL_LIMIT = 2
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def save_kwargs(save_steps=0, save_total_limit=None):
    """SFTConfig kwargs. Defaults reproduce the original behaviour (save_strategy='epoch', no limit)."""
    if save_steps and save_steps < 0: raise ValueError("--save-steps must be >= 1")
    if save_total_limit is not None and save_total_limit < 1: raise ValueError("--save-total-limit must be >= 1")
    if save_steps:
        return dict(save_strategy="steps", save_steps=save_steps,
                    save_total_limit=save_total_limit or DEFAULT_SAVE_TOTAL_LIMIT)
    kw = dict(save_strategy="epoch")
    if save_total_limit: kw["save_total_limit"] = save_total_limit
    return kw


def checkpoint_step(path):
    """Global step recorded in a checkpoint's trainer_state.json, or None if the checkpoint is incomplete."""
    f = os.path.join(path, "trainer_state.json")
    if not os.path.isfile(f): return None
    try:
        with open(f, encoding="utf-8") as fh: return int(json.load(fh)["global_step"])
    except (OSError, ValueError, KeyError): return None


def find_latest_checkpoint(out_dir):
    """Highest-numbered complete `checkpoint-N` in out_dir (ignores `tmp-checkpoint-*` and ones without trainer_state.json)."""
    if not os.path.isdir(out_dir): return None
    found = []
    for name in os.listdir(out_dir):
        m = _CKPT_RE.match(name)
        path = os.path.join(out_dir, name)
        if m and os.path.isdir(path) and checkpoint_step(path) is not None: found.append((int(m.group(1)), path))
    return max(found)[1] if found else None


def resolve_resume(resume, out_dir):
    """`--resume` value -> checkpoint path or None (start fresh).

    resume: None (not requested) | "latest" (newest checkpoint in out_dir; None if there is none yet)
            | a path to a checkpoint dir, or to a run dir that contains checkpoint-N dirs.
    """
    if resume is None: return None
    if resume == "latest": return find_latest_checkpoint(out_dir)
    if not os.path.isdir(resume): raise FileNotFoundError(f"--resume path not found: {resume}")
    if checkpoint_step(resume) is not None: return resume
    latest = find_latest_checkpoint(resume)
    if latest is None: raise FileNotFoundError(f"no complete checkpoint (trainer_state.json) in {resume}")
    return latest
