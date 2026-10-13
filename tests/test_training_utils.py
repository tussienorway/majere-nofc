"""GPU-free tests for the checkpoint/resume helpers and the wiring in train/qlora_sft.py."""
import ast, json, os

import pytest

from nofc.trainutil import (BASE_MODELS, BNB_4BIT, DEFAULT_SAVE_TOTAL_LIMIT, checkpoint_step, find_latest_checkpoint,
                            resolve_resume, save_kwargs)

H = os.path.dirname(os.path.dirname(__file__))


def mk(run, name, step=None):
    d = run / name; d.mkdir(parents=True)
    if step is not None: (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return str(d)


def test_save_kwargs_defaults_keep_epoch_behaviour():
    assert save_kwargs() == {"save_strategy": "epoch"}
    assert save_kwargs(0, None) == {"save_strategy": "epoch"}


def test_save_kwargs_steps():
    assert save_kwargs(25) == {"save_strategy": "steps", "save_steps": 25, "save_total_limit": DEFAULT_SAVE_TOTAL_LIMIT}
    assert save_kwargs(25, 5)["save_total_limit"] == 5
    assert save_kwargs(0, 3) == {"save_strategy": "epoch", "save_total_limit": 3}
    with pytest.raises(ValueError): save_kwargs(-1)
    with pytest.raises(ValueError): save_kwargs(10, 0)


def test_find_latest_checkpoint_orders_numerically_and_skips_incomplete(tmp_path):
    mk(tmp_path, "checkpoint-9", 9); c100 = mk(tmp_path, "checkpoint-100", 100); mk(tmp_path, "checkpoint-200")
    mk(tmp_path, "tmp-checkpoint-300", 300); mk(tmp_path, "checkpoint-final", 5)
    (tmp_path / "checkpoint-50").write_text("a file, not a dir")
    assert find_latest_checkpoint(str(tmp_path)) == c100
    assert find_latest_checkpoint(str(tmp_path / "missing")) is None
    assert checkpoint_step(c100) == 100 and checkpoint_step(str(tmp_path / "checkpoint-200")) is None


def test_resolve_resume(tmp_path):
    assert resolve_resume(None, str(tmp_path)) is None
    assert resolve_resume("latest", str(tmp_path)) is None  # nothing yet -> start fresh
    c1 = mk(tmp_path, "checkpoint-10", 10); c2 = mk(tmp_path, "checkpoint-20", 20)
    assert resolve_resume("latest", str(tmp_path)) == c2
    assert resolve_resume(c1, "elsewhere") == c1
    assert resolve_resume(str(tmp_path), "elsewhere") == c2
    with pytest.raises(FileNotFoundError): resolve_resume(str(tmp_path / "nope"), str(tmp_path))
    empty = tmp_path / "empty"; empty.mkdir()
    with pytest.raises(FileNotFoundError): resolve_resume(str(empty), str(tmp_path))


def test_training_settings_unchanged():
    assert BNB_4BIT == {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True}
    assert BASE_MODELS == {"qwen3-8b": "Qwen/Qwen3-8B", "normistral-7b": "ltg/normistral-7b-warm"}


def test_qlora_sft_wiring():
    """torch isn't importable in CI, so inspect the script's AST instead of running it."""
    src = open(os.path.join(H, "train", "qlora_sft.py"), encoding="utf-8-sig").read()
    tree = ast.parse(src)
    calls = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            calls[f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")] = n
    kw = {k.arg: k.value for k in calls["SFTConfig"].keywords}
    assert ast.literal_eval(kw["warmup_steps"]) == 5 and ast.literal_eval(kw["optim"]) == "paged_adamw_8bit"
    assert None in kw  # **save_kwargs(...) is spread into SFTConfig; save_strategy is no longer hard-coded
    assert "save_strategy" not in kw
    assert any(k.arg == "resume_from_checkpoint" for k in calls["train"].keywords)
    lora = {k.arg: k.value for k in calls["LoraConfig"].keywords}
    assert ast.literal_eval(lora["target_modules"]) == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    for flag in ("--save-steps", "--save-total-limit", "--resume"):
        assert flag in src
