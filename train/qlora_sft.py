"""QLoRA SFT pilot sized for one 16 GB GPU (e.g. RTX 5060 Ti 16 GB).

  python train/qlora_sft.py --base qwen3-8b       --data pilot_mix.jsonl
  python train/qlora_sft.py --base normistral-7b  --data pilot_mix.jsonl

Writes the adapter plus pilot_metrics.json (tokens/s, peak VRAM, wall time), the numbers we need
to replace the planning estimate on the compute page. Nothing here is a result until it has been run.
"""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from nofc.trainutil import BASE_MODELS as BASES, BNB_4BIT, checkpoint_step, resolve_resume, save_kwargs

class Throughput(TrainerCallback):
    def __init__(self): self.tokens = 0; self.t0 = None
    def on_train_begin(self, *a, **k): self.t0 = time.time(); torch.cuda.reset_peak_memory_stats()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", choices=BASES, required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="e.g. 50 for a quick smoke run")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-steps", type=int, default=0,
                   help="also write a checkpoint every N optimizer steps (default 0: one per epoch, as before)")
    p.add_argument("--save-total-limit", type=int, default=None,
                   help="keep only the newest N checkpoints (default: 2 with --save-steps, else unlimited)")
    p.add_argument("--resume", nargs="?", const="latest", default=None, metavar="PATH",
                   help="resume from the newest checkpoint in --out (no PATH), or from PATH (a checkpoint dir or a run dir); "
                        "starts fresh if --out has no checkpoint yet. Use the same data/--seed/hyperparameters as the original run")
    a = p.parse_args()
    name = BASES[a.base]; out = a.out or f"runs/{a.base}-r{a.rank}-s{a.seed}"
    ckpt = resolve_resume(a.resume, out)
    if a.resume is not None:
        print(f"resuming from {ckpt}" if ckpt else f"--resume: no checkpoint in {out}, starting from step 0", flush=True)
    start_step = checkpoint_step(ckpt) if ckpt else 0

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(**BNB_4BIT, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map={"": 0},
                                                 dtype=torch.bfloat16, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ds = load_dataset("json", data_files=a.data, split="train")
    n_tokens = sum(min(len(tok(t)["input_ids"]), a.max_len) for t in ds["text"])

    cfg = SFTConfig(output_dir=out, num_train_epochs=a.epochs, max_steps=a.max_steps,
                    per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum,
                    learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=5,
                    optim="paged_adamw_8bit", bf16=True, gradient_checkpointing=True,
                    logging_steps=10, seed=a.seed, **save_kwargs(a.save_steps, a.save_total_limit), report_to="none",
                    dataset_text_field="text", max_length=a.max_len)
    cb = Throughput()
    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora, processing_class=tok, callbacks=[cb])
    t0 = time.time(); tr.train(resume_from_checkpoint=ckpt); wall = time.time() - t0
    tr.save_model(out)
    steps = tr.state.global_step; total_steps = tr.state.max_steps or steps
    run_steps = steps - start_step  # steps done in this process; differs from `steps` only when resumed
    tokens_seen = n_tokens * (run_steps / total_steps) * a.epochs if total_steps else n_tokens
    m = {"base": name, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "examples": len(ds), "dataset_tokens": n_tokens,
         "steps": steps, "wall_seconds": round(wall, 1), "approx_tokens_per_second": round(tokens_seen / wall, 1),
         "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2), "max_len": a.max_len, "rank": a.rank, "grad_accum": a.grad_accum,
         "seconds_per_step": round(wall / max(run_steps, 1), 2),
         "tokens_per_step_est": round(tokens_seen / max(run_steps, 1), 1)}
    if ckpt: m.update(resumed_from=ckpt, resumed_from_step=start_step, steps_this_run=run_steps)
    json.dump(m, open(f"{out}/pilot_metrics.json", "w"), indent=2)
    print(json.dumps(m, indent=2))

if __name__ == "__main__":
    main()
