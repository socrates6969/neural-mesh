"""`python -m llm promote`: apply the eval gate to the latest evaluated candidate. On PASS: merge LoRA into the bf16 base
(CPU), save safetensors, import into Ollama (`ollama create mesh-llm:<ver>` from a Modelfile whose FROM points at the
GGUF), then alias
`mesh-llm:latest` -> the new version and record production. On FAIL: keep the old model, log why. Synthetic candidates
never move `mesh-llm:latest`.

GGUF path (measured 2026-10-10): Ollama 0.40's safetensors import rejects Qwen2ForCausalLM ("unsupported MLX
architecture"), so we convert with llama.cpp's own convert_hf_to_gguf.py from its official GitHub release tarball
(models/tools/llama.cpp-<tag>, its gguf-py installed into the venv) straight to q8_0, then `ollama create` FROM the .gguf."""
from __future__ import annotations

import gc
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import config

CHATML = ('{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
          '{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n')


def modelfile(src: Path) -> str:
    return (f'FROM {src.as_posix()}\nTEMPLATE """{CHATML}"""\nSYSTEM """You are a concise, careful coding assistant."""\n'
            'PARAMETER stop "<|im_end|>"\nPARAMETER stop "<|im_start|>"\nPARAMETER temperature 0\n')


def merge(base: Path, adapter: Path | None, out: Path, log=print) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": "cpu"})
    if adapter is not None:
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    del m; gc.collect()
    log(f"merged -> {out}")
    return out


def convert_script() -> Path:
    c = sorted((config.MODELS / "tools").glob("llama.cpp-*/convert_hf_to_gguf.py"))
    if not c:
        raise RuntimeError("llama.cpp convert script missing: download the official release tarball into llm/models/tools/ (see llm/README.md)")
    return c[-1]


def to_gguf(src: Path, outtype: str = "q8_0", log=print) -> Path:
    import sys
    out = src / f"model-{outtype}.gguf"
    r = subprocess.run([sys.executable, str(convert_script()), str(src), "--outtype", outtype, "--outfile", str(out)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"GGUF conversion failed: {(r.stderr or r.stdout)[-800:]}")
    log(f"gguf -> {out} ({out.stat().st_size / 2**30:.2f} GiB)")
    return out


def ollama_import(src: Path, tag: str, log=print) -> None:
    """src = merged safetensors dir; converted to q8_0 GGUF first."""
    gguf = to_gguf(src, log=log)
    mf = src / "Modelfile"
    mf.write_text(modelfile(gguf.resolve()), encoding="utf-8")
    cmd = ["ollama", "create", tag, "-f", str(mf)]
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"ollama create failed: {(r.stderr or r.stdout)[-800:]}")


def run(synthetic: bool = False, ver: str | None = None, force: bool = False, keep_merged: bool = False, log=print) -> dict:
    from .evaluate import latest_candidate
    ver = ver or latest_candidate(synthetic)
    st = config.load_state()
    ev_path = config.ADAPTERS / str(ver) / "eval.json"
    if not ver or not ev_path.exists():
        log("no evaluated candidate; nothing to promote"); return {"promoted": False, "reason": "no evaluated candidate"}
    if any(h.get("ver") == ver for h in st.get("history", [])) and not force:
        log(f"{ver} already decided; nothing to do"); return {"promoted": False, "reason": "already decided"}
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    meta = json.loads((config.ADAPTERS / ver / "meta.json").read_text(encoding="utf-8"))
    rec = {"ver": ver, "ts": datetime.now(timezone.utc).isoformat(), "synthetic": synthetic, "gate": ev["gate"],
           "loss_diff": ev["loss"]["diff_cluster_ci"], "battery_diff": ev["battery"]["diff_cluster_ci"]}
    if not ev["gate"]["promote"] and not force:
        rec["promoted"] = False
        st.setdefault("history", []).append(rec); config.save_state(st)
        log(f"NOT PROMOTED {ver}: {ev['gate']['reason']}; production stays {ev['production'] or 'BASE'}")
        return rec
    tag = f"{config.OLLAMA_NAME}:{ver}"
    out = config.MERGED / ver
    merge(config.base_path(meta["base"]), config.ADAPTERS / ver, out, log=log)
    ollama_import(out, tag, log=log)
    if not synthetic:
        subprocess.run(["ollama", "cp", tag, f"{config.OLLAMA_NAME}:latest"], check=True, capture_output=True)
        st["production"] = {"ver": ver, "ollama": tag, "base": meta["base"], "since": rec["ts"]}
        st["trained_ids"] = sorted(set(st.get("trained_ids", [])) | set(meta["new_ids"]))
    else:
        st["synthetic_production"] = {"ver": ver, "ollama": tag, "base": meta["base"], "since": rec["ts"]}
        st["synthetic_trained_ids"] = sorted(set(st.get("synthetic_trained_ids", [])) | set(meta["new_ids"]))
    if not keep_merged:
        shutil.rmtree(out, ignore_errors=True)  # Ollama holds its own GGUF copy; the adapter dir stays versioned
    rec.update(promoted=True, ollama=tag, forced=force)
    st.setdefault("history", []).append(rec); config.save_state(st)
    log(f"PROMOTED {ver} -> {tag}" + ("" if synthetic else f" (alias {config.OLLAMA_NAME}:latest)"))
    return rec
