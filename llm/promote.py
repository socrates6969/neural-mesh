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


def ollama(*args: str) -> subprocess.CompletedProcess:
    """one `ollama <args>` call (the single seam the tests stub)."""
    return subprocess.run(["ollama", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")


def ollama_ids() -> dict[str, str]:
    """name -> ID (digest prefix) from `ollama list`; {} if ollama is unavailable."""
    try:
        r = ollama("list")
    except OSError:
        return {}
    out = {}
    for line in (r.stdout or "").splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def ollama_import(src: Path, tag: str, log=print) -> None:
    """src = merged safetensors dir; converted to q8_0 GGUF first."""
    gguf = to_gguf(src, log=log)
    mf = src / "Modelfile"
    mf.write_text(modelfile(gguf.resolve()), encoding="utf-8")
    log(f"$ ollama create {tag} -f {mf}")
    r = ollama("create", tag, "-f", str(mf))
    if r.returncode != 0:
        raise RuntimeError(f"ollama create failed: {(r.stderr or r.stdout)[-800:]}")


def point_latest(tag: str, log=print) -> None:
    """`ollama cp <tag> mesh-llm:latest`, then verify that latest has the same digest as tag."""
    latest = f"{config.OLLAMA_NAME}:latest"
    r = ollama("cp", tag, latest)
    if r.returncode != 0:
        raise RuntimeError(f"ollama cp {tag} {latest} failed: {(r.stderr or r.stdout)[-400:]}")
    ids = ollama_ids()
    if not ids.get(tag) or ids.get(tag) != ids.get(latest):
        raise RuntimeError(f"{latest} digest {ids.get(latest)} != {tag} digest {ids.get(tag)} after cp")
    log(f"{latest} -> {tag} ({ids[tag]})")


def restore_latest(prev_tag: str | None, log=print) -> None:
    """best effort: put mesh-llm:latest back on the previous production tag (or remove it when there was none)."""
    latest = f"{config.OLLAMA_NAME}:latest"
    try:
        r = ollama("cp", prev_tag, latest) if prev_tag else ollama("rm", latest)
        log(f"restored {latest} -> {prev_tag or '(none)'}" + ("" if r.returncode == 0 else f" FAILED: {(r.stderr or r.stdout)[-200:]}"))
    except OSError as e:
        log(f"could not restore {latest}: {e}")


def decided(st: dict, ver: str) -> bool:
    """a version is decided once promoted/rejected/rolled back/abandoned; a failed promotion is retried once more."""
    hs = [h for h in st.get("history", []) if h.get("ver") == ver]
    fails = sum(1 for h in hs if h.get("action") == "promote_failed")
    return any(h.get("action") != "promote_failed" for h in hs) or fails >= 2


def lineage_ids(ver: str) -> list[str]:
    """union of meta.new_ids along ver's parent chain (what the production lineage has been trained on)."""
    ids: set[str] = set()
    seen = set()
    cur = ver
    while cur and cur not in seen:
        seen.add(cur)
        try:
            meta = json.loads((config.ADAPTERS / cur / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            break
        ids |= set(meta.get("new_ids", []))
        cur = meta.get("parent")
    return sorted(ids)


def run(synthetic: bool = False, ver: str | None = None, force: bool = False, keep_merged: bool = False, log=print) -> dict:
    """returns the history record; rec["exit"]: 0 promoted / not promoted / nothing to do, 3 promotion failed (cleaned up)."""
    from .evaluate import latest_candidate
    ver = ver or latest_candidate(synthetic)
    st = config.load_state()
    ev_path = config.ADAPTERS / str(ver) / "eval.json"
    if not ver or not ev_path.exists():
        log("no evaluated candidate; nothing to promote"); return {"promoted": False, "reason": "no evaluated candidate", "exit": 0}
    if decided(st, ver) and not force:
        log(f"{ver} already decided; nothing to do"); return {"promoted": False, "reason": "already decided", "exit": 0}
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    meta = json.loads((config.ADAPTERS / ver / "meta.json").read_text(encoding="utf-8"))
    rec = {"ver": ver, "ts": datetime.now(timezone.utc).isoformat(), "synthetic": synthetic, "gate": ev["gate"],
           "loss_diff": ev["loss"]["diff_cluster_ci"],
           "pairwise_winrate": (ev.get("pairwise") or {}).get("winrate_ci"), "battery_diff": (ev.get("battery") or {}).get("diff_cluster_ci")}
    if not ev["gate"]["promote"] and not force:
        rec.update(promoted=False, action="reject")
        st.setdefault("history", []).append(rec)
        if not synthetic:
            from . import lineage
            rec["lineage"] = lineage.record(st, ver, promoted=False, log=log)
        config.save_state(st)
        log(f"NOT PROMOTED {ver}: {ev['gate']['reason']}; production stays {ev['production'] or 'BASE'}")
        return {**rec, "exit": 0}
    tag = f"{config.OLLAMA_NAME}:{ver}"
    out = config.MERGED / ver
    prev = (st.get("production") or {}) if not synthetic else {}
    # tags that must survive a failed (e.g. --force re-)promotion: the live production tags, and a tag that already existed in
    # Ollama before this call (we only ever `ollama rm` a tag this call created)
    live = {t for t in ((st.get("production") or {}).get("ollama"), (st.get("synthetic_production") or {}).get("ollama")) if t}
    pre_existing = tag in ollama_ids()
    touched_latest = False
    try:
        merge(config.base_path(meta["base"]), config.ADAPTERS / ver, out, log=log)
        ollama_import(out, tag, log=log)
        if not synthetic:
            touched_latest = True
            point_latest(tag, log=log)
            st["production"] = {"ver": ver, "ollama": tag, "base": meta["base"], "since": rec["ts"], "prev_ver": prev.get("ver")}
            st["trained_ids"] = sorted(set(st.get("trained_ids", [])) | set(meta["new_ids"]))
        else:
            st["synthetic_production"] = {"ver": ver, "ollama": tag, "base": meta["base"], "since": rec["ts"]}
            st["synthetic_trained_ids"] = sorted(set(st.get("synthetic_trained_ids", [])) | set(meta["new_ids"]))
    except Exception as e:  # noqa: BLE001 - any failure: undo the Ollama side, record it, keep production as it was
        log(f"PROMOTION FAILED {ver}: {e}; cleaning up")
        if tag in live or pre_existing:
            log(f"keeping {tag}: " + ("it is the live production tag" if tag in live else "it existed before this call"))
        else:
            try:
                ollama("rm", tag)
            except OSError:
                pass
        if touched_latest:
            restore_latest(prev.get("ollama"), log=log)
        rec.update(promoted=False, action="promote_failed", error=str(e)[-800:])
        st = config.load_state()
        st.setdefault("history", []).append(rec); config.save_state(st)
        return {**rec, "exit": 3}
    finally:
        if not keep_merged:
            shutil.rmtree(out, ignore_errors=True)  # Ollama holds its own GGUF copy; the adapter dir stays versioned
    rec.update(promoted=True, action="promote", ollama=tag, forced=force)
    st.setdefault("history", []).append(rec)
    if not synthetic:
        from . import lineage
        rec["lineage"] = lineage.record(st, ver, promoted=True, log=log)
    config.save_state(st)
    log(f"PROMOTED {ver} -> {tag}" + ("" if synthetic else f" (alias {config.OLLAMA_NAME}:latest)"))
    return {**rec, "exit": 0}


def rollback(to: str, keep_merged: bool = False, log=print) -> dict:
    """`python -m llm promote --to VER`: make an earlier promoted, non-synthetic version production again.
    exit 4 if VER was never promoted (or is synthetic / its adapter is gone); re-imports the Ollama tag from the adapter if it
    is missing; points mesh-llm:latest at it (digest verified); trained_ids := union of meta.new_ids along VER's parent chain
    (so the next train round re-learns the rows the abandoned lineage had consumed); history action rollback. exit 3 on failure."""
    st = config.load_state()
    ok = any(h.get("ver") == to and h.get("promoted") and not h.get("synthetic") and h.get("action", "promote") == "promote"
             for h in st.get("history", []))
    meta_p = config.ADAPTERS / to / "meta.json"
    if not ok or to.endswith("-syn") or not meta_p.exists():
        log(f"cannot roll back to {to}: not a promoted, non-synthetic version with its adapter on disk")
        return {"ver": to, "promoted": False, "reason": "not a promoted non-synthetic version", "exit": 4}
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    tag = f"{config.OLLAMA_NAME}:{to}"
    prev = st.get("production") or {}
    out = config.MERGED / to
    ts = datetime.now(timezone.utc).isoformat()
    reimported = False
    try:
        if tag not in ollama_ids():
            log(f"{tag} missing in Ollama; re-importing from the adapter")
            reimported = True
            merge(config.base_path(meta["base"]), config.ADAPTERS / to, out, log=log)
            ollama_import(out, tag, log=log)
        point_latest(tag, log=log)
    except Exception as e:  # noqa: BLE001
        log(f"ROLLBACK FAILED {to}: {e}; restoring")
        if reimported:
            try:
                ollama("rm", tag)
            except OSError:
                pass
        restore_latest(prev.get("ollama"), log=log)
        rec = {"ver": to, "ts": ts, "synthetic": False, "promoted": False, "action": "rollback_failed", "error": str(e)[-800:]}
        st.setdefault("history", []).append(rec); config.save_state(st)
        return {**rec, "exit": 3}
    finally:
        if not keep_merged:
            shutil.rmtree(out, ignore_errors=True)
    st["production"] = {"ver": to, "ollama": tag, "base": meta["base"], "since": ts, "prev_ver": prev.get("ver"),
                        "rolled_back_from": prev.get("ver")}
    st["trained_ids"] = lineage_ids(to)
    from . import lineage
    lineage.reset_to(st, to, meta["base"], f"rollback to {to}")
    rec = {"ver": to, "ts": ts, "synthetic": False, "promoted": True, "action": "rollback", "from": prev.get("ver"),
           "reimported": reimported, "trained_ids": len(st["trained_ids"])}
    st.setdefault("history", []).append(rec); config.save_state(st)
    log(f"ROLLED BACK production {prev.get('ver')} -> {to} ({tag}); trained_ids reset to its lineage ({len(st['trained_ids'])})")
    return {**rec, "exit": 0}
