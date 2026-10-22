"""Human sign-off for a knowledge pack: print it, ask for confirmation, record its sha256 in the live registry.

    python -m train.approve --batch batch1 --agent mesh-mathematician --field math

Only packs whose training status is "promoted" (significant held-out gain) can be approved. Run it yourself in an
interactive terminal: it needs a typed confirmation and refuses to run without a console.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .loop import ROOT, slug

REGISTRY = Path(r"C:\Users\mariu\neural-mesh\data\lifelong\mesh\packs.json")


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--batch", required=True)
    a.add_argument("--agent", required=True)
    a.add_argument("--field", required=True)
    a.add_argument("--registry", default=str(REGISTRY))
    o = a.parse_args(argv)
    d = ROOT / "train" / o.batch / "packs" / slug(o.agent) / o.field
    st = json.loads((d / "status.json").read_text(encoding="utf-8"))
    if st.get("status") != "promoted":
        sys.exit("not promoted (status=%s): only packs with significant held-out gains can be approved" % st.get("status"))
    pack = st["current"]
    print(pack)
    print("\n--- held-out evidence:", json.dumps(st.get("pooled")))
    if not sys.stdin.isatty():
        sys.exit("refusing: run this in an interactive terminal")
    if input("Approve this exact text for live use by %s on %s tasks? Type APPROVE: " % (o.agent, o.field)).strip() != "APPROVE":
        sys.exit("not approved")
    reg_p = Path(o.registry)
    reg = json.loads(reg_p.read_text(encoding="utf-8")) if reg_p.exists() else {}
    reg.setdefault(o.agent, {})[o.field] = {"pack": pack, "status": "promoted", "pooled": st.get("pooled"),
                                           "approved_sha256": hashlib.sha256(pack.encode("utf-8")).hexdigest()}
    reg_p.parent.mkdir(parents=True, exist_ok=True)
    reg_p.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    print("approved ->", reg_p)


if __name__ == "__main__":
    main()
