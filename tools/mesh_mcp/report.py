"""report_imrad: assemble an IMRaD markdown report plus an audit-trail appendix and save it. A formatter, no LLM."""
from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path

from tools.mesh_mcp.common import as_json, cap

DEFAULT_DIR = r"C:\Users\mariu\ruv\hive\runs\reports"
ORDER = [("abstract", "Abstract"), ("introduction", "Introduction"), ("methods", "Methods"),
         ("results", "Results"), ("discussion", "Discussion"), ("conclusion", "Conclusion"),
         ("limitations", "Limitations")]
AUDIT_COLS = ["ts", "agent", "tool", "action", "result", "ref"]


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:60] or "report"


def _cell(v) -> str:
    return str("" if v is None else v).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def build(title: str, sections: dict, audit: list) -> str:
    if not isinstance(sections, dict):
        raise ValueError("sections_json must be an object like {\"introduction\": \"...\", \"methods\": \"...\"}")
    if not isinstance(audit, list):
        raise ValueError("audit_trail_json must be a list of step objects")
    sec = {str(k).lower(): v for k, v in sections.items()}
    missing = [n for k, n in ORDER[1:5] if not str(sec.get(k) or "").strip()]
    out = ["# " + title.strip(), "", "_Generated %s by mesh-tools report_imrad._"
           % dt.datetime.now().strftime("%Y-%m-%d %H:%M")]
    if missing:
        out += ["", "> Missing IMRaD sections: %s." % ", ".join(missing)]
    known = {k for k, _ in ORDER} | {"references"}
    for key, name in ORDER + [(k, k.replace("_", " ").title()) for k in sec if k not in known]:
        body = sec.get(key)
        if body is not None and str(body).strip():
            out += ["", "## " + name, "", str(body).strip()]
    refs = sec.get("references")
    if refs:
        out += ["", "## References", ""]
        out += ["%d. %s" % (i, r) for i, r in enumerate(refs, 1)] if isinstance(refs, list) else [str(refs)]
    out += ["", "## Appendix A. Audit trail", ""]
    if audit:
        cols = AUDIT_COLS + sorted({k for s in audit if isinstance(s, dict) for k in s} - set(AUDIT_COLS))
        cols = [c for c in cols if any(isinstance(s, dict) and s.get(c) not in (None, "") for s in audit)]
        out += ["| # | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for i, s in enumerate(audit, 1):
            s = s if isinstance(s, dict) else {"action": s}
            out.append("| %d | %s |" % (i, " | ".join(_cell(s.get(c)) for c in cols)))
    else:
        out.append("_No audit trail supplied: results in this report are not traceable._")
    return "\n".join(out) + "\n"


def report_imrad(title: str, sections_json, audit_trail_json=None) -> str:
    title = str(title or "").strip()
    if not title:
        raise ValueError("title is required")
    md = build(title, as_json(sections_json, "sections_json"),
               as_json(audit_trail_json, "audit_trail_json") if audit_trail_json not in (None, "") else [])
    d = Path(os.environ.get("MESH_REPORTS_DIR", DEFAULT_DIR))
    d.mkdir(parents=True, exist_ok=True)
    path = d / ("%s-%s.md" % (dt.datetime.now().strftime("%Y%m%d-%H%M%S"), _slug(title)))
    path.write_text(md, encoding="utf-8")
    return cap("saved %s (%d chars)\n--- preview ---\n%s" % (path, len(md), md[:1500]))
