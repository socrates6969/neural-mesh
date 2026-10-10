"""Builds questions.jsonl (agent-task set for the codemem A/B). Topics were mined from past Claude Code transcripts
(~/.claude/projects/*/*.jsonl, sessions whose cwd is one of the user's repos): every sentence there that cites `path:line` for a
claim about the code ("X is enforced at ...", "Y is defined at ...") became a candidate; question wording and the answer key (file +
line range, checked against the current tree by --check) were written by hand from the repo, NOT from codemem output.
Splits: i % 3 == 0 -> tune (used to set the hook gate/floor), the rest -> held (used for the A/B).
Usage: python build_questions.py [--check]"""
import json, pathlib, sys

FB = "C:/Users/mariu/ventures/freight-recovery-build"
FR = "C:/Users/mariu/ventures/freight-recovery"
NM = "C:/Users/mariu/neural-mesh"
REPOS = {"freight-recovery-build": FB, "freight-recovery": FR, "neural-mesh": NM}
B = "freight-recovery-build"
Q = [  # (repo, question, [(file, l0, l1), ...], transcript topic)
    (B, "Where does the API enforce that the person who authored a claim's evidence packet cannot also approve it (the four-eyes rule)?", [("api/src/claims/service.ts", 363, 376)], "four-eyes review"),
    (B, "Which function checks that the user accepting an imported document is not the one who uploaded or edited it, and in which file is it defined?", [("api/src/imports/review-service.ts", 62, 80)], "four-eyes review"),
    (B, "In which file are the allowed API key scopes (claims.read, exports.claims, imports.write) defined?", [("packages/shared/src/api-key-dto.ts", 11, 18)], "API keys step 4 review"),
    (B, "Where is it enforced that a tenant API key request takes its tenant only from the key row, and that CSRF checks are skipped only for key requests?", [("api/src/http/security.ts", 110, 127)], "API keys step 4 review"),
    (B, "Which function takes a SELECT ... FOR UPDATE row lock on a claim inside a tenant transaction?", [("api/src/db/tenant.ts", 195, 205)], "claims state machine review"),
    (B, "Where in the non-test API code is the Postgres session setting that scopes a transaction to a tenant (app.tenant_id) set?", [("api/src/db/tenant.ts", 220, 230)], "RLS tenant isolation"),
    (B, "Where is the transaction-local 'app.system' setting turned on for system-mode database access?", [("api/src/db/system.ts", 15, 25)], "RLS system mode"),
    (B, "Which migration creates the append-only trigger on the import_batches table?", [("api/prisma/migrations/20261009000000_import_export/migration.sql", 340, 350)], "import/export migration"),
    (B, "In which file is the SesMailer class, the not-yet-implemented production mail transport, defined?", [("api/src/auth/mailer.ts", 36, 50)], "security review F-01"),
    (B, "Where does production config validation reject MAIL_TRANSPORT=outbox?", [("api/src/config.ts", 525, 533)], "security review F-01"),
    (B, "Which file defines the LLM extraction provider that is only a NotImplemented stub?", [("api/src/imports/parse/provider.ts", 18, 30)], "import pipeline 'No LLM' claim"),
    (B, "Where are API key secrets hashed with HMAC-SHA-256 and a pepper?", [("api/src/apikeys/key-material.ts", 18, 32)], "API keys review"),
    (B, "Where does the PDF parser reject documents that have too many pages?", [("api/src/imports/parse/pdf.ts", 128, 132)], "import hardening"),
    (B, "Where is the allow-list of environment variables passed to the sandboxed import worker process defined?", [("api/src/imports/sandbox/spawn.ts", 28, 36)], "sandbox review"),
    (B, "Which file installs the synchronous module hooks (module.registerHooks) that block forbidden imports inside the import sandbox worker?", [("api/src/imports/sandbox/guard.ts", 60, 100)], "sandbox guard review"),
    (B, "Where is the double-submit CSRF token comparison (cookie versus X-CSRF-Token header) implemented?", [("api/src/security/csrf.ts", 1, 32)], "CSRF review"),
    (B, "Where is the append function of the hash-chained audit trail?", [("api/src/audit/audit.ts", 55, 75)], "audit chain"),
    (B, "In the web app, where is the Approve/Reject UI replaced by explanatory text when the current user authored the claim packet?", [("web/src/features/approvals/ApprovalsPage.tsx", 228, 240)], "four-eyes web UI"),
    (B, "Where does the import review queue page show the four-eyes notice instead of the Accept document button?", [("web/src/features/imports/ReviewQueuePage.tsx", 238, 248)], "four-eyes web UI"),
    (B, "Where is the fr_csrf cookie minted and set?", [("api/src/auth/cookies.ts", 30, 45)], "auth cookies"),
    (B, "Which file in the shared package defines the list of claim statuses (PENDING_REVIEW, APPROVED, ...)?", [("packages/shared/src/dto.ts", 12, 20)], "shared DTOs"),
    (B, "Where is it documented in code that web session state lives only in memory?", [("web/src/auth/session-store.ts", 1, 10)], "web session"),
    (B, "Which CI workflow job runs lint, typecheck, unit, migrate, seed, acceptance, build and audit for the TypeScript monorepo, and where is it declared?", [(".github/workflows/web-ci.yml", 55, 62)], "CI-only failure diagnosis"),
    (B, "Which migration creates the SECURITY DEFINER functions that support the four-eyes feature?", [("api/prisma/migrations/20261011000000_four_eyes/migration.sql", 1, 40)], "four-eyes migration"),
    (B, "Where is formula-injection neutralisation for exported CSV/XLSX cells implemented?", [("api/src/exports/neutralize.ts", 20, 40)], "export hardening"),
    (B, "Where is the maximum extracted-text size enforced in the import parse pipeline?", [("api/src/imports/parse/pipeline.ts", 28, 38), ("api/src/imports/parse/pdf.ts", 144, 150)], "import limits"),
    (B, "In which file is the permission matrix, the single source of truth for roles and their permissions, defined?", [("packages/shared/src/rbac.ts", 1, 80)], "RBAC"),
    (B, "Where does the API-key authentication code decide that a key is inactive because its creator was disabled or downgraded?", [("api/src/auth/api-key-auth.ts", 140, 160)], "API keys review G-e"),
    (B, "Where is the Terraform ECS setting that selects the mail transport?", [("infra/ecs.tf", 76, 84)], "infra review"),
    (B, "Which seed script decides how many demo claims get each status (PENDING_REVIEW, APPROVED, REJECTED, SEND_READY)?", [("api/scripts/seed.ts", 130, 140)], "seed data"),
    (B, "Where does the logger scrub fr_live_ API key strings from log output?", [("api/src/logging.ts", 1, 80)], "API keys logging"),
    (B, "Where is the check that rejects a request when the CSRF check fails for cookie-authenticated requests?", [("api/src/http/security.ts", 130, 145)], "CSRF review"),
    ("freight-recovery", "In the Python reference service, where is the INV-DUPLICATE invoice rule produced?", [("src/freight_recovery/rules/invoice_checks.py", 100, 120)], "python rules"),
    ("freight-recovery", "Where is the maximum API key length for the Python service's frk_ keys defined?", [("src/freight_recovery/keys.py", 20, 28)], "python keys"),
    ("freight-recovery", "Which function in the Python ingest loader finds document-type markers in a text?", [("src/freight_recovery/ingest/loader.py", 90, 100)], "python ingest"),
    ("freight-recovery", "Where does the Python settings class reject sandbox_max_workers below 1?", [("src/freight_recovery/config.py", 70, 80)], "python sandbox"),
    ("neural-mesh", "Which function in the ingest scrubber decides whether a tool call touches a sensitive path such as .env or secrets/?", [("ingest/scrub.py", 38, 44)], "scrubber review"),
    ("neural-mesh", "Where is the verified-cache hit rule used by the LLM front-end defined?", [("llm/frontend.py", 180, 215)], "front-end verified cache"),
    ("neural-mesh", "Where is KnnRouter::route_guarded defined, the guarded kNN tool router with tau and smin?", [("src/knnroute.rs", 70, 90)], "frontend tool route"),
    ("neural-mesh", "Where is the exact rational calculator entry point Calc::eval_str implemented?", [("src/tools.rs", 705, 720)], "tools executors"),
    ("neural-mesh", "Which Rust struct implements the BM25 index over a document store in the hybrid retrieval module?", [("src/hybrid.rs", 160, 180)], "hybrid retrieval"),
    ("neural-mesh", "Where is the constructor that builds a hybrid index with custom BM25 k1 and b parameters?", [("src/hybrid.rs", 320, 335)], "R54 BM25 profile"),
    ("neural-mesh", "Where is the minimum decoy-rejection rate (0.70) required by the front-end evaluation gates defined?", [("llm/frontend_eval.py", 35, 40)], "judge gate"),
    ("neural-mesh", "Which Python test file checks the front-end hook's forward-by-default and out-of-distribution behaviour?", [("llm/tests/test_pipeline.py", 126, 141)], "frontend hook tests"),
    ("neural-mesh", "Where is the pinned version of the ruflo CLI used by the memory mirror defined?", [("ingest/mirror.py", 34, 40)], "mirror review"),
    ("neural-mesh", "In the recall hook, where is the regular expression for detecting file-path-like tokens in a prompt?", [("hooks/recall_hook.py", 40, 46)], "recall hook gate"),
    ("neural-mesh", "Where is the function that scores whether a prompt is about code in the recall hook gate?", [("hooks/recall_hook.py", 65, 90)], "recall hook gate"),
]


def main():
    out = []
    for i, (repo, q, ans, topic) in enumerate(Q):
        out.append({"id": f"q{i+1:02d}", "repo": repo, "cwd": REPOS[repo], "question": q,
                    "answers": [{"file": f, "l0": a, "l1": b} for f, a, b in ans], "topic": topic, "split": "tune" if i % 3 == 0 else "held"})
    here = pathlib.Path(__file__).parent
    (here / "questions.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n", encoding="utf8")
    print(len(out), "questions;", sum(o["split"] == "tune" for o in out), "tune")
    if "--check" in sys.argv:
        for o in out:
            for a in o["answers"]:
                p = pathlib.Path(REPOS[o["repo"]]) / a["file"]
                n = len(p.read_text(encoding="utf8", errors="replace").split("\n")) if p.exists() else -1
                print(("OK  " if n >= a["l1"] else "BAD ") + o["id"], a["file"], a["l0"], a["l1"], "of", n)


main()
