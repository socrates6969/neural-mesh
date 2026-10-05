c='src/bin/meshstore.rs'
t=open(c,encoding='utf-8').read()
t=t.replace("//!   meshstore sync-docs <root> <shared-folder> <machine-id> <ns>   shared-folder CORPUS sync for one namespace (B15; int8 vectors, own shard replaced atomically)\n//!   meshstore gc-docs <root> <ns> <cos-threshold>      drop near-duplicate corpus docs (R47)",
"//!   meshstore sync-docs <root> <shared-folder> <machine-id> <ns> [--propagate-deletions] [--ttl-days N] [--deleters m1,m2]\n//!                                                      shared-folder CORPUS sync for one namespace (B15; int8 vectors, own shard replaced atomically).\n//!                                                      --propagate-deletions exchanges gc deletions (B38, keeper-safe); --ttl-days expires tombstones (B40);\n//!                                                      --deleters limits whose deletions are applied (B40; machine-id based, not cryptographic)\n//!   meshstore gc-docs <root> <ns> <cos-threshold> [--canonical]   drop near-duplicate corpus docs (R47); --canonical keeps the smallest doc id of each class so machines converge (B42)")
open(c,'w',encoding='utf-8').write(t)
p='RD-LOG-B.md'
s=open(p,encoding='utf-8').read()
i=s.index('## Deferred')
add='''## Round B43 - the import in `sync_docs_folder_with` checks tombstones under the store lock (`put_docs_untombstoned`; test `import_skips_ids_tombstoned_after_the_snapshot`)
Closes the residual listed under Deferred: tombstones were read once before the import, so a gc that tombstoned an id between that read and the write could be undone by the import. The skip is now decided inside `put_docs`' lock. Deterministic test: tombstone appears after the importer's decision, the tombstoned id is not written (old `put_docs` writes it, n would be 2). Explicit `put_docs` of a tombstoned id still works. Verdict: **KEPT (race fix)**.
## Round B44 - fuzz of the capped journal sync (`b44_tests::capped_sync_converges_to_newest_k_of_the_union`)
3 machines x 60 seeds x 30 random append/sync ops with a shared cap K=3 over 2 worlds, then a quiescent tail: every store ends with exactly the newest-K-per-world of the union of all appended outcomes (the documented convergence claim). **Holds 60/60; no bug found** (verified invariant, not a fix).
## Round B45 - CLI end-to-end of the new flags (manual, binary run)
`gc-docs --canonical`, `sync-docs --propagate-deletions --ttl-days 30 --deleters mZ|mA` run through the built binary on two stores: unauthorised peer -> `deletions_unauthorized:1`, deleted:0; authorised -> `deleted:1`. Not an automated test.
## Round B46 - `meshstore sync-docs` flag parsing returns a usage error (exit 2) instead of panicking on an unknown flag / missing value (checked by running the binary). **KEPT (CLI hardening).**
## Round B47 - concurrent gc vs sync stress (`b47_tests`)
Two threads (40 gc + 40 sync iterations x 15 rounds) on one store; asserts no tombstoned doc is ever present and vectors stay aligned: passes. HONEST NEGATIVE ON DISCRIMINATION: with the B43 fix reverted (plain `put_docs`) the stress still passes in 3/3 runs, i.e. the window is too small for a stress test to see; B43's deterministic test is the real proof, B47 is only a smoke test.
## Round B48 - the fuzz with tombstone expiry (`expiry_never_loses_a_class_and_canonical_gc_still_converges`)
TTL 2 days, clock advancing 0-2 days per op, 60 seeds, canonical gc: no class lost, vectors aligned, identical id sets after the maintenance tail (0/60 diverging). Verified invariant (expiry only re-opens re-import; the keeper rule still bounds deletion).
## Round B49 - `meshstore` header documents the new flags. (Docs only.)
## Round B50 - wrap-up: target of ~50 rounds reached. Code-bearing wins B40, B42, B43 (+B46 CLI); B41/B44/B48 are verified invariants with mutation-checked power for B41 (and B44/B48 oracle-based); B47 non-discriminating smoke test, stated above. No retrieval/accuracy claim was made in B40-B50 (none needed).
'''
s=s[:i]+add+s[i:]
s=s.replace("Still open: cryptographic identity for deleters; race between a concurrent gc and an in-flight import (tombstones are read once, before the import).)","Race closed in B43. Still open: cryptographic identity for deleters.)")
open(p,'w',encoding='utf-8').write(s)
