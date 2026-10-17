### VERIFY REQUEST — W1 — G1 EMBED-BAKEOFF: BGE-base vs MiniLM vs hash vs BM25 on a HARD synthetic corpus + retrieval-system levers + downstream routing

RESULT (plain, no "10x"/novel-knowledge claims; everything below is a measured number on a SYNTHETIC corpus I generated):
 (1) BGE-base earns a SMALL retrieval edge over MiniLM (+0.020 MRR / +0.026 hit@1 held-out; +0.023 / +0.035 on a fresh corpus; Holm wins both) at 5.4x encode latency, 4.8x model size, 2x vector memory. It does NOT earn it downstream: in knowledge-routing success BGE-vs-MiniLM has 0 Holm wins in either split (sib16 H2000: -0.012 held-out, -0.007 confirm; rand16: +0.007 / +0.001).
 (2) System-level work on MiniLM roughly TIES BGE-alone on retrieval at ~1/5 the encode cost (mini-final vs bge-alone: hit@1 +0.011 Holm-win held-out but +0.005 n.s. on the fresh corpus; MRR +0.003 / +0.004, n.s. after Holm) and is NOT a robust win. The same levers also lift BGE, so BGE-with-system beats MiniLM-with-system (+0.010 / +0.020 MRR, Holm wins). Downstream, MiniLM+hybrid beats BGE-alone (+0.025* held-out sib16, +0.014 confirm n.s. after Holm), i.e. directionally "system > bigger model" for routing, small, not replicated at Holm level.
 Almost all of the system gain is ONE lever (spell-correction, i.e. the 25% typo queries I put in the corpus); hybrid-BM25 / overlap-rerank add ~+0.01 each; chunk max-pool and Rocchio PRF are neutral-to-negative.

SOURCE / SECURITY: NO new download, install, crate, or model. Only models already on disk and already audited in earlier rounds: py/models/bge-base (BAAI/bge-base-en-v1.5 ONNX, sha256 9bc579ac...b91848 per results/f8_audit.json) and ~/.ruvector/models/all-MiniLM-L6-v2 (f5), run by the existing cognee-venv onnxruntime 1.23.2 / tokenizers / numpy via py/g1_embed.py (imports f8_embed.Embedder; no sockets, no network). A BIGGER model was searched for on disk (HF cache, ~/.cache, ~/.ruvector, ~/.ollama, cognee venv): none exists, so none tested. Rust has no ML runtime: it reads results/g1_emb_*.f32 fixtures. Cargo.toml/lib.rs/src untouched; new files only: src/bin/g1_retrieval.rs, src/bin/g1_route.rs, src/bin/g1_common/mod.rs, py/g1_corpus.py, py/g1_embed.py, results/g1_*. No git commits. Models run sequentially (python working set 290 MB MiniLM, 730 MB BGE).

CORPUS (py/g1_corpus.py, deterministic, seeded, no network; A = seed 1, B = seed 2 with shuffled surface-form roles, different docs/queries/typos = the "fresh" confirm corpus): 240 concepts = 30 procedure topics x 8 SIBLING details (environment/DB/cloud/language/OS/payment provider/CI/broker/platform/format/region; siblings differ in exactly one key detail). Each detail has 3 surface forms (e.g. staging / pre-production / pre-prod); each concept has 2 five-sentence docs using different forms and 4 procedure paraphrases; 480 concept docs + 300 off-topic docs. 8 queries per concept (3 clean paraphrases, 2 typo variants, 2 terse keyword queries, 1 chatty with irrelevant preamble) = 1920 + 200 off-topic queries. Two deterministic query rewrites from doc vocabulary only: spell-corrected (ed-1 snap to doc vocab + words recurring >=3x in the query log), keyword-extracted. Sample: "Looking for instructions on how to swap out the accesss keys used in pre-productino" -> concept "rotate credentials in staging". Tuning split = concepts with id%5==0 (48 concepts, 384 queries); held-out = the other 192 concepts (1536 queries).

RERUN (deterministic: reruns are byte-identical except the us_* / scan_us timing fields; I found and fixed one HashMap-order f32 ulp bug in tf-idf norm, so use the current files):
  export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"; export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r1 RAYON_NUM_THREADS=3; cd /c/Users/mariu/neural-mesh; PY=C:/Users/mariu/.cognee-plugin/venv/Scripts/python.exe
  $PY py/g1_corpus.py 1 results/g1_corpus_A.json ; $PY py/g1_corpus.py 2 results/g1_corpus_B.json          # <1 s each
  $PY py/g1_embed.py A ; $PY py/g1_embed.py B                                                              # ~3.5 min each (BGE 110 s of it) -> results/g1_emb_{A,B}_{mini,bge,bgei}.f32 (~55 MB per corpus; regenerable, gitignore if repo size matters), g1_embmeta_*.json
  cargo run --release --offline --bin g1_retrieval -- --corpus A                  # tunes levers on tune split, held-out seeds 31-60 -> results/g1_retrieval.json, g1_retrieval_picks.json   (~25 s)
  cargo run --release --offline --bin g1_retrieval -- --corpus B --confirm        # frozen picks, fresh corpus, seeds 61-90 -> results/g1_retrieval_confirm.json
  cargo run --release --offline --bin g1_route -- --corpus A                      # tunes (variant,k,thr,min_n) per system on seeds 1-8, held 31-60 -> results/g1_route.json, g1_route_picks.json (~40 s)
  cargo run --release --offline --bin g1_route -- --corpus B --confirm            # frozen picks, fresh corpus, seeds 61-90 -> results/g1_route_confirm.json
  (logs: results/g1_retrieval.log, g1_retrieval_confirm.log, g1_route.log, g1_route_confirm.log, g1_embed_{A,B}.log)

PROTOCOL: per-seed = random subset of 300 queries of the split (retrieval) / a fresh world+task stream (routing); levers tuned only on tune split (retrieval: greedy forward ladder, accept a lever if tune MRR +>=.003, plus each lever tuned alone) or seeds 1-8 (routing); held-out 31-60; frozen-pick confirm 61-90 on corpus B; paired 95% t-CI over 30 seeds; Holm over the family (retrieval 52 tests, routing 192 vs-baseline + 170 head-to-head). The retrieval CI is over resampled query subsets of a finite pool, NOT over independent corpora (A->B replication is the independent check).

NUMBERS 1 — RETRIEVAL (query -> its concept's 2 docs among 780; relevant=2; seeds mean; hit1 = P@1; P3/R3/F1 at k=3; sib@1 = top-1 is a SIBLING concept doc)
 held-out concepts, corpus A, seeds 31-60 | confirm corpus B, seeds 61-90 (frozen picks):
 system                  hit1 A/B       MRR A/B        P@3 A/B       R@3 A/B       F1@3 A/B      sib@1 A/B
 hash (word hash 384)    .431/.427      .562/.560      .225/.220     .337/.330     .270/.264     .339/.336
 BM25 raw                .823/.795      .873/.854      .325/.323     .488/.484     .390/.387     .154/.174
 MiniLM alone            .899/.881      .934/.924      .511/.507     .766/.761     .613/.609     .091/.099
 BGE alone (no instr)    .925/.916      .953/.946      .496/.492     .743/.738     .595/.591     .065/.071
 BGE + query instruction .934/.931      .958/.955      .478/.479     .717/.718     .574/.575     .057/.057
 MiniLM + spell (rewrite).920/.910      .950/.944      .523/.517     .785/.775     .628/.620     .074/.078
 MiniLM + BM25 hybrid    .921/.899      .946/.935      .427/.432     .640/.648     .512/.518     .076/.086
 MiniLM + overlap rerank .916/.898      .943/.933      .429/.426     .644/.638     .515/.511     .079/.089
 MiniLM-final (spell+overlap rerank g1) .936/.921 .957/.950  .432/.430 .648/.645 .518/.516 .062/.072
 BGE-final (spell+overlap g.5) .946/.949 .967/.969  .450/.453     .675/.679     .540/.544     .052/.048
 BGE+instr final         .951/.949      .970/.970      .482/.487     .723/.730     .578/.584     .046/.046
 MiniLM-final + BGE cascade (rerank top20) .953/.954 .970/.972 .482/.483 .723/.725 .578/.580 .046/.043
 Tuned picks (tune split only): mini ladder = spell, then overlap(g=1); hybrid/chunk/PRF not accepted after those. bge ladder = spell, overlap(g=.5). hybrid alpha .7 (mini) / .8 (bge); chunk lam .7; PRF beta .6 (mini) / .3 (bge).
 Paired (held-out A | confirm B; MRR diff [95% CI], all 30/30-seed wins unless noted; * = Holm win):
  BGE - MiniLM (alone):                +.0197 [+.0159,+.0236]* | +.0225 [+.0191,+.0259]*   (hit1 +.0262* | +.0351*)
  BGE+instr - MiniLM:                  +.0244* | +.0311*;  BGE+instr - BGE: +.0047* | +.0086* (instruction helps a little)
  MiniLM - BM25:                       +.0605* | +.0697*;  MiniLM - hash: +.372* | +.364*;  BGE - BM25 +.080* | +.092*
  levers on MiniLM (vs alone): chunk +.0050* | +.0027*; spell +.0158* | +.0203*; hybrid +.0123* | +.0113*; PRF -.0092! | -.0104!; overlap +.0093* | +.0097*; final +.0230* | +.0262*
  levers on BGE (vs alone):    chunk -.0128! | -.0107!; spell +.0152* | +.0192*; hybrid -.0010 n.s. | +.0048*; PRF -.0027! | -.0015!; overlap +.0015 n.s. | +.0053*; final +.0134* | +.0233*
  MiniLM-final - BGE-alone:            MRR +.0033 [+.0002,+.0063] n.s. | +.0037 [+.0001,+.0073] n.s.;  hit1 +.0111* | +.0048 n.s.  (TIE, not a robust win)
  MiniLM+hybrid - BGE-alone:           MRR -.0074! | -.0112!  (hybrid alone does not close the gap)
  BGE-final - MiniLM-final:            MRR +.0102* | +.0196*;  hit1 +.0101* | +.0277*   (BGE keeps its edge after the same system work)
  cascade - BGE-alone:                 MRR +.0167* | +.0261*  (but cascade needs BOTH encoders: cost >= BGE alone)
 Per query class, hit1 (corpus B confirm): clean/chatty/terse/typo -> MiniLM alone .934/.880/.927/.766; MiniLM-final .929/.844/.932/.938; BGE alone .955/.932/.964/.802; BGE-final .955/.896/.974/.943; BM25 .849/.771/.852/.674. => the whole system gain is the typo class (spell-correct); BGE's own edge is on clean/chatty/terse (+2..5 pts); overlap-rerank HURTS chatty queries (-3.6 pts for MiniLM).
 Negatives: hybrid/overlap raise top-1 but LOWER recall of the 2nd (differently-phrased) relevant doc: R@3 .766 -> .640 (MiniLM+hybrid), F1@3 .613 -> .512; sentence-chunk max-pool is neutral on MiniLM, -1.3 pts MRR on BGE; Rocchio PRF always <= 0 (it pulls toward sibling docs).

NUMBERS 2 — DOWNSTREAM KNOWLEDGE-ROUTING (g1_route; expected success averaged over prefix H; world d6-style: hidden best agent per CONCEPT, w=.8, siblings have independent best agents; outcome feedback only; policy = crate-faithful kn-direct+vote (anchor vote -> kn:<anchor> feature into guarded(hebr), vote min_n/tau .7, gate 6/.02/25) with a PLUGGABLE similarity; per-system (variant,k,thr,min_n) tuned on seeds 1-8 in sib16, mean(H500,H2000); oracle arm = ground-truth concept equality)
 H2000 means (held-out 31-60 | confirm 61-90):   sib16 | rand16 | sib64 (64 concepts) | sib16-mixed50 | ctl-irrel(w=0)
  best fair embedding-free baseline          .4662 guarded | .4654 keyword | .4545 guarded | .6075 guarded | .7673 warm
  hash                                       .516|.518  .515|.513  .467|.465  .599|.603  .746|.749
  tf-idf (lexical cosine)                    .529|.530  .538|.537  .471|.472  .599|.600  .745|.745
  MiniLM                                     .530|.530  .541|.538  .471|.469  .601|.601  .746|.748
  BGE-base                                   .518|.523  .547|.539  .472|.472  .601|.601  .743|.745
  MiniLM+spell                               .539|.535  .542|.533  .476|.473  .601|.603  .745|.747
  MiniLM-hybrid (.6 dense + .4 tf-idf)       .543|.537  .546|.543  .478|.476  .599|.599  .744|.746
  BGE-hybrid                                 .539|.536  .541|.534  .476|.473  .599|.603  .746|.747
  cascade (mini-hybrid -> BGE rerank)        .543|.534  .543|.531  .476|.475  .599|.602  .748|.747
  oracle (perfect retrieval)                 .632|.638  .631|.636  .542|.542  .651|.649  .738|.740
 Paired vs best fair baseline at H2000: sib16 hash +.050*, MiniLM +.064*, BGE +.052*, MiniLM-hybrid +.077* (confirm +.051/+.062/+.055/+.070, all *); rand16 hash +.049*, MiniLM +.075*, BGE +.082* (confirm +.046/+.071/+.072, *). ORACLE ceiling is +.166 (sib16): real retrievers capture ~40-45% of the available gain even on the hard corpus.
 Head-to-head H2000 (diff [CI], Holm p) held-out | confirm:
  BGE - MiniLM:         sib16 -.0123 [-.0196,-.0049] p=.28 | -.0071 p=1.0;   rand16 +.0065 p=.78 | +.0006;   sib64 +.0007 | +.0028.   => 0 Holm wins/0 Holm losses in either split. BGE does not beat MiniLM in routing.
  MiniLM - hash:        sib16 +.0143* | +.0116 (p=.18);  rand16 +.0259* | +.0250*   (semantic vs bag-of-words: real but ~.015-.026)
  MiniLM - tf-idf:      sib16 +.0019 | ~0;  rand16 +.0025 | +.0012   (n.s.)  => on this corpus a lexical idf cosine routes as well as MiniLM (surface forms are shared across paraphrases; real user text would share fewer words)
  MiniLM-hybrid - BGE:  sib16 +.0250 [+.0189,+.0311]* | +.0142 p=.24;  rand16 -.0015 | +.0040;  sib64 +.0060 p=.11 | +.0041
  MiniLM-hybrid - MiniLM: sib16 +.0127* | +.0071 n.s.;  rand16 +.0050 | +.0047 n.s.;  sib64 +.0067 | +.0069*(confirm)
  BGE-hybrid - BGE:     sib16 +.0207* | +.0131*;  rand16 -.0063 | -.0049
  BGE-hybrid - MiniLM-hybrid: sib16 -.0043 | -.0012; rand16 -.0048 | -.0089 (BGE still not better after the same system)
  oracle - BGE-hybrid:  +.094 / +.090 / +.066 (sib16/rand16/sib64) -> retrieval quality, not model size, is the headroom.
 CONTROLS (not free): ctl-irrel (context carries no info): every similarity incl. the oracle LOSES to warm: hash -.021!, MiniLM -.022!, BGE -.025!, MiniLM-hybrid -.024!, oracle -.029! (Holm losses, same in confirm). sib16-mixed50 (half the tasks unrelated one-offs): -.007..-.008! held-out (n.s. in confirm, -.002..-.006). sib64 (64 concepts, ~31 tasks each): gains shrink to +.012..+.024 vs guarded(hebr) at H2000 (oracle +.088): sample-efficiency, not embedding quality, is the bottleneck there. Early horizons: H100/H250 are n.s. (-.004..+.002 vs keyword) for every similarity; at H500 sib16 MiniLM/BGE are +.008/+.010 vs keyword (n.s.), rand16 +.013/+.017 (Holm *); the layer pays after ~500-1000 tasks.

NUMBERS 3 — COST (this box, shared, 4 intra-op threads; python onnxruntime; scoring on 1 Rust thread, 780 docs / 3600 sentence chunks)
 encode one text: MiniLM 2.44 ms (p95 3.9) | BGE 13.25 ms (p95 20.9) [corpus B: 3.18 / 14.45] = 5.4x; hash ~µs; tf-idf/BM25 ~4 µs (BM25 scoring of 780 docs); spell-correct 0.3 ms/typo-query (python ed-1 lookup, not optimised, not in Rust).
 model on disk: MiniLM 86.2 MiB | BGE 415.6 MiB (4.8x); python working set 290 vs 730 MB.
 vectors: docs 780x dim: 1.1 MB vs 2.3 MB; sentence chunks 3600x dim: 5.3 MB vs 10.5 MB; BM25 index 0.18 MB; knowledge store 2000 tasks: 2.9 MB vs 5.9 MB.
 scoring per query (1 thread, vector already encoded): dense doc 207 µs (MiniLM) vs 390 µs (BGE); sentence-chunk max-pool 1.0 ms vs 1.9 ms; MiniLM dense+BM25+overlap-rerank(20) 218 µs; Rocchio two-pass 413 / 791 µs. Routing neighbour scan over 2000 stored tasks: 0.49 ms (MiniLM) vs 0.96 ms (BGE) per task.
 => end-to-end query->top-1 latency: MiniLM-final ~2.4 ms encode + 0.2 ms score + 0.3 ms spell = ~3 ms; BGE-alone ~13.3 + 0.4 = ~13.7 ms; BGE-final ~14 ms; cascade ~16 ms (both encoders).

NEGATIVES / CAVEATS (blunt):
 1. Everything is on a SYNTHETIC, templated corpus I wrote (30 topics x 8 details, 3 surface forms per detail, filler sentences shared across all docs). Surface forms of a detail are limited to 3, so lexical methods are unrealistically strong (BM25 hit@1 .82) and the semantic gap is understated; real text may favour BGE/MiniLM more. It says nothing about real user text.
 2. The system gain is dominated by spell-correction on a typo class that I set to 25% of queries; at 0% typos the "MiniLM-with-system ~ BGE-alone" result would disappear (MiniLM-final vs MiniLM-alone on clean queries is ~0 or negative: .934 -> .929). Overlap rerank hurts chatty queries. The spell-corrector's vocabulary includes words recurring >=3x in the (eval) query log (mildly transductive).
 3. Retrieval CIs are over resampled query subsets of one finite pool (not independent corpora); the A->B replication is the real check and it agrees on all signs; magnitudes move by up to .01 (e.g. mini-final vs bge-alone hit@1 .011* -> .005 n.s.).
 4. Routing policy is a self-contained replica of knowledge::Knowledge/KnowledgeRouter semantics (not the crate's code) so rerank/hybrid/cascade similarities can be plugged in; the dense-only arms were NOT cross-checked against the crate (the f8 harness used 12 concepts; here 16-64). Regime W (weak feedback) not run. Tuning used one world (sib16); hash picks a high threshold (.92), i.e. ~exact-match behaviour. Thresholds/sim scales differ per system, each system is tuned separately.
 5. With 64 concepts the knowledge layer gains only ~+.01..+.02 over guarded(hebr): the layer is sample-hungry (needs ~min_n rewarded tries per concept anchor).
 6. The knowledge layer still loses on controls (ctl-irrel -.02, mixed50 -.007) for EVERY embedder including the oracle: unchanged finding from d6/f8, not an embedder issue.
 7. BGE used CLS pooling, fp32, max_len 128; the instruction-prefixed variant is the only prompt variant tried. No int8/quantised variant, no bigger model on disk, no Rust inference path. Latency is python onnxruntime on a shared box.
 8. The cascade (MiniLM -> BGE rerank) is the best retrieval system (hit@1 .953/.954) but needs both encoders per query, i.e. costs more than BGE alone; it is not a cost-saving answer to question 2.
 9. py/models/bge-base (416 MiB) and results/g1_emb_*.f32 (~110 MB for A+B) are large artefacts outside src; delete/gitignore as desired (regenerable).

RECOMMENDATION: keep MiniLM as the default embedder; add spell-correction (cheap, model-independent, +.016..+.026 MRR on typo-heavy input) and, if routing matters, a hybrid dense+tf-idf similarity (+.01 routing at no extra encode cost). BGE-base is only justified for retrieval-quality-critical use where +2-3 pts hit@1 matters more than 5x latency/4.8x size; it does not improve the knowledge-routing outcome measured here. Re-test on real (non-synthetic) text before any change of default.

FILES: py/g1_corpus.py, py/g1_embed.py, src/bin/g1_common/mod.rs, src/bin/g1_retrieval.rs, src/bin/g1_route.rs; results/g1_corpus_{A,B}.json, g1_texts_*.json, g1_qtexts_*.json, g1_emb_{A,B}_{mini,bge,bgei}.f32, g1_embmeta_{A,B}.json, g1_embed_{A,B}.log, g1_retrieval{,_confirm}.{json,log}, g1_retrieval_picks.json, g1_route{,_confirm}.{json,log}, g1_route_picks.json, g1_board_entry.md. Unchanged: Cargo.toml, src/lib.rs, rest of src/.
