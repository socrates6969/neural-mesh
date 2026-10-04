"""f6: embed a HELD-OUT paraphrase set with the local all-MiniLM-L6-v2 ONNX model and time the offline embedder.

Same hermetic recipe as py/f5_embed.py (audited in the f5 board block): imports only json/os/random/time/
numpy/onnxruntime/tokenizers/collections, reads only the two local model files and results/f5_corpus.json +
results/f5_embeddings.f32, opens no sockets, writes only results/f6_para_fixture.json and results/f6_py_latency.json.
The text is SYNTHETIC, written by the f6 worker (new subject phrasings + new question templates + typo variants of
f5 texts), NOT real user text. Fixture format = embedding.rs TableEmbedding JSON ({"name","dim","vectors":{text:[...]}})
plus an extra "meta" key (ignored by the loader) with the labels.
"""
import json, os, random, time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = os.path.expanduser("~/.ruvector/models/all-MiniLM-L6-v2")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results")

NEW_SUBJ = [
    ["evening espresso and insomnia", "does a late cup of coffee ruin a night's rest", "caffeine effects on sleep latency", "sleeplessness linked to coffee drinking", "energy drinks before bed and sleep duration"],
    ["gym routines and low mood", "does jogging reduce depression", "aerobic training for mental health", "being active and feeling less sad", "exercise as treatment for anxiety and depression"],
    ["candy and kids' ability to focus", "does sugar make children hyper", "high sugar diets and classroom behaviour", "sweetened drinks and short attention spans in young children", "refined carbohydrates and ADHD symptoms in kids"],
    ["mini-batch size versus generalization gap", "small batches generalize better than large batches in deep learning", "the effect of batch size on test error", "stochastic gradient noise from batch size and sharp minima", "why big batch training hurts validation accuracy"],
    ["augmenting training images to cope with covariate shift", "random crops and flips and model robustness", "does augmentation improve out-of-distribution performance", "robust classifiers trained with transformed data", "synthetic perturbations during training and corruption robustness"],
    ["pretrained language models for under-resourced languages", "machine translation when parallel data is scarce", "cross-lingual transfer for low-resource translation", "multilingual pretraining helping rare languages", "few-shot translation with big pretrained models"],
    ["residential solar panels and the electric grid", "rooftop PV penetration and voltage stability", "household solar causing grid instability", "net metering and distribution network reliability", "intermittent solar power and grid balancing"],
    ["arctic permafrost melting and CH4 release", "thawing frozen ground emits methane", "methane from warming tundra soils", "permafrost carbon feedback on climate", "greenhouse gases escaping from thawing northern soils"],
    ["coral reefs under ocean acidification", "lower ocean pH harming reef-building corals", "carbonate chemistry and coral skeleton growth", "CO2-driven acidification and bleaching of reefs", "do corals calcify slower in acidic seawater"],
    ["effects of a higher minimum wage on jobs", "raising the minimum wage and unemployment", "wage floors and low-wage employment outcomes", "does a mandated pay increase cost workers their jobs", "minimum wage hikes and hours worked"],
    ["city centre congestion charges and smog", "does charging drivers in downtown cut air pollution", "low emission zones and urban air quality", "road pricing schemes and nitrogen dioxide levels", "tolling vehicles entering the city and cleaner air"],
    ["telework and the cost of homes", "work-from-home trends driving property prices", "hybrid working and housing demand in suburbs", "remote jobs and rents in expensive cities", "pandemic remote work reshaping real estate markets"],
]
NEW_TEMPL = [
    "tl;dr on {s}",
    "{s}: what does the literature say?",
    "any recent evidence regarding {s}?",
    "help me understand {s}",
    "pros, cons and uncertainties of {s}",
]


def typo(text, rng):
    ws = text.split(" ")
    idx = [i for i, w in enumerate(ws) if len(w) >= 5]
    rng.shuffle(idx)
    for i in idx[:2]:
        w = ws[i]
        j = rng.randrange(1, len(w) - 2)
        if rng.random() < 0.5:
            w = w[:j] + w[j + 1] + w[j] + w[j + 2:]  # swap adjacent letters
        else:
            w = w[:j] + w[j + 1:]  # drop a letter
        ws[i] = w
    return " ".join(ws)


def build(f5):
    items = []
    seen = set(it["text"] for it in f5["items"])
    rng = random.Random(606)
    qs = {}
    for it in f5["items"]:
        if it["kind"] == "q":
            qs.setdefault(it["concept"], []).append(it["text"])

    def add(text, c, kind):
        if text in seen:
            return
        seen.add(text)
        items.append(dict(text=text, concept=c, topic=f5["concepts"][c]["topic"], kind=kind))

    for c, subs in enumerate(NEW_SUBJ):
        for s in subs:
            add(s, c, "newsubj")
        for s in subs:
            for t in NEW_TEMPL:
                add(t.format(s=s), c, "newq")
        for q in rng.sample(qs[c], 4):
            add(typo(q, rng), c, "typo")
    return items


def make_session():
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"), so, providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(max_length=128)
    return sess, tok


def embed(sess, tok, texts):
    names = [i.name for i in sess.get_inputs()]
    enc = tok.encode_batch(texts)
    L = max(len(e.ids) for e in enc)
    ids = np.zeros((len(texts), L), dtype=np.int64)
    am = np.zeros((len(texts), L), dtype=np.int64)
    for r, e in enumerate(enc):
        ids[r, :len(e.ids)] = e.ids
        am[r, :len(e.ids)] = 1
    feed = {"input_ids": ids, "attention_mask": am}
    if "token_type_ids" in names:
        feed["token_type_ids"] = np.zeros_like(ids)
    hs = sess.run(None, feed)[0]
    m = am[:, :, None].astype(np.float32)
    p = (hs * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
    p /= np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None)
    return p.astype(np.float32)


def fmt(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def main():
    f5 = json.load(open(os.path.join(OUT, "f5_corpus.json"), encoding="utf8"))
    items = build(f5)
    sess, tok = make_session()
    texts = [it["text"] for it in items]
    embed(sess, tok, texts[:4])  # warm-up
    t0 = time.perf_counter()
    vecs = np.concatenate([embed(sess, tok, texts[i:i + 32]) for i in range(0, len(texts), 32)])
    batch_ms = (time.perf_counter() - t0) * 1000 / len(texts)
    sample = texts[:200]
    t0 = time.perf_counter()
    for s in sample:
        embed(sess, tok, [s])
    single_ms = (time.perf_counter() - t0) * 1000 / len(sample)
    q = [it["text"] for it in f5["items"] if it["kind"] == "q"][:200]
    t0 = time.perf_counter()
    for s in q:
        embed(sess, tok, [s])
    single_q_ms = (time.perf_counter() - t0) * 1000 / len(q)
    body = ",".join(json.dumps(t, ensure_ascii=False) + ":" + fmt(v) for t, v in zip(texts, vecs))
    meta = json.dumps(dict(note="SYNTHETIC held-out paraphrase set written by the f6 worker; labels only",
                           items=[dict(text=it["text"], concept=it["concept"], topic=it["topic"], kind=it["kind"]) for it in items]), ensure_ascii=False)
    with open(os.path.join(OUT, "f6_para_fixture.json"), "w", encoding="utf8") as f:
        f.write('{"name":"minilm-l6-f6-para","dim":384,"vectors":{' + body + '},"meta":' + meta + "}")
    lat = dict(n_texts=len(texts), batch32_ms_per_text=batch_ms, single_ms_per_text_new=single_ms, single_ms_per_text_f5_questions=single_q_ms,
               onnxruntime=ort.__version__, intra_op_threads=2, note="end-to-end python embed incl. tokenisation and mean-pool, CPU, machine possibly loaded")
    json.dump(lat, open(os.path.join(OUT, "f6_py_latency.json"), "w"), indent=1)
    print(json.dumps(lat, indent=1))
    from collections import Counter
    print("para items", len(items), Counter(it["kind"] for it in items))
    emb5 = np.fromfile(os.path.join(OUT, "f5_embeddings.f32"), dtype=np.float32).reshape(-1, 384)
    cent = np.zeros((12, 384), dtype=np.float32)
    for i, it in enumerate(f5["items"]):
        if it["kind"] in ("q", "subj"):
            cent[it["concept"]] += emb5[i]
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    sims = vecs @ cent.T
    acc = {}
    for it, row in zip(items, sims):
        acc.setdefault(it["kind"], []).append(int(np.argmax(row) == it["concept"]))
    print("nearest-f5-centroid accuracy by kind:", {k: round(float(np.mean(v)), 3) for k, v in acc.items()})


if __name__ == "__main__":
    main()
