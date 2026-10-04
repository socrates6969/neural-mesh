"""f5: build a SYNTHETIC research-question corpus and embed it with the local all-MiniLM-L6-v2 ONNX model.

Security/hermetic: imports only json/hashlib/random/numpy/onnxruntime/tokenizers; reads only the two local
model files; no network, no downloads, no pip. Writes results/f5_corpus.json + results/f5_embeddings.f32.
The text is SYNTHETIC (templated by me), not a real user corpus.
"""
import hashlib, json, os, random, sys, time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = os.path.expanduser("~/.ruvector/models/all-MiniLM-L6-v2")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

TOPICS = ["health", "ml", "climate", "econ"]
CONCEPTS = [  # (topic, name, [3 subject phrasings])
    (0, "caffeine-sleep", ["caffeine intake and sleep quality", "drinking coffee late in the day and how well people sleep", "stimulant consumption and nighttime rest"]),
    (0, "exercise-mood", ["regular exercise and depressive symptoms", "physical activity and mental wellbeing", "working out and mood in adults"]),
    (0, "sugar-attention", ["sugar consumption and attention in children", "sweet snacks and concentration at school", "dietary sucrose and childhood hyperactivity"]),
    (1, "batchsize-generalisation", ["batch size and generalisation in neural networks", "large minibatches and test accuracy of deep models", "how training batch size affects overfitting"]),
    (1, "augmentation-robustness", ["data augmentation and robustness to distribution shift", "image transformations during training and model robustness", "synthetic training variations and out-of-distribution accuracy"]),
    (1, "pretraining-lowresource", ["pretraining and low-resource language translation", "large language model pretraining for languages with little data", "transfer learning for scarce-language text tasks"]),
    (2, "solar-grid", ["rooftop solar adoption and grid stability", "household photovoltaic panels and electricity network reliability", "distributed solar generation and power grid balance"]),
    (2, "permafrost-methane", ["permafrost thaw and methane emissions", "melting frozen arctic soil and greenhouse gas release", "thawing tundra and atmospheric methane"]),
    (2, "acidification-coral", ["ocean acidification and coral reef growth", "falling seawater pH and coral calcification", "carbon dioxide absorption by oceans and reef health"]),
    (3, "minwage-employment", ["minimum wage increases and employment levels", "raising the wage floor and job losses", "statutory pay minimums and hiring in low-wage sectors"]),
    (3, "congestion-air", ["congestion pricing and urban air quality", "charging drivers to enter city centres and pollution", "road tolls in downtown areas and emissions"]),
    (3, "remote-housing", ["remote work and housing prices", "working from home and property values in suburbs", "telecommuting and rental costs in large cities"]),
]
TEMPLATES = [
    "What is known about {s}?",
    "How strong is the evidence on {s}?",
    "Can you summarise the main findings on {s}?",
    "I am researching {s}; what are the key open questions?",
    "Is there a causal link in the case of {s}?",
    "Which studies examine {s}, and what do they conclude?",
    "Explain the mechanisms behind {s}.",
    "What are common confounders when studying {s}?",
]
STARTERS = ["How do I repair", "Where can I buy", "What is the best way to clean", "Can you recommend a recipe for", "Who sells",
            "How long does it take to bake", "Suggest a weekend trip involving", "What should I pack for", "How much does it cost to rent",
            "Any tips for photographing", "Where can I learn to paint", "Is it worth upgrading"]
ADJ = ["a vintage", "a rusty", "a small", "a colourful", "an old", "a wooden", "a folding", "a spicy", "a cheap", "a heavy", "a handmade", "a waterproof", "a tiny", "a luxury", "a second-hand"]
NOUN = ["bicycle", "teapot", "sourdough loaf", "guitar", "tent", "camera", "sofa", "kayak", "lantern", "backpack", "ceramic bowl", "typewriter", "skateboard", "curry", "bookshelf", "umbrella", "violin", "wardrobe", "pizza", "motorbike"]
PLACE = ["in Lisbon", "near Osaka", "in a small village", "for a family of five", "on a tight budget", "in winter", "in Buenos Aires", "before the holidays", "in rainy weather", "for beginners", "in Cape Town", "at home", "in Oslo", "for a birthday", "in the mountains"]


def build_corpus():
    items = []  # dict(text, kind, concept, topic, sub, tmpl)
    for ci, (tp, name, subs) in enumerate(CONCEPTS):
        for si, s in enumerate(subs):
            items.append(dict(text=s, kind="subj", concept=ci, topic=tp, sub=si, tmpl=-1))
        for si, s in enumerate(subs):
            for ti, t in enumerate(TEMPLATES):
                items.append(dict(text=t.format(s=s), kind="q", concept=ci, topic=tp, sub=si, tmpl=ti))
    rng = random.Random(20251004)
    combos = [(a, b, c, d) for a in range(len(STARTERS)) for b in range(len(ADJ)) for c in range(len(NOUN)) for d in range(len(PLACE))]
    rng.shuffle(combos)
    for (a, b, c, d) in combos[:2100]:
        items.append(dict(text=f"{STARTERS[a]} {ADJ[b]} {NOUN[c]} {PLACE[d]}?", kind="dist", concept=-1, topic=-1, sub=-1, tmpl=-1))
    return items


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(max_length=128)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"), so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    print("inputs", names, "outputs", [o.name for o in sess.get_outputs()])
    items = build_corpus()
    texts = [it["text"] for it in items]
    enc = tok.encode_batch(texts)
    order = sorted(range(len(texts)), key=lambda i: len(enc[i].ids))
    out = np.zeros((len(texts), 384), dtype=np.float32)
    B = 32
    for s in range(0, len(order), B):
        idx = order[s:s + B]
        L = max(len(enc[i].ids) for i in idx)
        ids = np.zeros((len(idx), L), dtype=np.int64)
        am = np.zeros((len(idx), L), dtype=np.int64)
        for r, i in enumerate(idx):
            n = len(enc[i].ids)
            ids[r, :n] = enc[i].ids
            am[r, :n] = 1
        feed = {"input_ids": ids, "attention_mask": am}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hs = sess.run(None, feed)[0]  # [B,L,384]
        m = am[:, :, None].astype(np.float32)
        pooled = (hs * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        for r, i in enumerate(idx):
            out[i] = pooled[r]

    def cos(a, b):
        return float(out[a] @ out[b])

    q = [i for i, it in enumerate(items) if it["kind"] == "q"]
    rngs = random.Random(1)
    same, sib, diff, dist = [], [], [], []
    for _ in range(4000):
        i, j = rngs.sample(q, 2)
        a, b = items[i], items[j]
        (same if a["concept"] == b["concept"] else sib if a["topic"] == b["topic"] else diff).append(cos(i, j))
    d = [i for i, it in enumerate(items) if it["kind"] == "dist"]
    for _ in range(2000):
        i, j = rngs.sample(d, 2)
        dist.append(cos(i, j))
    mean = lambda x: round(float(np.mean(x)), 3)
    print("MiniLM mean cos: same-concept", mean(same), "sibling", mean(sib), "diff-topic", mean(diff), "distractor-distractor", mean(dist))
    os.makedirs(OUT_DIR, exist_ok=True)
    out.tofile(os.path.join(OUT_DIR, "f5_embeddings.f32"))
    meta = dict(
        note="SYNTHETIC templated research-question corpus written by the f5 worker; not real user text. Vectors = all-MiniLM-L6-v2 (ONNX, local ruvector copy), mean-pooled over tokens, L2-normalised, 384-d, row-major f32 LE in f5_embeddings.f32",
        dim=384, n=len(items), model_sha256=sha(os.path.join(MODEL_DIR, "model.onnx")), tokenizer_sha256=sha(os.path.join(MODEL_DIR, "tokenizer.json")),
        onnxruntime=ort.__version__, concepts=[dict(topic=c[0], name=c[1], subjects=c[2]) for c in CONCEPTS], templates=TEMPLATES,
        sanity_mean_cos=dict(same_concept=mean(same), sibling=mean(sib), diff_topic=mean(diff), distractor_distractor=mean(dist)),
        items=items,
    )
    with open(os.path.join(OUT_DIR, "f5_corpus.json"), "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print("wrote", len(items), "vectors in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
