from v2 import *
import pickle
B = pickle.load(open(os.path.join(RES, "v2_base.pkl"), "rb"))
def mk(tmpl_t=None, tmpl_o=None):
    def f(t, tm): return tm.format(t) if tm else t
    return f
# --- vocab candidates
def vocab_templ(tw, to=None):
    E = lambda t: emb(t)
    def pred(sk, x):
        if sk != "vocab": return base_pred(sk, x)
        O = np.stack([emb(to.format(o) if to else o) for o in x["opts"]])
        return int(np.argmax(cosm(emb(tw.format(x["t"]) if tw else x["t"]), O)))
    def texts(sk, x):
        if sk != "vocab": return base_texts(sk, x)
        return [tw.format(x["t"]) if tw else x["t"]] + [to.format(o) if to else o for o in x["opts"]]
    return pred, texts
def vocab_contrast(lam, opp="the opposite of {}"):
    def pred(sk, x):
        if sk != "vocab": return base_pred(sk, x)
        O = np.stack([emb(o) for o in x["opts"]])
        return int(np.argmax(cosm(emb(x["t"]), O) - lam * cosm(emb(opp.format(x["t"])), O)))
    def texts(sk, x):
        if sk != "vocab": return base_texts(sk, x)
        return [x["t"], opp.format(x["t"])] + x["opts"]
    return pred, texts
CANDS = {"tmpl_meaning": vocab_templ("the meaning of {}", "the meaning of {}"), "tmpl_word": vocab_templ("the word {}", "the word {}"),
 "tmpl_synonym": vocab_templ("another word for {}", None), "tmpl_means": vocab_templ("{} means", "{}"), "tmpl_similar": vocab_templ("a word similar to {}", "{}")}
for lam in (0.25, 0.5, 1.0): CANDS[f"contrast{lam}"] = vocab_contrast(lam)
if __name__ == "__main__":
    for n, (p, t) in CANDS.items():
        c = run_v2(p, t, skills=("vocab",))
        print(n, "TUNE", round(tab_mean(c, "v2_vocab", TUNE), 3), "(base", round(tab_mean(B, "v2_vocab", TUNE), 3), ")")
