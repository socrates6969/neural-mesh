from sem import *
tmpls = ["the word {}", "{} (a word)", "a synonym of {}", "the meaning of {}", "a {}", "something related to {}"]
def E_t(tm): 
    def f(t):
        # sentence-completion stems/filled sentences are full sentences: only wrap short texts (<=3 words)
        return D.v(tm.format(t)) if len(t.split()) <= 3 else D.v(t)
    return f
base = semantic_acc(D, [it for w in WORLDS for s in TUNE for it in D.items[f"{w}{s}"]])
print("TUNE base", {k: round(v, 3) for k, v in base.items()})
res = {}
for tm in tmpls:
    a = semantic_acc(D, [it for w in WORLDS for s in TUNE for it in D.items[f"{w}{s}"]], E_t(tm)); res[tm] = a
    print(tm, {k: round(v, 3) for k, v in a.items()})
D.save_extra()
import pickle; pickle.dump(res, open("/tmp/r16.pkl", "wb")) if False else None
for tm in tmpls:
    pass
