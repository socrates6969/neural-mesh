from v2 import *
import pickle
B = pickle.load(open(os.path.join(RES, "v2_base.pkl"), "rb"))
def an_templ(tm):
    def pred(sk, x):
        if sk != "analogies": return base_pred(sk, x)
        a, b, c = [emb(tm.format(q)) for q in x["abc"]]; O = np.stack([emb(tm.format(o)) for o in x["opts"]])
        return int(np.argmax(cosm(b - a + c, O)))
    def texts(sk, x):
        if sk != "analogies": return base_texts(sk, x)
        return [tm.format(q) for q in x["abc"] + x["opts"]]
    return pred, texts
def an_sentrel(fmt):   # cos(embed("a is to b"), embed("c is to o"))
    def pred(sk, x):
        if sk != "analogies": return base_pred(sk, x)
        a, b, c = x["abc"]; r = emb(fmt.format(a, b)); O = np.stack([emb(fmt.format(c, o)) for o in x["opts"]]); return int(np.argmax(cosm(r, O)))
    def texts(sk, x):
        if sk != "analogies": return base_texts(sk, x)
        a, b, c = x["abc"]; return [fmt.format(a, b)] + [fmt.format(c, o) for o in x["opts"]]
    return pred, texts
def an_mix(lam):  # offset + direct similarity of option to b (relation fillers share class)
    def pred(sk, x):
        if sk != "analogies": return base_pred(sk, x)
        a, b, c = [emb(q) for q in x["abc"]]; O = np.stack([emb(o) for o in x["opts"]])
        return int(np.argmax(cosm(b - a + c, O) + lam * cosm(b, O)))
    return pred, base_texts
C = {"tmpl_word": an_templ("the word {}"), "tmpl_meaning": an_templ("the meaning of {}"), "tmpl_bare_quote": an_templ("'{}'"),
     "sentrel_is_to": an_sentrel("{} is to {}"), "sentrel_arrow": an_sentrel("{} : {}"), "sentrel_has": an_sentrel("the relation between {} and {}"),
     "mix0.5": an_mix(0.5), "mix1": an_mix(1.0), "mix2": an_mix(2.0)}
if __name__ == "__main__":
    for n, (p, t) in C.items():
        c = run_v2(p, t, skills=("analogies",))
        print(n, "TUNE", round(tab_mean(c, "v2_analogies", TUNE), 3), "(base", round(tab_mean(B, "v2_analogies", TUNE), 3), ")")
