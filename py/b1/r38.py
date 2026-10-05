from v2 import *
import pickle, re
B = pickle.load(open(os.path.join(RES, "v2_base.pkl"), "rb"))
def window(st, o, k):
    w = st.split(); i = [j for j, x in enumerate(w) if "___" in x][0]; seg = w[max(0, i - k): i + k + 1]
    return " ".join(seg).replace("___", o)
def cl(kind, lam=1.0, k=3):
    def txt(x):
        st = x["st"]; T = [st, st.replace("___", "").replace("  ", " ")] + list(x["opts"]) + [st.replace("___", o) for o in x["opts"]]
        T += [window(st, o, k) for o in x["opts"]]; T += [window(st, "", k)]; T += ["Fill the blank: " + st.replace("___", o) for o in x["opts"]] + ["Fill the blank: " + st]
        return T
    def pred(sk, x):
        if sk != "cloze": return base_pred(sk, x)
        st, ops = x["st"], x["opts"]; F = np.stack([emb(st.replace("___", o)) for o in ops]); O = np.stack([emb(o) for o in ops])
        s0 = cosm(emb(st), F)
        if kind == "base": return int(np.argmax(s0))
        if kind == "opt": return int(np.argmax(cosm(emb(st), O)))
        if kind == "mix": return int(np.argmax(s0 + lam * cosm(emb(st), O)))
        if kind == "window": W = np.stack([emb(window(st, o, k)) for o in ops]); return int(np.argmax(cosm(emb(window(st, "", k)), W)))
        if kind == "winmix": W = np.stack([emb(window(st, o, k)) for o in ops]); return int(np.argmax(s0 + lam * cosm(emb(window(st, "", k)), W)))
        if kind == "blankless": return int(np.argmax(cosm(emb(st.replace("___", "").replace("  ", " ")), F)))
        if kind == "prompt": P = np.stack([emb("Fill the blank: " + st.replace("___", o)) for o in ops]); return int(np.argmax(cosm(emb("Fill the blank: " + st), P)))
    def texts(sk, x): return txt(x) if sk == "cloze" else base_texts(sk, x)
    return pred, texts
C = {"opt_only": cl("opt"), "mix0.5": cl("mix", 0.5), "mix1": cl("mix", 1.0), "mix2": cl("mix", 2.0), "window3": cl("window", k=3), "window5": cl("window", k=5),
     "winmix1_k3": cl("winmix", 1.0, 3), "winmix1_k5": cl("winmix", 1.0, 5), "blankless": cl("blankless"), "prompt": cl("prompt")}
if __name__ == "__main__":
    for n, (p, t) in C.items():
        c = run_v2(p, t, skills=("cloze",))
        print(n, "TUNE", round(tab_mean(c, "v2_cloze", TUNE), 3), "(base", round(tab_mean(B, "v2_cloze", TUNE), 3), ")")
