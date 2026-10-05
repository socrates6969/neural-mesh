from v2 import *
from scipy import stats
HARD = [("The doctor prescribed ___ to treat the bacterial infection.","antibiotics",["stethoscope","hospital","surgeon"]),
("The pilot announced that the plane would ___ in ten minutes.","land",["cockpit","runway","altitude"]),
("The chef seasoned the soup with a pinch of ___.","salt",["kitchen","recipe","saucepan"]),
("The teacher asked the class to ___ the essay by Friday.","submit",["classroom","blackboard","homework"]),
("The goalkeeper managed to ___ the penalty kick.","save",["stadium","referee","whistle"]),
("The farmer rose early to ___ the cows before sunrise.","milk",["tractor","barn","harvest"]),
("The judge asked the jury to ___ the evidence carefully.","consider",["courtroom","verdict","lawyer"]),
("The sailor tied the boat to the ___ with a strong rope.","dock",["ocean","compass","anchor"]),
("The programmer had to ___ the bug before the release.","fix",["keyboard","software","server"]),
("The astronomer used a telescope to ___ the distant galaxy.","observe",["planet","orbit","universe"]),
("The baker let the dough ___ for an hour before baking.","rise",["flour","oven","bread"]),
("The librarian asked him to ___ the book by the end of the month.","return",["shelf","novel","catalogue"]),
("The mechanic needed to ___ the flat tyre on the car.","replace",["engine","garage","wheel"]),
("The conductor raised his baton and the orchestra began to ___.","play",["violin","concert","symphony"]),
("The detective began to ___ the suspects one by one.","question",["crime","fingerprint","evidence"]),
("The tailor will ___ the trousers so they fit properly.","shorten",["fabric","needle","suit"]),
("The gardener used shears to ___ the hedge into a neat shape.","trim",["flower","soil","garden"]),
("The banker explained how to ___ money into a savings account.","deposit",["vault","interest","loan"]),
("The surgeon washed her hands before she began to ___.","operate",["scalpel","patient","theatre"]),
("The photographer adjusted the lens to ___ the bird in flight.","capture",["camera","tripod","flash"]),
("The cyclist had to ___ hard to climb the steep hill.","pedal",["bicycle","helmet","saddle"]),
("The painter chose to ___ the wall a pale shade of blue.","paint",["brush","canvas","easel"]),
("The hikers stopped to ___ the map before choosing a path.","consult",["mountain","backpack","compass"]),
("The professor will ___ a lecture on thermodynamics tomorrow.","deliver",["university","physics","textbook"])]
import random
def run(lam):
    out = []
    for st, an, ds in HARD:
        opts = [an] + ds
        for sh in range(3):  # 3 shuffles irrelevant for argmax; keep single
            break
        embed_many([st] + opts + [st.replace("___", o) for o in opts])
        F = np.stack([emb(st.replace("___", o)) for o in opts]); O = np.stack([emb(o) for o in opts])
        s = cosm(emb(st), F) + lam * cosm(emb(st), O); out.append(int(np.argmax(s)) == 0)
    return np.array(out)
b = run(0); 
for lam in (0.5, 1.0):
    c = run(lam); n01 = int(((~b) & c).sum()); n10 = int((b & ~c).sum())
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    print(f"hard-distractor cloze n={len(b)}: base {b.mean():.3f} mix{lam} {c.mean():.3f} gains {n01} losses {n10} sign-test p={p:.3f}")
