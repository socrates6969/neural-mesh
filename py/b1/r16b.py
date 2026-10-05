from sem import *
from r16 import E_t
for tm in ("the meaning of {}", "the word {}"):
    semrun(f"R16 template '{tm}' for verbal_analogies only", E={"verbal_analogies": E_t(tm)}, target=["sem_verbal_analogies"])
D.save_extra()
