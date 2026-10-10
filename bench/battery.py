"""bench/battery.py - build the verifiable battery, split by the iron rules.

  TUNE  today's 120 items (bench/battery_tune.py, seed 20261010), frozen: bench/battery.json is reproduced byte for byte.
        v1/v2 front-ends were designed on these items, so TUNE results are tuning evidence only (descriptive).
  HELD  seed 20261011 } template-built, disjoint from TUNE and from each other; 30 items per axis:
  CONF  seed 20261012 }   math, physics, code (expected outputs precomputed on 4 fixed + 4 rng inputs), research_public
                          (stdlib facts, errno dropped: platform-dependent) and research_private (numeric consts in src/*.rs,
                          minus the 15 TUNE ones, assigned to HELD/CONF by sha256(file:name) % 2; fewer than 30 if the pool
                          is short - the shortfall is recorded in the header, never filled by reusing TUNE constants).
Answers are computed by code, not typed by hand. Each HELD/CONF item gets an offline leak check through
frontend.frontend_prompt (the pre-registered front-end): research_private is leak=by-construction (the answer lives in the
corpus; that is the claim under test); any other item whose answer shows up in the retrieved context is resampled (<= 5
times), else exit 1.

Output: bench/battery_<split>.json = {"header": {split, seed, counts, shortfall, items_sha, corpus_sha, frontend_sha, ...},
"items": [...]} (+ bench/battery.json for TUNE, unchanged format).
  python bench/battery.py --split tune|held|conf|all      build (exit 0; 1 leak/disjointness failure)
  python bench/battery.py --check [--split ...]          verify existing files: a changed/missing src constant -> exit 2;
                                                          a changed retrieval corpus or front-end only warns."""
import argparse, base64, calendar, datetime, hashlib, http, json, math, os, pathlib, random, re, string, subprocess, sys, unicodedata, uuid, zlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent
NM = ROOT.parent
sys.path.insert(0, str(ROOT))
from common import AXES, axis_of, battery_path, file_sha, json_sha  # noqa: E402
import sandbox  # noqa: E402  (canon() only: the reference code below is trusted and runs in this process, never the sandbox)

SEEDS = {"tune": 20261010, "held": 20261011, "conf": 20261012}
N_PER_AXIS = 30
MAX_RESAMPLE = 5
# the front-end variant whose retrieval the leak check uses (= the pre-registered B/D front-end: v2, identifier-aware)
FRONTEND_ENV = {"MESH_V2": "1", "MESH_THRESH": "", "MESH_FRONTEND": ""}
FMT = "\n\nGive the final answer as a single number (no units) on the last line in the form `ANSWER: <number>`."
FMT2 = "\n\nGive the final answer on the last line in the form `ANSWER: <answer>` (a number, or a short exact string)."
CODE_FMT = "\nReturn ONLY the code in a single ```python block (no explanation, no tests)."
ABBR = {"math": "ma", "physics": "ph", "code": "co", "research_public": "rp", "research_private": "rv"}


class BuildError(Exception):
    pass


# =================================================================== templates (each: rng -> dict)
def _fr(rng, a, b, n=2):
    return round(rng.uniform(a, b), n)


def _num(q, a, tol=None):
    isint = float(a).is_integer()
    return dict(prompt=q + FMT, answer=a, kind="num", tol=(0 if isint else 1e-6) if tol is None else tol, sub="exact" if isint else "dec")


def _divisors(n):
    return sum(1 if d * d == n else 2 for d in range(1, math.isqrt(n) + 1) if n % d == 0)


def _base(n, b):
    d = []
    while n: d.append(str(n % b)); n //= b
    return int("".join(reversed(d)) or "0")


def _det3(m):
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])


def _fib(n):
    a, b = 0, 1
    for _ in range(n): a, b = b, a + b
    return a


WORDS = ["BANANA", "COMMITTEE", "BOOKKEEPER", "TENNESSEE", "ASSESSMENT", "SUCCESSES", "MISSISSAUGA", "PEPPERMILL", "PARALLEL", "BALLOON", "COFFEE", "STATISTICS"]


def _perm_word(w):
    c = Counter(w); r = math.factorial(len(w))
    for v in c.values(): r //= math.factorial(v)
    return r


MATH = [
    lambda R: (lambda a, b, m: _num(f"Compute {a}^{b} mod {m}.", pow(a, b, m)))(R.randint(3, 19), R.randint(150, 900), R.choice([997, 1000, 1009, 9973, 10007])),
    lambda R: (lambda a, b: _num(f"What is {a} * {b}?", a * b))(R.randint(10000, 99999), R.randint(10000, 99999)),
    lambda R: (lambda n: _num(f"How many primes are less than {n}?", sum(1 for k in range(2, n) if all(k % p for p in range(2, math.isqrt(k) + 1)))))(R.randint(1200, 6000)),
    lambda R: (lambda n: _num(f"What is the sum of the decimal digits of 2^{n}?", sum(map(int, str(2 ** n)))))(R.randint(120, 400)),
    lambda R: (lambda g, a, b: _num(f"What is gcd({g * a}, {g * b})?", math.gcd(g * a, g * b)))(R.randint(100, 9999), R.randint(1000, 99999), R.randint(1000, 99999)),
    lambda R: (lambda a, b, c: _num(f"What is lcm({a}, {b}, {c})?", math.lcm(a, b, c)))(R.randint(12, 240), R.randint(12, 240), R.randint(12, 240)),
    lambda R: (lambda n, k: _num(f"How many ways can {k} items be chosen (order irrelevant) from {n} distinct items?", math.comb(n, k)))(R.randint(30, 70), R.randint(3, 8)),
    lambda R: (lambda x: _num(f"Convert 0x{x:X} to decimal.", x))(R.randint(0x1000, 0xFFFFF)),
    lambda R: (lambda n, b: _num(f"Convert decimal {n} to base {b}. Give the base-{b} digits written as a plain integer (digits 1,2,3 -> 123).", _base(n, b)))(R.randint(5000, 900000), R.randint(3, 9)),
    lambda R: (lambda n, k: _num(f"What is {n}! / {n - k}! ?", math.factorial(n) // math.factorial(n - k)))(R.randint(14, 30), R.randint(2, 4)),
    lambda R: (lambda n, a, b: _num(f"What is the sum of all integers from 1 to {n} that are divisible by {a} or {b}?", sum(x for x in range(1, n + 1) if x % a == 0 or x % b == 0)))(R.randint(500, 5000), R.choice([3, 4, 6, 7]), R.choice([5, 9, 11, 13])),
    lambda R: (lambda n: _num(f"What is the {n}th Fibonacci number, with F1=1, F2=1?", _fib(n)))(R.randint(25, 60)),
    lambda R: (lambda n: _num(f"How many positive divisors does {n} have?", _divisors(n)))(R.choice([2 ** a * 3 ** b * 5 ** c * 7 ** d for a in range(1, 6) for b in range(0, 4) for c in range(0, 3) for d in range(0, 2)])),
    lambda R: (lambda m: _num(f"What is the determinant of the matrix {m}?", _det3(m)))([[R.randint(-9, 9) for _ in range(3)] for _ in range(3)]),
    lambda R: (lambda n, k: _num(f"A fair coin is flipped {n} times. What is the probability of exactly {k} heads? Give a decimal rounded to 5 places.", round(math.comb(n, k) / 2 ** n, 5)))(R.randint(8, 16), R.randint(2, 7)),
    lambda R: (lambda p, a: _num(f"What is the value of the integral of x^{p} from 0 to {a}?", a ** (p + 1) / (p + 1)))(R.randint(2, 5), R.randint(2, 6)),
    lambda R: (lambda a, b, c, x: _num(f"What is the derivative of f(x)=x^4 - {a}x^3 + {b}x^2 - {c}x at x={x}?", 4 * x ** 3 - 3 * a * x * x + 2 * b * x - c))(R.randint(1, 9), R.randint(1, 9), R.randint(1, 9), R.randint(2, 7)),
    lambda R: (lambda p, r, y: _num(f"Compound interest: {p} invested at {r}% annual interest compounded monthly for {y} years. Final amount rounded to 2 decimals?", round(p * (1 + r / 1200) ** (12 * y), 2)))(R.randint(10, 200) * 100, R.choice([2, 3, 4, 5, 7, 8]), R.randint(3, 15)),
    lambda R: (lambda n: _num(f"How many trailing zeros does {n}! have?", sum(n // 5 ** k for k in range(1, 8))))(R.randint(120, 2000)),
    lambda R: (lambda s, k: _num(f"How many ordered {k}-tuples of non-negative integers sum to {s}?", math.comb(s + k - 1, k - 1)))(R.randint(10, 40), R.randint(3, 5)),
    lambda R: (lambda n: _num(f"What is the square root of {n} to 8 decimal places?", round(math.sqrt(n), 8)))(R.choice([k for k in range(3, 200) if math.isqrt(k) ** 2 != k])),
    lambda R: (lambda n: _num(f"What is the sum of squares of the integers from 1 to {n}?", n * (n + 1) * (2 * n + 1) // 6))(R.randint(40, 300)),
    lambda R: (lambda xs: _num(f"What is the population standard deviation of the numbers {', '.join(map(str, xs))}? Round to 4 decimals.", round((sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs)) ** .5, 4)))([R.randint(1, 99) for _ in range(R.randint(5, 8))]),
    lambda R: (lambda w: _num(f"How many distinct permutations are there of the letters of {w}?", _perm_word(w)))(R.choice(WORDS)),
    lambda R: (lambda a, m: _num(f"What is the multiplicative inverse of {a} modulo {m} (the integer x with 0 < x < {m})?", pow(a, -1, m)))(R.choice([k for k in range(2, 500) if math.gcd(k, 1009) == 1]), 1009),
]

G0, C0, H0, RG, E0 = 9.80665, 299792458, 6.62607015e-34, 8.314462618, 1.602176634e-19


def _phys(q, a):
    return dict(prompt=q + FMT, answer=a, kind="num", tol=0.01, sub="numeric")


PHYS = [
    lambda R: (lambda v, th: _phys(f"A projectile is launched on level ground at {v} m/s at {th} degrees above horizontal, no air resistance, g=9.80665 m/s^2. What is its horizontal range in metres?", v * v * math.sin(2 * math.radians(th)) / G0))(_fr(R, 10, 40, 0), R.choice([25, 30, 35, 40, 50, 55, 60, 65])),
    lambda R: (lambda m, v: _phys(f"What is the kinetic energy in joules of a {m} kg mass moving at {v} m/s?", .5 * m * v * v))(_fr(R, 1, 20, 1), _fr(R, 2, 30, 1)),
    lambda R: (lambda rs: _phys(f"Three resistors of {rs[0]}, {rs[1]} and {rs[2]} ohms are connected in parallel. What is the equivalent resistance in ohms?", 1 / sum(1 / x for x in rs)))([_fr(R, 10, 100, 0) for _ in range(3)]),
    lambda R: (lambda n, T, V: _phys(f"{n} mol of ideal gas at {T} K occupy {V} m^3. Using R=8.314462618 J/(mol K), what is the pressure in pascals?", n * RG * T / V))(_fr(R, .5, 3, 2), _fr(R, 250, 400, 0), _fr(R, .01, .1, 3)),
    lambda R: (lambda f, d: _phys(f"A thin converging lens has focal length {f} cm. An object is {d} cm from the lens. At what distance in cm (positive means real image) is the image formed?", 1 / (1 / f - 1 / d)))(_fr(R, 5, 30, 0), _fr(R, 40, 120, 0)),
    lambda R: (lambda v: _phys(f"A particle moves at {v}c. What is its Lorentz factor gamma?", 1 / math.sqrt(1 - v * v)))(_fr(R, .3, .97, 3)),
    lambda R: (lambda M, r: _phys(f"What is the escape velocity in m/s from the surface of a planet of mass {M:.3e} kg and radius {r:.3e} m (G=6.674e-11)?", math.sqrt(2 * 6.674e-11 * M / r)))(float(f"{R.uniform(1e23, 2e27):.3e}"), float(f"{R.uniform(2e6, 7e7):.3e}")),
    lambda R: (lambda L: _phys(f"What is the period in seconds of a simple pendulum of length {L} m for small oscillations, g=9.80665 m/s^2?", 2 * math.pi * math.sqrt(L / G0)))(_fr(R, .3, 3, 2)),
    lambda R: (lambda fs, vs: _phys(f"A source emitting {fs} Hz moves toward a stationary observer at {vs} m/s. Speed of sound is 343 m/s. What frequency in Hz does the observer hear?", fs * 343 / (343 - vs)))(_fr(R, 300, 900, 0), _fr(R, 10, 40, 0)),
    lambda R: (lambda Rc, C: _phys(f"An RC circuit has R={Rc} ohms and C={C} F. What is the time constant in seconds?", Rc * C))(_fr(R, 1, 50, 1) * 1000, _fr(R, 1, 100, 0) * 1e-6),
    lambda R: (lambda lam: _phys(f"What is the energy in joules of a photon of wavelength {lam} nm? (h=6.62607015e-34, c=299792458)", H0 * C0 / (lam * 1e-9)))(_fr(R, 300, 800, 0)),
    lambda R: (lambda a: _phys(f"What is the orbital period in seconds of a satellite in a circular orbit of radius {a} m around Earth (GM = 3.986004418e14 m^3/s^2)?", 2 * math.pi * math.sqrt(a ** 3 / 3.986004418e14)))(_fr(R, 6.5e6, 4.2e7, 0)),
    lambda R: (lambda m1, v1, m2: _phys(f"A {m1} kg mass moving at {v1} m/s collides perfectly inelastically with a stationary {m2} kg mass. What is the final speed in m/s?", m1 * v1 / (m1 + m2)))(_fr(R, 1, 10, 1), _fr(R, 5, 20, 1), _fr(R, 1, 10, 1)),
    lambda R: (lambda I, V: _phys(f"A {V} V source drives {I} A through a resistor. How much power in watts is dissipated?", V * I))(_fr(R, .5, 5, 1), _fr(R, 5, 48, 0)),
    lambda R: (lambda T: _phys(f"By Wien's displacement law (b=2.897771955e-3 m K), what is the peak wavelength in nanometres of a black body at {T} K?", 2.897771955e-3 / T * 1e9))(_fr(R, 300, 6000, 0)),
    lambda R: (lambda h: _phys(f"A ball is dropped from {h} m with no air resistance (g=9.80665). With what speed in m/s does it hit the ground?", math.sqrt(2 * G0 * h)))(_fr(R, 5, 100, 0)),
    lambda R: (lambda m, k: _phys(f"A mass of {m} kg hangs on a spring of constant {k} N/m. What is the angular frequency in rad/s?", math.sqrt(k / m)))(_fr(R, .1, 5, 1), _fr(R, 50, 500, 0)),
    lambda R: (lambda d, q1, q2: _phys(f"Two point charges of {q1} microcoulomb and {q2} microcoulomb are {d} m apart in vacuum. Coulomb constant 8.9875517923e9. What is the magnitude of the force in newtons?", 8.9875517923e9 * q1 * 1e-6 * q2 * 1e-6 / d ** 2))(_fr(R, .5, 5, 1), _fr(R, 1, 10, 1), _fr(R, 1, 10, 1)),
    lambda R: (lambda V: _phys(f"What is the buoyant force in newtons on an object fully submerged in water (density 1000 kg/m^3) with volume {V} m^3, g=9.80665?", 1000 * V * G0))(_fr(R, 1e-4, 1e-2, 5)),
    lambda R: (lambda ang, mu: _phys(f"A block slides down a rough incline of {ang} degrees with kinetic friction coefficient {mu}, g=9.80665. What is its acceleration in m/s^2?", G0 * (math.sin(math.radians(ang)) - mu * math.cos(math.radians(ang)))))(R.choice([25, 30, 35, 40, 45]), _fr(R, .1, .3, 2)),
    lambda R: (lambda T, gas, M: _phys(f"What is the rms speed in m/s of {gas} molecules (molar mass {M} kg/mol) at {T} K? (R=8.314462618)", math.sqrt(3 * RG * T / M)))(_fr(R, 250, 400, 0), *R.choice([("oxygen", 0.0319988), ("helium", 0.004002602), ("argon", 0.039948), ("nitrogen", 0.0280134)])),
    lambda R: (lambda m, dT: _phys(f"How much heat in joules is needed to raise {m} kg of water by {dT} K? Use c=4186 J/(kg K).", m * 4186 * dT))(_fr(R, .2, 2, 1), R.randint(5, 60)),
    lambda R: (lambda L, n: _phys(f"What is the frequency in Hz of harmonic number {n} on a string fixed at both ends, of length {L} m, with wave speed 340 m/s?", n * 340 / (2 * L)))(_fr(R, .5, 2, 2), R.randint(1, 6)),
    lambda R: (lambda B, v: _phys(f"A proton (m=1.67262192e-27 kg, q=1.602176634e-19 C) moves at {v} m/s perpendicular to a {B} T magnetic field. What is the radius of its circular path in metres?", 1.67262192e-27 * v / (E0 * B)))(_fr(R, .1, 2, 2), _fr(R, 1e5, 1e7, 0)),
    lambda R: (lambda n, a: _phys(f"Light passes from air (n=1.0) into glass with n={n} at an incidence angle of {a} degrees. What is the refraction angle in degrees (Snell's law)?", math.degrees(math.asin(math.sin(math.radians(a)) / n))))(_fr(R, 1.3, 1.9, 2), _fr(R, 10, 70, 0)),
    lambda R: (lambda rho, V: _phys(f"What is the weight in newtons (g=9.80665) of a metal block with density {rho} kg/m^3 and volume {V} m^3?", rho * V * G0))(_fr(R, 2500, 11000, 0), _fr(R, 1e-4, 1e-3, 5)),
    lambda R: (lambda t12, t: _phys(f"A radioactive sample has half-life {t12} days. What fraction remains after {t} days? Give a decimal.", .5 ** (t / t12)))(_fr(R, 2, 30, 1), _fr(R, 5, 100, 0)),
    lambda R: (lambda P, t: _phys(f"A {P} W heater runs for {t} s. How much energy in kilojoules does it deliver?", P * t / 1000))(_fr(R, 100, 2000, 0), _fr(R, 10, 3600, 0)),
    lambda R: (lambda A, v: _phys(f"Water (density 1000 kg/m^3) flows at {v} m/s through a pipe of cross-section {A} m^2. What is the mass flow rate in kg/s?", 1000 * v * A))(_fr(R, .001, .1, 4), _fr(R, 1, 8, 1)),
    lambda R: (lambda C, V: _phys(f"What energy in joules is stored in a {C} F capacitor charged to {V} V?", .5 * C * V * V))(_fr(R, 1, 100, 0) * 1e-6, _fr(R, 5, 50, 0)),
    lambda R: (lambda kv: _phys(f"What is the de Broglie wavelength in picometres of an electron accelerated through {kv} V (non-relativistic; m=9.1093837e-31 kg, h=6.62607015e-34, e=1.602176634e-19)?", H0 / math.sqrt(2 * 9.1093837e-31 * E0 * kv) * 1e12))(_fr(R, 50, 5000, 0)),
]


# ---- code: parametrised specs. (name, description, reference source, 4 fixed inputs, rng -> one random input); {p...} params
def _rs(R, n=None, alpha="abcdeaeiou xyz"):
    return "".join(R.choice(alpha) for _ in range(R.randint(0, 14) if n is None else n))


def _rl(R, lo=-20, hi=20, n=None):
    return [R.randint(lo, hi) for _ in range(R.randint(0, 9) if n is None else n)]


CODE = [
    (lambda R: dict(k=R.randint(1, 25)), "shift_letters", "Shift every ASCII letter of the string forward by {k} positions in the alphabet (wrapping z->a, preserving case); other characters are unchanged.",
     "def shift_letters(s):\n    o = ''\n    for c in s:\n        if 'a' <= c <= 'z': o += chr((ord(c) - 97 + {k}) % 26 + 97)\n        elif 'A' <= c <= 'Z': o += chr((ord(c) - 65 + {k}) % 26 + 65)\n        else: o += c\n    return o",
     [["Hello, World!"], [""], ["xyz XYZ"], ["a1b2"]], lambda R: [_rs(R, None, "abcXYZ z!")]),
    (lambda R: dict(a=R.randint(2, 9), b=R.randint(10, 17)), "count_multiples", "Return how many integers in 1..n (inclusive) are divisible by {a} or by {b} (n may be 0 or negative, giving 0).",
     "def count_multiples(n):\n    return sum(1 for x in range(1, n + 1) if x % {a} == 0 or x % {b} == 0)", [[0], [1], [100], [-5]], lambda R: [R.randint(0, 3000)]),
    (lambda R: dict(k=R.randint(2, 5)), "chunk_sums", "Split the list of ints into consecutive chunks of size {k} (the last chunk may be shorter) and return the list of chunk sums.",
     "def chunk_sums(xs):\n    return [sum(xs[i:i + {k}]) for i in range(0, len(xs), {k})]", [[[]], [[1]], [[1, 2, 3, 4, 5, 6, 7]], [[-1, 1, -1, 1]]], lambda R: [_rl(R)]),
    (lambda R: (lambda k: dict(k=k, kth=["", "1st", "2nd", "3rd", "4th"][k]))(R.randint(2, 4)), "kth_largest_distinct", "Return the {kth} largest DISTINCT value in the list of ints, or None if there are fewer than {k} distinct values.",
     "def kth_largest_distinct(xs):\n    d = sorted(set(xs), reverse=True)\n    return d[{k} - 1] if len(d) >= {k} else None", [[[]], [[5, 5, 5]], [[3, 1, 4, 1, 5, 9, 2, 6]], [[-1, -2, -3, -4]]], lambda R: [_rl(R, -5, 5)]),
    (lambda R: dict(w=R.randint(4, 9), c=R.choice("*0.#")), "pad_left", "Left-pad the string with the character '{c}' to width {w}; strings of length >= {w} are returned unchanged.",
     "def pad_left(s):\n    return '{c}' * max(0, {w} - len(s)) + s", [[""], ["abc"], ["abcdefghijk"], ["{c}"]], lambda R: [_rs(R)]),
    (lambda R: dict(b=R.randint(3, 9)), "digit_sum_base", "Return the sum of the digits of the non-negative int n written in base {b}.",
     "def digit_sum_base(n):\n    s = 0\n    while n:\n        s += n % {b}; n //= {b}\n    return s", '[[0], [1], [{b}], [1000]]', lambda R: [R.randint(0, 10 ** 6)]),
    (lambda R: dict(k=R.randint(1, 4)), "rotate_left", "Return a new list rotated LEFT by {k} positions (wrapping around; an empty list stays empty).",
     "def rotate_left(xs):\n    if not xs: return []\n    k = {k} % len(xs)\n    return xs[k:] + xs[:k]", [[[]], [[1]], [[1, 2, 3, 4, 5]], [[1, 2]]], lambda R: [_rl(R)]),
    (lambda R: dict(k=R.randint(2, 4), o=R.randint(0, 1)), "every_nth", "Return the elements of the list at indices {o}, {o}+{k}, {o}+2*{k}, ... (every {k} positions, starting at index {o}).",
     "def every_nth(xs):\n    return xs[{o}::{k}]", [[[]], [[1]], [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], [["a", "b", "c"]]], lambda R: [_rl(R)]),
    (lambda R: dict(lo=R.randint(-10, 0), hi=R.randint(1, 10)), "clamp_all", "Return the list with every value clamped to the closed range [{lo}, {hi}].",
     "def clamp_all(xs):\n    return [min({hi}, max({lo}, x)) for x in xs]", '[[[]], [[-100, 100]], [[{lo}, {hi}, 0]], [[5, -5]]]', lambda R: [_rl(R)]),
    (lambda R: dict(c=[R.randint(-5, 5) for _ in range(R.randint(3, 5))]), "poly_eval", "Evaluate the polynomial with coefficients {c} (constant term first, i.e. c0 + c1*x + c2*x^2 + ...) at the integer x.",
     "def poly_eval(x):\n    return sum(c * x ** i for i, c in enumerate({c}))", [[0], [1], [-1], [10]], lambda R: [R.randint(-50, 50)]),
    (lambda R: dict(k=R.randint(2, 4)), "window_max", "Return the list of maxima of every contiguous window of size {k} (empty list if the input is shorter than {k}).",
     "def window_max(xs):\n    return [max(xs[i:i + {k}]) for i in range(len(xs) - {k} + 1)]", [[[]], [[1]], [[1, 3, -1, -3, 5, 3, 6, 7]], [[2, 2, 2, 2]]], lambda R: [_rl(R)]),
    (lambda R: dict(p=R.choice(["aa", "ab", "aba", "xx", "ana"])), "count_overlapping", "Return how many times the substring '{p}' occurs in the string, counting overlapping occurrences.",
     "def count_overlapping(s):\n    return sum(1 for i in range(len(s) - len('{p}') + 1) if s[i:i + len('{p}')] == '{p}')", [[""], ["{p}"], ["{p}{p}{p}"], ["banana"]], lambda R: [_rs(R, None, "abnx")]),
    (lambda R: dict(b=R.randint(2, 9)), "to_base", "Return the representation of the non-negative int n in base {b} as a string of digits ('0' for zero).",
     "def to_base(n):\n    if n == 0: return '0'\n    d = ''\n    while n:\n        d = str(n % {b}) + d; n //= {b}\n    return d", '[[0], [1], [{b}], [255]]', lambda R: [R.randint(0, 10 ** 5)]),
    (lambda R: dict(a=R.randint(2, 5), b=R.randint(6, 9)), "foo_bar", "Return the list of strings for 1..n: 'FooBar' for multiples of both {a} and {b}, 'Foo' for multiples of {a}, 'Bar' for multiples of {b}, else the number as a string.",
     "def foo_bar(n):\n    o = []\n    for i in range(1, n + 1):\n        o.append('FooBar' if i % {a} == 0 and i % {b} == 0 else 'Foo' if i % {a} == 0 else 'Bar' if i % {b} == 0 else str(i))\n    return o", '[[0], [1], [{a}], [{b}]]', lambda R: [R.randint(0, 80)]),
    (lambda R: dict(c=R.choice("*#_")), "mask_vowels", "Replace every vowel (a, e, i, o, u in either case) in the string with '{c}'.",
     "def mask_vowels(s):\n    return ''.join('{c}' if ch in 'aeiouAEIOU' else ch for ch in s)", [[""], ["AEIOU"], ["rhythm"], ["Hello World"]], lambda R: [_rs(R, None, "aEiOuxyzBC ")]),
    (lambda R: dict(k=R.randint(2, 4)), "long_runs", "Return the list of characters that start a run of at least {k} identical consecutive characters, in order of appearance (a character may appear more than once).",
     "def long_runs(s):\n    o = []\n    i = 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]: j += 1\n        if j - i >= {k}: o.append(s[i])\n        i = j\n    return o", [[""], ["a"], ["aaabbbbcc"], ["abab"]], lambda R: [_rs(R, None, "aab")]),
]


# ---- research_public: stdlib facts verified by executing the installed stdlib (errno dropped: platform-dependent)
def _word(R):
    return "".join(R.choice(string.ascii_lowercase) for _ in range(R.randint(5, 9)))


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]          # locale-independent
MONTHS = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def _date(R):
    return datetime.date(R.randint(1990, 2099), R.randint(1, 12), R.randint(1, 28))


_STATUS = [s for s in http.HTTPStatus if s.value not in (418,)]
_UNAMES = ["GREEK SMALL LETTER LAMDA", "EURO SIGN", "BLACK HEART SUIT", "SNOWMAN", "INFINITY", "SECTION SIGN", "PILCROW SIGN", "DEGREE SIGN",
           "LATIN SMALL LETTER SHARP S", "HEBREW LETTER ALEF", "CYRILLIC CAPITAL LETTER ZHE", "INTEGRAL", "NABLA", "WHITE SMILING FACE",
           "YEN SIGN", "COPYRIGHT SIGN", "MICRO SIGN", "BULLET", "HORIZONTAL ELLIPSIS", "CHECK MARK"]


def _pub(q, a):
    return dict(prompt=q + FMT2, answer=str(a), kind="num" if isinstance(a, int) else "str", tol=0, sub="public")


PUB = [
    lambda R: (lambda w: _pub(f"What are the first 8 hex characters of the SHA-256 digest of the ASCII bytes of '{w}'?", hashlib.sha256(w.encode()).hexdigest()[:8]))(_word(R)),
    lambda R: (lambda w: _pub(f"What are the first 8 hex characters of the MD5 digest of the ASCII bytes of '{w}'?", hashlib.md5(w.encode()).hexdigest()[:8]))(_word(R)),
    lambda R: (lambda w: _pub(f"What are the first 8 hex characters of the SHA-1 digest of the ASCII bytes of '{w}'?", hashlib.sha1(w.encode()).hexdigest()[:8]))(_word(R)),
    lambda R: (lambda w: _pub(f"What is the CRC-32 (zlib.crc32, unsigned) of the ASCII bytes of '{w}', as a decimal integer?", zlib.crc32(w.encode())))(_word(R)),
    lambda R: (lambda w: _pub(f"What is the Adler-32 checksum (zlib.adler32, unsigned) of the ASCII bytes of '{w}' as a decimal integer?", zlib.adler32(w.encode())))(_word(R)),
    lambda R: (lambda w: _pub(f"What is the standard base64 encoding of the ASCII text '{w}'?", base64.b64encode(w.encode()).decode()))(_word(R) + " " + _word(R)),
    lambda R: (lambda d: _pub(f"What day of the week is {d.isoformat()}? Answer with the English weekday name.", DAYS[d.weekday()]))(_date(R)),
    lambda R: (lambda y, m: _pub(f"How many days does {MONTHS[m]} {y} have in the Gregorian calendar?", calendar.monthrange(y, m)[1]))(R.randint(1800, 2400), R.choice([2, 2, 2, 4, 6, 9, 11, 1])),
    lambda R: (lambda d: _pub(f"What is the day of the year (1-based; January 1 is day 1) of {d.isoformat()}?", d.timetuple().tm_yday))(_date(R)),
    lambda R: (lambda d: _pub(f"What is the ISO 8601 week number of {d.isoformat()}?", d.isocalendar()[1]))(_date(R)),
    lambda R: (lambda s: _pub(f"In Python's http.HTTPStatus, what is the phrase for status code {s.value}?", s.phrase))(R.choice(_STATUS)),
    lambda R: (lambda n: _pub(f"What is the Unicode code point (decimal) of the character named {n}?", ord(unicodedata.lookup(n))))(R.choice(_UNAMES)),
    lambda R: (lambda w: _pub(f"What is str(uuid.uuid5(uuid.NAMESPACE_DNS, '{w}.example')) in Python?", str(uuid.uuid5(uuid.NAMESPACE_DNS, w + ".example"))))(_word(R)),
    lambda R: (lambda w: _pub(f"What is hashlib.blake2b(b'{w}', digest_size=4).hexdigest() in Python?", hashlib.blake2b(w.encode(), digest_size=4).hexdigest()))(_word(R)),
    lambda R: (lambda a, b: _pub(f"How many leap years are there from year {a} up to but not including year {b} in the Gregorian calendar (calendar.leapdays)?", calendar.leapdays(a, b)))(R.randint(1600, 2000), R.randint(2001, 2600)),
]


# =================================================================== private constants
CONST_RE = re.compile(r'^\s*(?:pub(?:\([a-z]+\))?\s+)?const\s+([A-Z][A-Z_0-9]*)\s*:\s*([a-z0-9]+)\s*=\s*(-?[0-9][0-9_]*(?:\.[0-9_]+)?(?:[eE][-+]?[0-9]+)?)'
                      r'(?:_?(?:[iu](?:8|16|32|64|128|size)|f32|f64))?\s*;\s*(?://.*)?$', re.M)
TUNE_CONSTS = [('cognition', 'N_DOMAINS'), ('compact', 'MAX_AGENTS'), ('fin', 'MAX_LIST'), ('fin', 'MAX_PERIODS'), ('fin', 'R_GAS'), ('flatq', 'FLAT_HNSW_CROSSOVER_N'),
               ('hebbian', 'INIT_WEIGHT'), ('hebbian', 'INIT_BASELINE'), ('mesh', 'DEFAULT_HELPER_EPSILON'), ('mesh', 'EDGE_BASELINE_ALPHA'), ('mesh', 'INIT_EDGE_BASELINE'),
               ('mesh', 'DISPATCH_RISK_ALPHA'), ('popguard', 'RECOMMENDED_INF_MIN'), ('scan', 'PAR_MIN_ROWS'), ('fin', 'RTOL')]


def _norm_value(v):
    return v.replace("_", "")


def scan_consts(text):
    """{name: (value, stripped line)} for names declared exactly once in this file"""
    found = [(m.group(1), _norm_value(m.group(3)), m.group(0).strip()) for m in CONST_RE.finditer(text)]
    cnt = Counter(n for n, _, _ in found)
    return {n: (v, line) for n, v, line in found if cnt[n] == 1}


def _git_show(rel):
    try:
        p = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=str(NM), capture_output=True, timeout=30)
        return p.stdout.decode("utf8", "replace") if p.returncode == 0 else None
    except Exception:
        return None


def const_pool(src_dir=None, committed_only=True):
    """[(file, name, value, line_sha)] of numeric consts in src/*.rs, declared once per file; with committed_only the same
    line must also be in git HEAD (uncommitted work in progress is not a stable answer key)"""
    src_dir = pathlib.Path(src_dir or NM / "src")
    out = []
    for p in sorted(src_dir.glob("*.rs")):
        cur = scan_consts(p.read_text(encoding="utf8", errors="replace"))
        head = None
        if committed_only:
            t = _git_show(f"src/{p.name}")
            head = scan_consts(t) if t is not None else None
        for name, (v, line) in sorted(cur.items()):
            if head is not None and head.get(name) != (v, line): continue
            out.append((p.name, name, v, hashlib.sha256(line.encode()).hexdigest()))
    return out


def split_of_const(file, name):
    return ("held", "conf")[int(hashlib.sha256(f"{file}:{name}".encode()).hexdigest(), 16) % 2]


def private_items(split, pool=None):
    tune = {(f + ".rs", n) for f, n in TUNE_CONSTS}
    pool = const_pool() if pool is None else pool
    mine = [c for c in pool if (c[0], c[1]) not in tune and split_of_const(c[0], c[1]) == split]
    mine.sort(key=lambda c: hashlib.sha256(f"{SEEDS[split]}:{c[0]}:{c[1]}".encode()).hexdigest())
    out = []
    for file, name, v, lsha in mine[:N_PER_AXIS]:
        out.append(dict(prompt=f"In the neural-mesh Rust crate, what is the value of `const {name}` in src/{file}? Give a plain decimal number (no underscores, no exponent notation)." + FMT2,
                        answer=v, kind="num", tol=1e-9, sub="private", src_file=f"src/{file}", const=name, src_line_sha=lsha, leak="by-construction"))
    return out, len(mine)


# =================================================================== item generation
def make_item(split, axis, idx, attempt):
    """one HELD/CONF item from its own rng stream (seed, axis, index, attempt) so a resample changes nothing else"""
    R = random.Random(f"{SEEDS[split]}/{axis}/{idx}/{attempt}")
    base = dict(id=f"{split[0]}-{ABBR[axis]}{idx + 1:02d}", split=split)
    if axis == "math":
        d = MATH[(idx + attempt) % len(MATH)](R); return dict(base, cat="math", **d)
    if axis == "physics":
        d = PHYS[(idx + attempt) % len(PHYS)](R); return dict(base, cat="physics", **d)
    if axis == "research_public":
        d = PUB[(idx + attempt) % len(PUB)](R); return dict(base, cat="research", **d)
    if axis == "code":
        params_fn, name, desc, ref_t, fixed, rnd = CODE[(idx + attempt) % len(CODE)]
        prm = params_fn(R)
        fmt = lambda s: s.format(**prm)
        ref = fmt(ref_t)
        tests = json.loads(fmt(fixed if isinstance(fixed, str) else json.dumps(fixed))) + [rnd(R) for _ in range(4)]
        tests = [json.loads(json.dumps(t)) for t in tests]
        g = {}
        exec(ref, g)            # trusted reference, run here once; the sandbox only ever runs model code
        expected = [sandbox.canon(g[name](*json.loads(json.dumps(t)))) for t in tests]
        prompt = f"Write a Python 3 function `{name}` with this behaviour: {fmt(desc)}" + CODE_FMT
        return dict(base, cat="code", sub="func", prompt=prompt, answer=None, kind="code", fn=name, ref=ref, tests=tests, expected=expected)
    raise ValueError(axis)


def answer_needles(item):
    """strings whose presence in retrieved context counts as a leak (short/common values are too noisy to count)"""
    if item["kind"] == "code": return [f"def {item['fn']}("]
    a = item["answer"]
    if item["kind"] == "str": return [str(a)] if len(str(a)) >= 4 else []
    v = float(a); out = set()
    if v.is_integer():
        s = str(int(v))
        if len(s.lstrip("-")) >= 3: out.add(s)
    else:
        for k in range(2, 9):
            s = f"{v:.{k}f}".rstrip("0")
            if len(re.sub(r"\D", "", s)) >= 4: out.add(s)
        out.add(repr(v))
    return sorted(out)


def leaks(item, fp):
    new, meta = fp(item["prompt"])
    if not meta.get("retrieved"): return False
    ctx = new[: max(0, len(new) - len(item["prompt"]))]
    for s in answer_needles(item):
        if re.search(r"(?<![\w.])" + re.escape(s) + r"(?![\w])", ctx, re.I if item["kind"] == "str" else 0): return True
    return False


class _FrontendConfig:
    """configure the imported front-end module to the pre-registered variant for the duration of the leak check"""
    def __init__(self, env): self.env = env
    def __enter__(self):
        import frontend
        self.f = frontend
        self.old = (frontend.V2, frontend.HOOK_FE, frontend.THRESH)
        frontend.V2 = self.env.get("MESH_V2", "") == "1"; frontend.HOOK_FE = self.env.get("MESH_FRONTEND", "") == "hook"
        frontend.THRESH = float(self.env.get("MESH_THRESH") or "30.0")
        return frontend.frontend_prompt
    def __exit__(self, *a):
        self.f.V2, self.f.HOOK_FE, self.f.THRESH = self.old


def build_split(split, fp=None, pool=None, taken_prompts=(), make=None, max_resample=MAX_RESAMPLE):
    """-> (items, report). fp = frontend_prompt-like callable used for the leak check (default: the pre-registered front-end)."""
    if split not in ("held", "conf"): raise ValueError(split)
    if fp is None:
        with _FrontendConfig(FRONTEND_ENV) as f:
            return build_split(split, f, pool, taken_prompts, make, max_resample)
    make = make or make_item
    seen = set(taken_prompts); items = []; resampled = Counter(); retrieved = Counter()
    for axis in ("math", "physics", "code", "research_public"):
        for idx in range(N_PER_AXIS):
            for attempt in range(max_resample + 1):
                it = make(split, axis, idx, attempt)
                bad = it["prompt"] in seen or leaks(it, fp)
                if not bad: break
                resampled[axis] += 1
            else:
                raise BuildError(f"{split} {axis}[{idx}]: still leaking or duplicate after {max_resample} resamples")
            seen.add(it["prompt"]); items.append(dict(it, leak="checked"))
    priv, pool_n = private_items(split, pool)
    for idx, d in enumerate(priv):
        it = dict(id=f"{split[0]}-{ABBR['research_private']}{idx + 1:02d}", split=split, cat="research", **d)
        if it["prompt"] in seen: raise BuildError(f"{split}: duplicate private prompt {it['id']}")
        retrieved["research_private"] += bool(fp(it["prompt"])[1].get("retrieved"))
        seen.add(it["prompt"]); items.append(it)
    counts = Counter(axis_of(i) for i in items)
    return items, dict(counts=dict(counts), shortfall={a: N_PER_AXIS - counts.get(a, 0) for a in AXES if counts.get(a, 0) < N_PER_AXIS},
                       private_pool_for_split=pool_n, resampled=dict(resampled), private_retrieved=retrieved["research_private"])


def tune_items():
    """the frozen legacy items exactly as battery.json holds them (bench/battery_tune.py)"""
    import importlib
    import battery_tune
    importlib.reload(battery_tune)
    return battery_tune.items


def tune_annotated(items):
    """battery_tune.json view of the legacy items: split, axis-level provenance and precomputed expected outputs for code"""
    out = []
    srcmap = dict((n, f) for f, n in TUNE_CONSTS)
    for it in items:
        it = dict(it, split="tune")
        if it["kind"] == "code":
            g = {}
            exec(it["ref"], g)
            exp = []
            for t in it["tests"]:
                try: exp.append(sandbox.canon(g[it["fn"]](*json.loads(json.dumps(t)))))
                except Exception: exp.append("EXC")
            it["expected"] = exp
        if it["cat"] == "research" and it["sub"] == "private":
            name = re.search(r"`pub const (\w+)`", it["prompt"]).group(1)
            it.update(src_file=f"src/{srcmap[name]}.rs", const=name, leak="by-construction")
        out.append(it)
    return out


def header_for(split, items, report):
    import frontend
    return dict(split=split, seed=SEEDS[split], n_items=len(items), items_sha=json_sha(items), corpus_sha=frontend.corpus_sha(), frontend_sha=frontend.frontend_sha(),
                frontend_env=FRONTEND_ENV, battery_py_sha=file_sha(__file__), battery_tune_py_sha=file_sha(ROOT / "battery_tune.py"), n_per_axis=N_PER_AXIS, **report)


def write_split(split, items, header):
    battery_path(split).write_text(json.dumps(dict(header=header, items=items), indent=1) + "\n", encoding="utf8", newline="\n")


def check_disjoint(splits):
    """splits: {name: items}. ids, prompts and private constants must not be shared between splits"""
    errs = []
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for key in ("id", "prompt"):
                common = {x[key] for x in splits[a]} & {x[key] for x in splits[b]}
                if common: errs.append(f"{a}/{b} share {len(common)} {key}s")
            ca = {(x.get("src_file"), x.get("const")) for x in splits[a] if x["sub"] == "private"}
            cb = {(x.get("src_file"), x.get("const")) for x in splits[b] if x["sub"] == "private"}
            if ca & cb: errs.append(f"{a}/{b} share private constants {sorted(ca & cb)[:3]}")
    return errs


def check_items(split, header, items):
    """[(level, msg)]: ("stale", ...) a private constant changed or vanished; ("warn", ...) corpus / front-end changed"""
    probs = []
    cache = {}
    for it in items:
        if it.get("sub") != "private": continue
        f = NM / it["src_file"]
        if f not in cache: cache[f] = scan_consts(f.read_text(encoding="utf8", errors="replace")) if f.exists() else {}
        name = it.get("const") or re.search(r"`(?:pub )?const (\w+)`", it["prompt"]).group(1)
        cur = cache[f].get(name)
        if cur is None: probs.append(("stale", f"{it['id']}: {it['src_file']}::{name} missing or ambiguous")); continue
        try: same = float(cur[0]) == float(it["answer"])
        except ValueError: same = cur[0] == it["answer"]
        if not same: probs.append(("stale", f"{it['id']}: {it['src_file']}::{name} = {cur[0]} (battery has {it['answer']})"))
    try:
        import frontend
        if header.get("corpus_sha") and header["corpus_sha"] != frontend.corpus_sha(): probs.append(("warn", f"{split}: retrieval corpus changed since the build"))
        if header.get("frontend_sha") and header["frontend_sha"] != frontend.frontend_sha(): probs.append(("warn", f"{split}: frontend.py changed since the build"))
    except Exception as e:
        probs.append(("warn", f"corpus check failed: {e!r}"))
    return probs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="battery.py")
    ap.add_argument("--split", default="all", choices=["tune", "held", "conf", "all"]); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    splits = ["tune", "held", "conf"] if a.split == "all" else [a.split]
    if a.check:
        rc = 0
        for s in splits:
            if not battery_path(s).exists(): print(f"{s}: {battery_path(s).name} missing"); rc = max(rc, 2); continue
            j = json.load(open(battery_path(s), encoding="utf8"))
            if json_sha(j["items"]) != j["header"]["items_sha"]: print(f"{s}: items_sha mismatch (file edited)"); rc = 2
            for level, msg in check_items(s, j["header"], j["items"]):
                print(f"{level}: {msg}")
                if level == "stale": rc = 2
        if "tune" in splits:
            fresh = json.dumps(tune_items(), indent=1)
            if (ROOT / "battery.json").read_text(encoding="utf8") != fresh: print("stale: battery.json differs from a fresh TUNE build"); rc = 2
        print("check:", "OK" if rc == 0 else "STALE"); return rc
    built = {}
    try:
        tune = tune_items()
        if "tune" in splits:
            json.dump(tune, open(ROOT / 'battery.json', 'w'), indent=1)       # legacy format, byte-identical to the frozen file
            ti = tune_annotated(tune)
            write_split("tune", ti, header_for("tune", ti, dict(counts=dict(Counter(axis_of(i) for i in ti)), legacy_battery_json_sha=file_sha(ROOT / "battery.json"))))
            built["tune"] = ti
        with _FrontendConfig(FRONTEND_ENV) as fp:
            pool = const_pool()
            taken = {i["prompt"] for i in tune}
            if "held" in splits or "conf" in splits:      # always HELD then CONF (CONF avoids HELD prompts): same result whatever was asked
                for s in ("held", "conf"):
                    items, rep = build_split(s, fp, pool, taken)
                    taken |= {i["prompt"] for i in items}; built[s] = items
                    if s not in splits: continue
                    write_split(s, items, header_for(s, items, rep))
                    print(f"{s}: {len(items)} items {rep['counts']} shortfall={rep['shortfall'] or 'none'} resampled={rep['resampled'] or 'none'} "
                          f"private pool={rep['private_pool_for_split']} private retrieved={rep['private_retrieved']}")
        errs = check_disjoint(dict(built, tune=built.get("tune") or tune))
        if errs: raise BuildError("splits not disjoint: " + "; ".join(errs))
    except BuildError as e:
        print(f"battery.py: {e}", file=sys.stderr); return 1
    if "tune" in splits: print("tune:", Counter((i['cat'], i['sub']) for i in tune), len(tune))
    return 0


if __name__ == "__main__":
    sys.exit(main())
