"""bench/battery_tune.py - the TUNE split: today's 120-item battery, FROZEN (code moved verbatim from battery.py; importing
this module builds `items`, writing is done by battery.py). v1/v2 front-ends were designed on these items, so they are tuning
evidence only. Deterministic (seed 20261010); `python bench/battery.py --split tune` must reproduce bench/battery.json
byte for byte (tests/test_bench.py::test_tune_identical).
research = 15 public (facts verified by executing the installed Python stdlib) + 15 private (constants read
from neural-mesh/src by regex at build time; these need retrieval over the repo)."""
import json, math, random, re, sys, hashlib, zlib, base64, http, http.client, calendar, datetime, errno, string, uuid, pathlib
from functools import reduce
R = random.Random(20261010)
ROOT = pathlib.Path(__file__).resolve().parent
NM = ROOT.parent
FMT = "\n\nGive the final answer as a single number (no units) on the last line in the form `ANSWER: <number>`."
FMT2 = "\n\nGive the final answer on the last line in the form `ANSWER: <answer>` (a number, or a short exact string)."
items = []


def add(cat, sub, prompt, answer, kind, **kw):
    n = sum(1 for i in items if i["cat"] == cat) + 1
    items.append(dict(id=f"{cat[:2]}{n:02d}", cat=cat, sub=sub, prompt=prompt, answer=answer, kind=kind, **kw))


def isprime(n):
    return n > 1 and all(n % p for p in range(2, int(n ** .5) + 1))


def base7(n):
    d = []
    while n:
        d.append(str(n % 7)); n //= 7
    return int("".join(reversed(d)))


def fib(n):
    return reduce(lambda ab, _: (ab[1], ab[0] + ab[1]), range(n), (0, 1))[0]


xs = [4, 8, 15, 16, 23, 42]
mu_x = sum(xs) / 6
# ---------------- math (30)
M = [
    ("Compute 7^222 mod 1000.", pow(7, 222, 1000)),
    ("What is 48271 * 93817?", 48271 * 93817),
    ("How many primes are less than 3000?", sum(isprime(n) for n in range(3000))),
    ("What is the sum of the decimal digits of 2^200?", sum(map(int, str(2 ** 200)))),
    ("What is gcd(123456789, 987654321)?", math.gcd(123456789, 987654321)),
    ("What is lcm(84, 126, 198)?", math.lcm(84, 126, 198)),
    ("How many ways can a 5-card hand be chosen from a 52-card deck?", math.comb(52, 5)),
    ("Convert 0xBEEF to decimal.", 0xBEEF),
    ("Convert decimal 123456 to base 7. Give the base-7 digits written as a plain integer (digits 1,2,3 -> 123).", base7(123456)),
    ("What is 17! / 15! ?", math.factorial(17) // math.factorial(15)),
    ("What is the sum of all integers from 1 to 1000 that are divisible by 3 or 5?", sum(n for n in range(1, 1001) if n % 3 == 0 or n % 5 == 0)),
    ("What is the 30th Fibonacci number, with F1=1, F2=1?", fib(30)),
    ("What is the remainder when 3^1000 is divided by 13?", pow(3, 1000, 13)),
    ("How many positive divisors does 720720 have?", sum(1 for d in range(1, 720721) if 720720 % d == 0)),
    ("What is the sum of the first 50 odd positive integers?", 50 * 50),
    ("What is the determinant of the matrix [[2,3,1],[4,1,5],[7,2,6]]?", 2 * (1 * 6 - 5 * 2) - 3 * (4 * 6 - 5 * 7) + 1 * (4 * 2 - 1 * 7)),
    ("Solve for x: 3x + 7 = 2x - 5 + 4x. Give x as a decimal.", 4.0),
    ("A fair coin is flipped 10 times. What is the probability of exactly 4 heads? Give a decimal rounded to 5 places.", round(math.comb(10, 4) / 1024, 5)),
    ("What is the value of the integral of x^3 from 0 to 4?", 64.0),
    ("What is the derivative of f(x)=x^4 - 3x^2 + 2x at x=3?", 4 * 27 - 6 * 3 + 2),
    ("Compound interest: 5000 invested at 6% annual interest compounded monthly for 7 years. Final amount rounded to 2 decimals?", round(5000 * (1 + .06 / 12) ** 84, 2)),
    ("What is the sum of the series 1/2 + 1/4 + ... + 1/1024 as a decimal to 8 places?", round(sum(1 / 2 ** k for k in range(1, 11)), 8)),
    ("How many trailing zeros does 100! have?", sum(100 // 5 ** k for k in range(1, 4))),
    ("How many ordered triples (x,y,z) of non-negative integers satisfy x + y + z = 20?", math.comb(22, 2)),
    ("What is the square root of 2 to 10 decimal places?", round(math.sqrt(2), 10)),
    ("What is 987654321 mod 97?", 987654321 % 97),
    ("What is the largest prime factor of 600851475143?", 6857),
    ("What is the sum of squares of the integers from 1 to 60?", sum(k * k for k in range(1, 61))),
    ("What is the population standard deviation of the numbers 4, 8, 15, 16, 23, 42? Round to 4 decimals.", round((sum((x - mu_x) ** 2 for x in xs) / 6) ** .5, 4)),
    ("How many distinct permutations are there of the letters of MISSISSIPPI?", math.factorial(11) // (math.factorial(4) * math.factorial(4) * math.factorial(2))),
]
assert len(M) == 30
for q, a in M:
    isint = float(a).is_integer()
    add("math", "exact" if isint else "dec", q + FMT, a, "num", tol=0 if isint else 1e-6)

# ---------------- physics (30): generated parameters, SI, rel tol 1%
g = 9.80665; c = 299792458; G = 6.674e-11; h = 6.62607015e-34; Rg = 8.314462618; e = 1.602176634e-19; Me = 5.972e24; Re = 6.371e6


def r(a, b, n=2):
    return round(R.uniform(a, b), n)


P = []
v = r(10, 40, 0); th = R.choice([30, 35, 40, 45, 50, 55, 60])
P.append((f"A projectile is launched on level ground at {v} m/s at {th} degrees above horizontal, no air resistance, g=9.80665 m/s^2. What is its horizontal range in metres?", v * v * math.sin(2 * math.radians(th)) / g))
m = r(1, 20, 1); vv = r(2, 30, 1)
P.append((f"What is the kinetic energy in joules of a {m} kg mass moving at {vv} m/s?", .5 * m * vv * vv))
Rr = [r(10, 100, 0) for _ in range(3)]
P.append((f"Three resistors of {Rr[0]}, {Rr[1]} and {Rr[2]} ohms are connected in parallel. What is the equivalent resistance in ohms?", 1 / sum(1 / x for x in Rr)))
n = r(.5, 3, 2); T = r(250, 400, 0); V = r(.01, .1, 3)
P.append((f"{n} mol of ideal gas at {T} K occupy {V} m^3. Using R=8.314462618 J/(mol K), what is the pressure in pascals?", n * Rg * T / V))
f = r(5, 30, 0); do = r(40, 120, 0)
P.append((f"A thin converging lens has focal length {f} cm. An object is {do} cm from the lens. At what distance in cm (positive means real image) is the image formed?", 1 / (1 / f - 1 / do)))
vf = r(.3, .95, 2)
P.append((f"A particle moves at {vf}c. What is its Lorentz factor gamma?", 1 / math.sqrt(1 - vf * vf)))
P.append(("What is the escape velocity from Earth's surface in m/s, using G=6.674e-11, M=5.972e24 kg, R=6.371e6 m?", math.sqrt(2 * G * Me / Re)))
L = r(.3, 3, 2)
P.append((f"What is the period in seconds of a simple pendulum of length {L} m for small oscillations, g=9.80665 m/s^2?", 2 * math.pi * math.sqrt(L / g)))
fs = r(300, 900, 0); vs = r(10, 40, 0)
P.append((f"A source emitting {fs} Hz moves toward a stationary observer at {vs} m/s. Speed of sound is 343 m/s. What frequency in Hz does the observer hear?", fs * 343 / (343 - vs)))
Rc = r(1, 50, 1) * 1000; C = r(1, 100, 0) * 1e-6
P.append((f"An RC circuit has R={Rc} ohms and C={C} F. What is the time constant in seconds?", Rc * C))
lam = r(300, 800, 0)
P.append((f"What is the energy in joules of a photon of wavelength {lam} nm? (h=6.62607015e-34, c=299792458)", h * c / (lam * 1e-9)))
a = r(6.5e6, 4.2e7, 0)
P.append((f"What is the orbital period in seconds of a satellite in a circular orbit of radius {a} m around Earth (GM = 3.986004418e14 m^3/s^2)?", 2 * math.pi * math.sqrt(a ** 3 / 3.986004418e14)))
m1 = r(1, 10, 1); v1 = r(5, 20, 1); m2 = r(1, 10, 1)
P.append((f"A {m1} kg mass moving at {v1} m/s collides perfectly inelastically with a stationary {m2} kg mass. What is the final speed in m/s?", m1 * v1 / (m1 + m2)))
I = r(.5, 5, 1); Vv = r(5, 48, 0)
P.append((f"A {Vv} V source drives {I} A through a resistor. How much power in watts is dissipated?", Vv * I))
Tc = r(300, 6000, 0)
P.append((f"By Wien's displacement law (b=2.897771955e-3 m K), what is the peak wavelength in nanometres of a black body at {Tc} K?", 2.897771955e-3 / Tc * 1e9))
dh = r(5, 100, 0)
P.append((f"A ball is dropped from {dh} m with no air resistance (g=9.80665). With what speed in m/s does it hit the ground?", math.sqrt(2 * g * dh)))
mk = r(.1, 5, 1); k = r(50, 500, 0)
P.append((f"A mass of {mk} kg hangs on a spring of constant {k} N/m. What is the angular frequency in rad/s?", math.sqrt(k / mk)))
d = r(.5, 5, 1); q1 = r(1, 10, 1); q2 = r(1, 10, 1)
P.append((f"Two point charges of {q1} microcoulomb and {q2} microcoulomb are {d} m apart in vacuum. Coulomb constant 8.9875517923e9. What is the magnitude of the force in newtons?", 8.9875517923e9 * q1 * 1e-6 * q2 * 1e-6 / d ** 2))
Vb = r(1e-4, 1e-2, 5)
P.append((f"What is the buoyant force in newtons on an object fully submerged in water (density 1000 kg/m^3) with volume {Vb} m^3, g=9.80665?", 1000 * Vb * g))
ang = R.choice([20, 25, 30, 35, 40]); mu = r(.1, .3, 2)
P.append((f"A block slides down a rough incline of {ang} degrees with kinetic friction coefficient {mu}, g=9.80665. What is its acceleration in m/s^2?", g * (math.sin(math.radians(ang)) - mu * math.cos(math.radians(ang)))))
Ts = r(280, 330, 0)
P.append((f"What is the rms speed in m/s of nitrogen molecules (molar mass 0.0280134 kg/mol) at {Ts} K? (R=8.314462618)", math.sqrt(3 * Rg * Ts / 0.0280134)))
mw = r(.2, 2, 1)
P.append((f"How much heat in joules is needed to raise {mw} kg of water by 25 K? Use c=4186 J/(kg K).", mw * 4186 * 25))
Lw = r(.5, 2, 2); Fq = R.choice([1, 2, 3])
P.append((f"What is the frequency in Hz of harmonic number {Fq + 1} on a string fixed at both ends, of length {Lw} m, with wave speed 340 m/s?", (Fq + 1) * 340 / (2 * Lw)))
B = r(.1, 2, 2); vq = r(1e5, 1e7, 0)
P.append((f"A proton (m=1.67262192e-27 kg, q=1.602176634e-19 C) moves at {vq} m/s perpendicular to a {B} T magnetic field. What is the radius of its circular path in metres?", 1.67262192e-27 * vq / (e * B)))
n2 = r(1.3, 1.9, 2); a1 = r(10, 60, 0)
P.append((f"Light passes from air (n=1.0) into glass with n={n2} at an incidence angle of {a1} degrees. What is the refraction angle in degrees (Snell's law)?", math.degrees(math.asin(math.sin(math.radians(a1)) / n2))))
rho = r(7000, 9000, 0); Vm = r(1e-4, 1e-3, 5)
P.append((f"What is the weight in newtons (g=9.80665) of a metal block with density {rho} kg/m^3 and volume {Vm} m^3?", rho * Vm * g))
t12 = r(2, 30, 1); tt = r(5, 100, 0)
P.append((f"A radioactive sample has half-life {t12} days. What fraction remains after {tt} days? Give a decimal.", .5 ** (tt / t12)))
Pw = r(100, 2000, 0); tm = r(10, 3600, 0)
P.append((f"A {Pw} W heater runs for {tm} s. How much energy in kilojoules does it deliver?", Pw * tm / 1000))
Aw = r(.001, .1, 4); v3 = r(1, 8, 1)
P.append((f"Water (density 1000 kg/m^3) flows at {v3} m/s through a pipe of cross-section {Aw} m^2. What is the mass flow rate in kg/s?", 1000 * v3 * Aw))
Ci = r(1, 100, 0) * 1e-6; Vc = r(5, 50, 0)
P.append((f"What energy in joules is stored in a {Ci} F capacitor charged to {Vc} V?", .5 * Ci * Vc * Vc))
Ee = r(.5, 5, 2)
P.append((f"What is the de Broglie wavelength in picometres of an electron accelerated through {Ee * 1000} V (non-relativistic; m=9.1093837e-31 kg, h=6.62607015e-34, e=1.602176634e-19)?", h / math.sqrt(2 * 9.1093837e-31 * e * Ee * 1000) * 1e12))
P = P[:30]
assert len(P) == 30, len(P)
for q, a_ in P:
    add("physics", "numeric", q + FMT, a_, "num", tol=0.01)

# ---------------- code (30): function spec + hidden tests vs reference
C = [
    ("rev_words", "Return the string with the order of whitespace-separated words reversed, joined by single spaces.", "def rev_words(s): return ' '.join(s.split()[::-1])", [("a b c",), ("  hello   world ",), ("",), ("one",)]),
    ("is_palindrome", "Return True if the string is a palindrome ignoring case and non-alphanumeric characters.", "def is_palindrome(s):\n    t=[c.lower() for c in s if c.isalnum()]\n    return t==t[::-1]", [("A man, a plan, a canal: Panama",), ("race a car",), ("",), ("0P",)]),
    ("two_sum", "Given a list of ints and a target, return the pair of indices (i,j) with i<j and nums[i]+nums[j]==target, choosing the smallest i then the smallest j; return None if no pair.", "def two_sum(n,t):\n    for i in range(len(n)):\n        for j in range(i+1,len(n)):\n            if n[i]+n[j]==t: return (i,j)\n    return None", [([2, 7, 11, 15], 9), ([3, 2, 4], 6), ([1, 2], 7), ([], 0)]),
    ("fizz", "Return the list of strings for 1..n using FizzBuzz rules (Fizz for multiples of 3, Buzz for 5, FizzBuzz for both, else the number as a string).", "def fizz(n): return ['FizzBuzz' if i%15==0 else 'Fizz' if i%3==0 else 'Buzz' if i%5==0 else str(i) for i in range(1,n+1)]", [(15,), (1,), (0,), (5,)]),
    ("flatten", "Flatten an arbitrarily nested list of ints into one flat list, preserving order.", "def flatten(x):\n    o=[]\n    for i in x:\n        o+= flatten(i) if isinstance(i,list) else [i]\n    return o", [([1, [2, [3, [4]]], 5],), ([],), ([[[]]],), ([1, 2, 3],)]),
    ("run_length", "Run-length encode a string as char followed by count, e.g. 'aaab' -> 'a3b1'; the empty string gives ''.", "def run_length(s):\n    o=''\n    i=0\n    while i<len(s):\n        j=i\n        while j<len(s) and s[j]==s[i]: j+=1\n        o+=s[i]+str(j-i); i=j\n    return o", [("aaabcc",), ("",), ("z",), ("aabbaa",)]),
    ("valid_parens", "Return True if the string of ()[]{} brackets is balanced and properly nested.", "def valid_parens(s):\n    st=[];m={')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in '([{': st.append(c)\n        elif not st or st.pop()!=m[c]: return False\n    return not st", [("()[]{}",), ("([)]",), ("",), ("(((",), ("{[()]}",)]),
    ("merge_sorted", "Merge two sorted lists of ints into one sorted list.", "def merge_sorted(a,b): return sorted(a+b)", [([1, 3, 5], [2, 4, 6]), ([], [1]), ([1, 1], [1]), ([], [])]),
    ("anagram", "Return True if two strings are anagrams ignoring case and spaces.", "def anagram(a,b): return sorted(a.replace(' ','').lower())==sorted(b.replace(' ','').lower())", [("Dormitory", "Dirty room"), ("abc", "abd"), ("", ""), ("a b", "ba")]),
    ("max_subarray", "Return the maximum sum of a non-empty contiguous subarray.", "def max_subarray(a):\n    b=c=a[0]\n    for x in a[1:]:\n        c=max(x,c+x); b=max(b,c)\n    return b", [([-2, 1, -3, 4, -1, 2, 1, -5, 4],), ([-3],), ([1, 2, 3],), ([-2, -1],)]),
    ("roman", "Convert a positive integer (1..3999) to a Roman numeral string.", "def roman(n):\n    v=[(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')];o=''\n    for a,s in v:\n        while n>=a: o+=s; n-=a\n    return o", [(1994,), (4,), (3999,), (58,)]),
    ("from_roman", "Convert a valid Roman numeral string to an int.", "def from_roman(s):\n    v={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000};t=0\n    for i,c in enumerate(s):\n        t+= -v[c] if i+1<len(s) and v[c]<v[s[i+1]] else v[c]\n    return t", [("MCMXCIV",), ("IV",), ("MMMCMXCIX",), ("LVIII",)]),
    ("primes_upto", "Return the list of primes <= n in ascending order.", "def primes_upto(n): return [p for p in range(2,n+1) if all(p%d for d in range(2,int(p**.5)+1))]", [(30,), (1,), (2,), (100,)]),
    ("collatz_len", "Return the number of steps for n to reach 1 in the Collatz sequence (n=1 gives 0).", "def collatz_len(n):\n    s=0\n    while n!=1:\n        n= n//2 if n%2==0 else 3*n+1; s+=1\n    return s", [(1,), (6,), (27,), (97,)]),
    ("word_freq", "Return a dict of lowercase word -> count, splitting on whitespace and stripping the characters .,!?;: from both ends of each word; skip empty words.", "def word_freq(s):\n    d={}\n    for w in s.lower().split():\n        w=w.strip('.,!?;:')\n        if w: d[w]=d.get(w,0)+1\n    return d", [("The cat. The dog! the end",), ("",), ("a,b a",)]),
    ("rotate", "Rotate a list right by k positions (k may exceed the length or be 0); return a new list.", "def rotate(a,k):\n    if not a: return []\n    k%=len(a)\n    return a[-k:]+a[:-k] if k else a[:]", [([1, 2, 3, 4, 5], 2), ([1, 2, 3], 0), ([], 3), ([1, 2, 3], 7)]),
    ("dedupe", "Remove duplicates from a list keeping the first occurrence order.", "def dedupe(a):\n    s=set();o=[]\n    for x in a:\n        if x not in s: s.add(x); o.append(x)\n    return o", [([3, 1, 3, 2, 1],), ([],), (['a', 'b', 'a'],)]),
    ("binary_search", "Return the index of target in a sorted list of distinct ints, or -1.", "def binary_search(a,t):\n    lo,hi=0,len(a)-1\n    while lo<=hi:\n        m=(lo+hi)//2\n        if a[m]==t: return m\n        if a[m]<t: lo=m+1\n        else: hi=m-1\n    return -1", [([1, 3, 5, 7, 9], 7), ([1, 3, 5], 4), ([], 1), ([2], 2)]),
    ("caesar", "Caesar-shift letters by k (k may be negative), preserving case and leaving non-letters unchanged.", "def caesar(s,k):\n    o=''\n    for c in s:\n        if c.isalpha():\n            b=65 if c.isupper() else 97\n            o+=chr((ord(c)-b+k)%26+b)\n        else: o+=c\n    return o", [("Hello, World!", 3), ("abc", -1), ("xyz", 4), ("", 5)]),
    ("matrix_transpose", "Transpose a rectangular list-of-lists matrix; an empty matrix gives [].", "def matrix_transpose(m): return [list(r) for r in zip(*m)] if m else []", [([[1, 2, 3], [4, 5, 6]],), ([],), ([[1]],)]),
    ("longest_common_prefix", "Return the longest common prefix of a list of strings ('' if none or the list is empty).", "def longest_common_prefix(a):\n    if not a: return ''\n    p=a[0]\n    for s in a[1:]:\n        while not s.startswith(p): p=p[:-1]\n    return p", [(["flower", "flow", "flight"],), (["dog", "car"],), ([],), (["a"],)]),
    ("to_binary", "Return the binary representation of a non-negative int as a string without the 0b prefix.", "def to_binary(n): return bin(n)[2:]", [(0,), (5,), (255,), (1024,)]),
    ("count_vowels", "Count vowels (aeiou, case-insensitive) in a string.", "def count_vowels(s): return sum(c in 'aeiouAEIOU' for c in s)", [("Hello World",), ("",), ("rhythm",), ("AEIOUaeiou",)]),
    ("pascal_row", "Return row n (0-indexed) of Pascal's triangle as a list.", "def pascal_row(n):\n    r=[1]\n    for _ in range(n): r=[1]+[a+b for a,b in zip(r,r[1:])]+[1]\n    return r", [(0,), (1,), (5,), (10,)]),
    ("gcd_list", "Return the gcd of a non-empty list of positive ints.", "def gcd_list(a):\n    import math\n    g=a[0]\n    for x in a[1:]: g=math.gcd(g,x)\n    return g", [([12, 18, 24],), ([7],), ([5, 10, 15, 25],), ([17, 31],)]),
    ("interval_merge", "Merge overlapping closed intervals (list of [a,b]) and return the sorted merged list of [a,b] lists; touching intervals (a == previous b) merge.", "def interval_merge(iv):\n    o=[]\n    for a,b in sorted(iv):\n        if o and a<=o[-1][1]: o[-1][1]=max(o[-1][1],b)\n        else: o.append([a,b])\n    return o", [([[1, 3], [2, 6], [8, 10], [15, 18]],), ([[1, 4], [4, 5]],), ([],), ([[5, 6], [1, 2]],)]),
    ("is_leap", "Return True if a Gregorian year is a leap year.", "def is_leap(y): return y%4==0 and (y%100!=0 or y%400==0)", [(1900,), (2000,), (2024,), (2023,)]),
    ("camel_to_snake", "Convert CamelCase / camelCase to snake_case: lowercase everything, and insert an underscore before each uppercase letter that is not the first character.", "def camel_to_snake(s):\n    o=''\n    for i,c in enumerate(s):\n        if c.isupper() and i>0: o+='_'\n        o+=c.lower()\n    return o", [("camelCase",), ("CamelCaseString",), ("x",), ("",)]),
    ("lru_sim", "Simulate an LRU cache of capacity cap over a list of integer keys accessed in order (a miss inserts the key, evicting the least recently used when over capacity; a hit refreshes recency). Return the number of hits.", "def lru_sim(keys,cap):\n    from collections import OrderedDict\n    d=OrderedDict();h=0\n    for k in keys:\n        if k in d: h+=1; d.move_to_end(k)\n        else:\n            d[k]=1\n            if len(d)>cap: d.popitem(last=False)\n    return h", [([1, 2, 1, 3, 1, 2], 2), ([1, 1, 1], 1), ([], 3), ([1, 2, 3, 4, 1, 2, 3, 4], 3)]),
    ("topo_order", "Given n nodes 0..n-1 and a list of directed edges (u,v) meaning u must come before v, return the lexicographically smallest topological order as a list, or None if there is a cycle.", "def topo_order(n,e):\n    import heapq\n    ind=[0]*n;g=[[] for _ in range(n)]\n    for u,v in e: g[u].append(v); ind[v]+=1\n    h=[i for i in range(n) if ind[i]==0];heapq.heapify(h);o=[]\n    while h:\n        u=heapq.heappop(h);o.append(u)\n        for v in g[u]:\n            ind[v]-=1\n            if ind[v]==0: heapq.heappush(h,v)\n    return o if len(o)==n else None", [(4, [(0, 1), (2, 1), (1, 3)]), (3, [(0, 1), (1, 2), (2, 0)]), (3, []), (5, [(4, 0), (3, 0), (0, 1)])]),
]
assert len(C) == 30, len(C)
for name, desc, ref, tests in C:
    add("code", "func", f"Write a Python 3 function `{name}` with this behaviour: {desc}\nReturn ONLY the code in a single ```python block (no explanation, no tests).", None, "code", fn=name, ref=ref, tests=[list(t) for t in tests])

# ---------------- research
PUB = [
    ("What are the first 8 hex characters of the SHA-256 digest of the ASCII bytes of 'abc'?", hashlib.sha256(b'abc').hexdigest()[:8]),
    ("What is the CRC-32 (zlib.crc32, unsigned) of the ASCII bytes of 'hello', as a decimal integer?", zlib.crc32(b'hello')),
    ("In Python's http.HTTPStatus, what is the phrase for status code 418?", http.HTTPStatus(418).phrase),
    ("What is the standard base64 encoding of the ASCII text 'Many hands'?", base64.b64encode(b'Many hands').decode()),
    ("What day of the week is 2026-10-10? Answer with the English weekday name.", datetime.date(2026, 10, 10).strftime('%A')),
    ("In Python on Linux, what is the integer value of errno.ENOENT?", errno.ENOENT),
    ("What is the Unicode code point (decimal) of the euro sign character?", ord('\u20ac')),
    ("What are the first 8 hex characters of the MD5 digest of the ASCII bytes of 'The quick brown fox'?", hashlib.md5(b'The quick brown fox').hexdigest()[:8]),
    ("How many characters are in Python's string.punctuation?", len(string.punctuation)),
    ("What is the Adler-32 checksum (zlib.adler32, unsigned) of b'Wikipedia' as a decimal integer?", zlib.adler32(b'Wikipedia')),
    ("What does Python's repr() give for the float expression 0.1 + 0.2?", repr(0.1 + 0.2)),
    ("What is the reason phrase for HTTP status 451 in Python's http.client.responses?", http.client.responses[451] if 451 in http.client.responses else http.HTTPStatus(451).phrase),
    ("What is repr(sys.float_info.max) in CPython?", repr(sys.float_info.max)),
    ("What is str(uuid.NAMESPACE_DNS) in Python's uuid module?", str(uuid.NAMESPACE_DNS)),
    ("How many days does February 2100 have in the Gregorian calendar?", calendar.monthrange(2100, 2)[1]),
]
for q, a_ in PUB:
    add("research", "public", q + FMT2, str(a_), "num" if isinstance(a_, int) else "str", tol=0)
consts = {}
for fn in ['cognition', 'compact', 'fin', 'flatq', 'hebbian', 'mesh', 'popguard', 'scan']:
    p = NM / 'src' / f'{fn}.rs'
    for mm in re.finditer(r'^pub const ([A-Z_0-9]+): [a-z0-9]+ = ([0-9_\.e\-]+);', p.read_text(encoding='utf8'), re.M):
        consts[(fn, mm.group(1))] = mm.group(2).replace('_', '')
WANT = [('cognition', 'N_DOMAINS'), ('compact', 'MAX_AGENTS'), ('fin', 'MAX_LIST'), ('fin', 'MAX_PERIODS'), ('fin', 'R_GAS'), ('flatq', 'FLAT_HNSW_CROSSOVER_N'), ('hebbian', 'INIT_WEIGHT'), ('hebbian', 'INIT_BASELINE'), ('mesh', 'DEFAULT_HELPER_EPSILON'), ('mesh', 'EDGE_BASELINE_ALPHA'), ('mesh', 'INIT_EDGE_BASELINE'), ('mesh', 'DISPATCH_RISK_ALPHA'), ('popguard', 'RECOMMENDED_INF_MIN'), ('scan', 'PAR_MIN_ROWS'), ('fin', 'RTOL')]
for fn, nme in WANT:
    v_ = consts[(fn, nme)]
    add("research", "private", f"In the neural-mesh Rust crate, what is the value of `pub const {nme}` in src/{fn}.rs? Give a plain decimal number (no underscores, no exponent notation)." + FMT2, v_, "num", tol=1e-9)
assert sum(1 for i in items if i['cat'] == 'research') == 30
