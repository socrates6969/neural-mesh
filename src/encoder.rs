//! R101: pure-Rust query encoder for all-MiniLM-L6-v2 (384-d): BERT WordPiece tokenizer + 6-layer transformer + mean pool + L2 normalise. No dependencies beyond std/serde_json,
//! so it builds on the Windows/GNU toolchain where `ort` / `tokenizers` (C deps) do not. Weights come from `py/b1/r101_tools/export_weights.py` (flat f32 file, matrices [out][in]).
//! Optional int8 mode (`with_int8(true)`): per-output-row symmetric int8 weights, per-token dynamic int8 activations, the AVX2 integer dot of `scan` (R91). Opt-in; nothing else uses this module.
//! Limits: ASCII-focused text normalisation (Latin-1 accents are folded, other scripts are passed through without NFD), max 128 tokens, single sequence (no padding / batching).
use std::collections::HashMap;

const D: usize = 384;
const H: usize = 12;
const DH: usize = 32;
const FF: usize = 1536;
const L: usize = 6;
const MAXLEN: usize = 128;

struct Lin {
    w: Vec<f32>,
    b: Vec<f32>,
    q: Vec<i8>,
    qs: Vec<f32>,
    out: usize,
    inn: usize,
}
struct Layer {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln1: (Vec<f32>, Vec<f32>),
    f1: Lin,
    f2: Lin,
    ln2: (Vec<f32>, Vec<f32>),
}
pub struct Encoder {
    vocab: HashMap<String, u32>,
    word: Vec<f32>,
    pos: Vec<f32>,
    typ: Vec<f32>,
    eln: (Vec<f32>, Vec<f32>),
    layers: Vec<Layer>,
    int8: bool,
}

#[derive(Debug)]
pub struct EncoderError(pub String);
impl std::fmt::Display for EncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for EncoderError {}


/// 4 tokens against one weight row at once (weight loaded once): f32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot4_f32_avx2(w: &[f32], x: [&[f32]; 4]) -> [f32; 4] {
    use std::arch::x86_64::*;
    let n = w.len();
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut acc2 = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 16 <= n {
        let wv = _mm256_loadu_ps(w.as_ptr().add(i));
        let wv2 = _mm256_loadu_ps(w.as_ptr().add(i + 8));
        for j in 0..4 {
            acc[j] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x[j].as_ptr().add(i)), acc[j]);
            acc2[j] = _mm256_fmadd_ps(wv2, _mm256_loadu_ps(x[j].as_ptr().add(i + 8)), acc2[j]);
        }
        i += 16;
    }
    for j in 0..4 {
        acc[j] = _mm256_add_ps(acc[j], acc2[j]);
    }
    while i + 8 <= n {
        let wv = _mm256_loadu_ps(w.as_ptr().add(i));
        for j in 0..4 {
            acc[j] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x[j].as_ptr().add(i)), acc[j]);
        }
        i += 8;
    }
    let mut out = [0f32; 4];
    for j in 0..4 {
        let mut t = [0f32; 8];
        _mm256_storeu_ps(t.as_mut_ptr(), acc[j]);
        out[j] = t.iter().sum();
    }
    while i < n {
        for j in 0..4 {
            out[j] += w[i] * x[j][i];
        }
        i += 1;
    }
    out
}
/// 4 tokens against one int8 weight row: i32 dots
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot4_i8_avx2(w: &[i8], x: [&[i8]; 4]) -> [i32; 4] {
    use std::arch::x86_64::*;
    let n = w.len();
    let mut acc = [_mm256_setzero_si256(); 4];
    let mut acc2 = [_mm256_setzero_si256(); 4];
    let mut i = 0;
    while i + 32 <= n {
        let wv = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(i) as *const __m128i));
        let wv2 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(i + 16) as *const __m128i));
        for j in 0..4 {
            let xv = _mm256_cvtepi8_epi16(_mm_loadu_si128(x[j].as_ptr().add(i) as *const __m128i));
            let xv2 = _mm256_cvtepi8_epi16(_mm_loadu_si128(x[j].as_ptr().add(i + 16) as *const __m128i));
            acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(wv, xv));
            acc2[j] = _mm256_add_epi32(acc2[j], _mm256_madd_epi16(wv2, xv2));
        }
        i += 32;
    }
    for j in 0..4 {
        acc[j] = _mm256_add_epi32(acc[j], acc2[j]);
    }
    while i + 16 <= n {
        let wv = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(i) as *const __m128i));
        for j in 0..4 {
            let xv = _mm256_cvtepi8_epi16(_mm_loadu_si128(x[j].as_ptr().add(i) as *const __m128i));
            acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(wv, xv));
        }
        i += 16;
    }
    let mut out = [0i32; 4];
    for j in 0..4 {
        let mut t = [0i32; 8];
        _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, acc[j]);
        out[j] = t.iter().sum();
    }
    while i < n {
        for j in 0..4 {
            out[j] += w[i] as i32 * x[j][i] as i32;
        }
        i += 1;
    }
    out
}
/// 2 weight rows x 4 tokens (8 accumulators; 6 loads per 8 FMAs): f32
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot24_f32_avx2(w0: &[f32], w1: &[f32], x: [&[f32]; 4]) -> [[f32; 4]; 2] {
    use std::arch::x86_64::*;
    let n = w0.len();
    let mut a0 = [_mm256_setzero_ps(); 4];
    let mut a1 = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 8 <= n {
        let v0 = _mm256_loadu_ps(w0.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(w1.as_ptr().add(i));
        for j in 0..4 {
            let xv = _mm256_loadu_ps(x[j].as_ptr().add(i));
            a0[j] = _mm256_fmadd_ps(v0, xv, a0[j]);
            a1[j] = _mm256_fmadd_ps(v1, xv, a1[j]);
        }
        i += 8;
    }
    let mut out = [[0f32; 4]; 2];
    for j in 0..4 {
        let mut t = [0f32; 8];
        _mm256_storeu_ps(t.as_mut_ptr(), a0[j]);
        out[0][j] = t.iter().sum();
        _mm256_storeu_ps(t.as_mut_ptr(), a1[j]);
        out[1][j] = t.iter().sum();
    }
    while i < n {
        for j in 0..4 {
            out[0][j] += w0[i] * x[j][i];
            out[1][j] += w1[i] * x[j][i];
        }
        i += 1;
    }
    out
}
/// 2 int8 weight rows x 4 tokens
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot24_i8_avx2(w0: &[i8], w1: &[i8], x: [&[i8]; 4]) -> [[i32; 4]; 2] {
    use std::arch::x86_64::*;
    let n = w0.len();
    let mut a0 = [_mm256_setzero_si256(); 4];
    let mut a1 = [_mm256_setzero_si256(); 4];
    let mut i = 0;
    while i + 16 <= n {
        let v0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w0.as_ptr().add(i) as *const __m128i));
        let v1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w1.as_ptr().add(i) as *const __m128i));
        for j in 0..4 {
            let xv = _mm256_cvtepi8_epi16(_mm_loadu_si128(x[j].as_ptr().add(i) as *const __m128i));
            a0[j] = _mm256_add_epi32(a0[j], _mm256_madd_epi16(v0, xv));
            a1[j] = _mm256_add_epi32(a1[j], _mm256_madd_epi16(v1, xv));
        }
        i += 16;
    }
    let mut out = [[0i32; 4]; 2];
    for j in 0..4 {
        let mut t = [0i32; 8];
        _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, a0[j]);
        out[0][j] = t.iter().sum();
        _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, a1[j]);
        out[1][j] = t.iter().sum();
    }
    while i < n {
        for j in 0..4 {
            out[0][j] += w0[i] as i32 * x[j][i] as i32;
            out[1][j] += w1[i] as i32 * x[j][i] as i32;
        }
        i += 1;
    }
    out
}
fn avx_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    }
    #[allow(unreachable_code)]
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let (mut s0, mut s1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0;
    while i + 16 <= n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)), _mm256_loadu_ps(b.as_ptr().add(i + 8)), s1);
        i += 16;
    }
    let s = _mm256_add_ps(s0, s1);
    let mut t = [0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), s);
    let mut r: f32 = t.iter().sum();
    while i < n {
        r += a[i] * b[i];
        i += 1;
    }
    r
}
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: features checked at runtime
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// branch-free exp (relative error ~2e-7 on [-80, 80]); lets the activation loops vectorise
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    let x = x.clamp(-80.0, 80.0);
    let n = (x * std::f32::consts::LOG2_E + 12582912.0) - 12582912.0;
    let r = x - n * std::f32::consts::LN_2;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * (0.041666668 + r * (0.008333334 + r * 0.0013888889)))));
    f32::from_bits(((n as i32 + 127) as u32) << 23) * p
}

#[inline(always)]
fn erf(x: f32) -> f32 {
    // Abramowitz-Stegun 7.1.26 (max abs error 1.5e-7)
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * fast_exp(-x * x);
    s * y
}
#[inline(always)]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_all_avx2(v: &mut [f32]) {
    for z in v.iter_mut() {
        *z = gelu(*z);
    }
}
fn gelu_all(v: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx_ok() {
            // SAFETY: avx2+fma checked
            return unsafe { gelu_all_avx2(v) };
        }
    }
    for z in v.iter_mut() {
        *z = gelu(*z);
    }
}

fn layer_norm(x: &mut [f32], g: &[f32], b: &[f32]) {
    let n = x.len() as f32;
    let m = x.iter().sum::<f32>() / n;
    let v = x.iter().map(|t| (t - m) * (t - m)).sum::<f32>() / n;
    let inv = 1.0 / (v + 1e-12).sqrt();
    for i in 0..x.len() {
        x[i] = (x[i] - m) * inv * g[i] + b[i];
    }
}

impl Lin {
    fn new(w: Vec<f32>, b: Vec<f32>, out: usize, inn: usize) -> Lin {
        Lin { w, b, q: Vec::new(), qs: Vec::new(), out, inn }
    }
    fn quantize(&mut self) {
        self.q.clear();
        self.qs.clear();
        for o in 0..self.out {
            let (c, s) = crate::scan::quantize_i8(&self.w[o * self.inn..(o + 1) * self.inn]);
            self.q.extend(c);
            self.qs.push(s);
        }
    }
    /// y[t] = W x[t] + b for `t` tokens (x: t*inn, y: t*out); AVX2: 2 weight rows x 4 tokens per register block, remainder tokens / rows via the single-row kernels
    fn apply(&self, x: &[f32], t: usize, y: &mut Vec<f32>, int8: bool) {
        y.clear();
        y.resize(t * self.out, 0.0);
        let simd = avx_ok();
        let xq: Vec<(Vec<i8>, f32)> = if int8 { (0..t).map(|i| crate::scan::quantize_i8(&x[i * self.inn..(i + 1) * self.inn])).collect() } else { Vec::new() };
        let mut i = 0;
        #[cfg(target_arch = "x86_64")]
        if simd {
            while i + 4 <= t {
                let mut o = 0;
                while o + 2 <= self.out {
                    if int8 {
                        let (r0, r1) = (&self.q[o * self.inn..(o + 1) * self.inn], &self.q[(o + 1) * self.inn..(o + 2) * self.inn]);
                        // SAFETY: avx2 checked
                        let d = unsafe { dot24_i8_avx2(r0, r1, [&xq[i].0, &xq[i + 1].0, &xq[i + 2].0, &xq[i + 3].0]) };
                        for r in 0..2 {
                            for j in 0..4 {
                                y[(i + j) * self.out + o + r] = d[r][j] as f32 * self.qs[o + r] * xq[i + j].1 + self.b[o + r];
                            }
                        }
                    } else {
                        let (r0, r1) = (&self.w[o * self.inn..(o + 1) * self.inn], &self.w[(o + 1) * self.inn..(o + 2) * self.inn]);
                        let xs = |k: usize| &x[(i + k) * self.inn..(i + k + 1) * self.inn];
                        // SAFETY: avx2+fma checked
                        let d = unsafe { dot24_f32_avx2(r0, r1, [xs(0), xs(1), xs(2), xs(3)]) };
                        for r in 0..2 {
                            for j in 0..4 {
                                y[(i + j) * self.out + o + r] = d[r][j] + self.b[o + r];
                            }
                        }
                    }
                    o += 2;
                }
                i += 4;
            }
        }
        for tt in i..t {
            for o in 0..self.out {
                y[tt * self.out + o] = if int8 {
                    crate::scan::dot_i8_fast(&self.q[o * self.inn..(o + 1) * self.inn], &xq[tt].0) as f32 * self.qs[o] * xq[tt].1 + self.b[o]
                } else {
                    dot_f32(&self.w[o * self.inn..(o + 1) * self.inn], &x[tt * self.inn..(tt + 1) * self.inn]) + self.b[o]
                };
            }
        }
    }
}

fn fold_accent(c: char) -> char {
    match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => 'a',
        'ç' => 'c',
        'è' | 'é' | 'ê' | 'ë' => 'e',
        'ì' | 'í' | 'î' | 'ï' => 'i',
        'ñ' => 'n',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' => 'o',
        'ù' | 'ú' | 'û' | 'ü' => 'u',
        'ý' | 'ÿ' => 'y',
        _ => c,
    }
}
fn is_punct(c: char) -> bool {
    let u = c as u32;
    (33..=47).contains(&u) || (58..=64).contains(&u) || (91..=96).contains(&u) || (123..=126).contains(&u) || (!c.is_ascii() && !c.is_alphanumeric() && !c.is_whitespace() && !c.is_control())
}
fn is_cjk(c: char) -> bool {
    let u = c as u32;
    (0x4E00..=0x9FFF).contains(&u) || (0x3400..=0x4DBF).contains(&u) || (0x20000..=0x2A6DF).contains(&u) || (0xF900..=0xFAFF).contains(&u) || (0x2F800..=0x2FA1F).contains(&u)
}

fn rd(buf: &[f32], pos: &mut usize, n: usize) -> Vec<f32> {
    let v = buf[*pos..*pos + n].to_vec();
    *pos += n;
    v
}

impl Encoder {
    /// `weights`: flat f32 file from `export_weights.py`; `tokenizer_json`: the model's tokenizer.json (WordPiece vocab)
    pub fn load(weights: &std::path::Path, tokenizer_json: &std::path::Path) -> Result<Encoder, EncoderError> {
        let raw = std::fs::read(weights).map_err(|e| EncoderError(format!("weights: {e}")))?;
        let buf: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let layer_floats = 4 * (D * D + D) + 2 * D + (FF * D + FF) + (D * FF + D) + 2 * D;
        let need = 30522 * D + 512 * D + D + 2 * D + L * layer_floats;
        if buf.len() != need {
            return Err(EncoderError(format!("weights file has {} floats, expected {need}", buf.len())));
        }
        let tj: serde_json::Value = serde_json::from_slice(&std::fs::read(tokenizer_json).map_err(|e| EncoderError(format!("tokenizer: {e}")))?).map_err(|e| EncoderError(format!("tokenizer json: {e}")))?;
        let mut vocab = HashMap::new();
        for (k, v) in tj["model"]["vocab"].as_object().ok_or_else(|| EncoderError("no vocab".into()))? {
            vocab.insert(k.clone(), v.as_u64().unwrap_or(0) as u32);
        }
        let mut p = 0usize;
        let word = rd(&buf, &mut p, 30522 * D);
        let pos = rd(&buf, &mut p, 512 * D);
        let typ = rd(&buf, &mut p, D);
        let eln = (rd(&buf, &mut p, D), rd(&buf, &mut p, D));
        let mut layers = Vec::new();
        for _ in 0..L {
            let lin = |o: usize, i: usize, p: &mut usize| -> Lin {
                let w = rd(&buf, p, o * i);
                let b = rd(&buf, p, o);
                Lin::new(w, b, o, i)
            };
            let q = lin(D, D, &mut p);
            let k = lin(D, D, &mut p);
            let v = lin(D, D, &mut p);
            let o = lin(D, D, &mut p);
            let ln1 = (rd(&buf, &mut p, D), rd(&buf, &mut p, D));
            let f1 = lin(FF, D, &mut p);
            let f2 = lin(D, FF, &mut p);
            let ln2 = (rd(&buf, &mut p, D), rd(&buf, &mut p, D));
            layers.push(Layer { q, k, v, o, ln1, f1, f2, ln2 });
        }
        Ok(Encoder { vocab, word, pos, typ, eln, layers, int8: false })
    }
    /// opt-in int8 matmuls (per-row weights, per-token dynamic activations)
    pub fn with_int8(mut self, on: bool) -> Encoder {
        self.int8 = on;
        if on {
            for l in self.layers.iter_mut() {
                for m in [&mut l.q, &mut l.k, &mut l.v, &mut l.o, &mut l.f1, &mut l.f2] {
                    m.quantize();
                }
            }
        }
        self
    }
    /// BERT basic tokenisation + WordPiece -> ids with [CLS] ... [SEP], at most 128 tokens
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut clean = String::new();
        for c in text.chars() {
            if c == '\0' || c == '\u{fffd}' || (c.is_control() && !c.is_whitespace()) {
                continue;
            }
            let c = c.to_lowercase().next().unwrap_or(c);
            let c = fold_accent(c);
            if is_cjk(c) {
                clean.push(' ');
                clean.push(c);
                clean.push(' ');
            } else if c.is_whitespace() {
                clean.push(' ');
            } else {
                clean.push(c);
            }
        }
        let unk = *self.vocab.get("[UNK]").unwrap_or(&100);
        let mut ids = vec![*self.vocab.get("[CLS]").unwrap_or(&101)];
        for w in clean.split_whitespace() {
            let mut pieces: Vec<String> = Vec::new();
            let mut cur = String::new();
            for c in w.chars() {
                if is_punct(c) {
                    if !cur.is_empty() {
                        pieces.push(std::mem::take(&mut cur));
                    }
                    pieces.push(c.to_string());
                } else {
                    cur.push(c);
                }
            }
            if !cur.is_empty() {
                pieces.push(cur);
            }
            for pc in pieces {
                let chars: Vec<char> = pc.chars().collect();
                if chars.len() > 100 {
                    ids.push(unk);
                    continue;
                }
                let (mut start, mut sub, mut bad) = (0usize, Vec::new(), false);
                while start < chars.len() {
                    let mut end = chars.len();
                    let mut found = None;
                    while start < end {
                        let mut s: String = chars[start..end].iter().collect();
                        if start > 0 {
                            s = format!("##{s}");
                        }
                        if let Some(&id) = self.vocab.get(&s) {
                            found = Some(id);
                            break;
                        }
                        end -= 1;
                    }
                    match found {
                        Some(id) => {
                            sub.push(id);
                            start = end;
                        }
                        None => {
                            bad = true;
                            break;
                        }
                    }
                }
                if bad {
                    ids.push(unk);
                } else {
                    ids.extend(sub);
                }
            }
        }
        ids.truncate(MAXLEN - 1);
        ids.push(*self.vocab.get("[SEP]").unwrap_or(&102));
        ids
    }
    /// L2-normalised 384-d mean-pooled embedding of `text`
    pub fn encode(&self, text: &str) -> Vec<f32> {
        let ids = self.tokenize(text);
        let t = ids.len();
        let mut x = vec![0f32; t * D];
        for (i, &id) in ids.iter().enumerate() {
            let w = &self.word[id as usize * D..(id as usize + 1) * D];
            let p = &self.pos[i * D..(i + 1) * D];
            for j in 0..D {
                x[i * D + j] = w[j] + p[j] + self.typ[j];
            }
            layer_norm(&mut x[i * D..(i + 1) * D], &self.eln.0, &self.eln.1);
        }
        let (mut q, mut k, mut v, mut o, mut h1, mut h2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let scale = 1.0 / (DH as f32).sqrt();
        for l in &self.layers {
            l.q.apply(&x, t, &mut q, self.int8);
            l.k.apply(&x, t, &mut k, self.int8);
            l.v.apply(&x, t, &mut v, self.int8);
            let mut ctx = vec![0f32; t * D];
            let mut sc = vec![0f32; t];
            for h in 0..H {
                for i in 0..t {
                    let qi = &q[i * D + h * DH..i * D + (h + 1) * DH];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..t {
                        sc[j] = dot_f32(qi, &k[j * D + h * DH..j * D + (h + 1) * DH]) * scale;
                        mx = mx.max(sc[j]);
                    }
                    let mut z = 0.0;
                    for s in sc.iter_mut() {
                        *s = fast_exp(*s - mx);
                        z += *s;
                    }
                    for j in 0..t {
                        let a = sc[j] / z;
                        for d in 0..DH {
                            ctx[i * D + h * DH + d] += a * v[j * D + h * DH + d];
                        }
                    }
                }
            }
            l.o.apply(&ctx, t, &mut o, self.int8);
            for i in 0..t {
                for j in 0..D {
                    x[i * D + j] += o[i * D + j];
                }
                layer_norm(&mut x[i * D..(i + 1) * D], &l.ln1.0, &l.ln1.1);
            }
            l.f1.apply(&x, t, &mut h1, self.int8);
            gelu_all(&mut h1);
            l.f2.apply(&h1, t, &mut h2, self.int8);
            for i in 0..t {
                for j in 0..D {
                    x[i * D + j] += h2[i * D + j];
                }
                layer_norm(&mut x[i * D..(i + 1) * D], &l.ln2.0, &l.ln2.1);
            }
        }
        let mut out = vec![0f32; D];
        for i in 0..t {
            for j in 0..D {
                out[j] += x[i * D + j] / t as f32;
            }
        }
        let n = out.iter().map(|z| z * z).sum::<f32>().sqrt().max(1e-12);
        out.iter_mut().for_each(|z| *z /= n);
        out
    }
}
