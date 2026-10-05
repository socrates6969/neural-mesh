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
    /// per-row sum of the int8 weights (VNNI path: dot(w, x) = dpbusd(x + 128, w) - 128 * sum(w))
    sw: Vec<i32>,
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
    threads: usize,
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

/// R111: global A/B switch for the AVX512-VNNI int8 kernel (default on when the CPU has it; results are bit-identical either way - integer dot products).
static VNNI_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn set_vnni(on: bool) {
    VNNI_OFF.store(!on, std::sync::atomic::Ordering::Relaxed);
}
fn vnni_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return !VNNI_OFF.load(std::sync::atomic::Ordering::Relaxed) && std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512vnni");
    }
    #[allow(unreachable_code)]
    false
}
/// 2 int8 weight rows x 4 tokens with AVX512-VNNI `vpdpbusd` (64 byte-MACs per instruction). The activations arrive as u8 = i8 + 128
/// (unsigned operand of dpbusd), so the exact signed dot is `dpbusd(xu, w) - 128 * sum(w)` (`sw0`/`sw1` = the rows' weight sums).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dot24_i8_vnni(w0: &[i8], w1: &[i8], sw0: i32, sw1: i32, x: [&[u8]; 4]) -> [[i32; 4]; 2] {
    use std::arch::x86_64::*;
    let n = w0.len();
    let mut a0 = [_mm512_setzero_si512(); 4];
    let mut a1 = [_mm512_setzero_si512(); 4];
    let mut i = 0;
    while i + 64 <= n {
        let v0 = _mm512_loadu_si512(w0.as_ptr().add(i) as *const _);
        let v1 = _mm512_loadu_si512(w1.as_ptr().add(i) as *const _);
        for j in 0..4 {
            let xv = _mm512_loadu_si512(x[j].as_ptr().add(i) as *const _);
            a0[j] = _mm512_dpbusd_epi32(a0[j], xv, v0);
            a1[j] = _mm512_dpbusd_epi32(a1[j], xv, v1);
        }
        i += 64;
    }
    let mut out = [[0i32; 4]; 2];
    for j in 0..4 {
        out[0][j] = _mm512_reduce_add_epi32(a0[j]);
        out[1][j] = _mm512_reduce_add_epi32(a1[j]);
    }
    // scalar tail on the u8 form so the correction below stays uniform
    while i < n {
        for j in 0..4 {
            out[0][j] += w0[i] as i32 * x[j][i] as i32;
            out[1][j] += w1[i] as i32 * x[j][i] as i32;
        }
        i += 1;
    }
    for j in 0..4 {
        out[0][j] -= 128 * sw0;
        out[1][j] -= 128 * sw1;
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
        Lin { w, b, q: Vec::new(), qs: Vec::new(), sw: Vec::new(), out, inn }
    }
    fn quantize(&mut self) {
        self.q.clear();
        self.qs.clear();
        self.sw.clear();
        for o in 0..self.out {
            let (c, s) = crate::scan::quantize_i8(&self.w[o * self.inn..(o + 1) * self.inn]);
            self.sw.push(c.iter().map(|&v| v as i32).sum());
            self.q.extend(c);
            self.qs.push(s);
        }
    }
    /// y[t] = W x[t] + b for `t` tokens (x: t*inn, y: t*out); with `threads` > 1 and a batch of >= 48 tokens the OUTPUT ROWS are split over threads (each streams its own slice of the weights)
    fn apply(&self, x: &[f32], t: usize, y: &mut Vec<f32>, int8: bool, threads: usize) {
        y.clear();
        y.resize(t * self.out, 0.0);
        let xq: Vec<(Vec<i8>, f32)> = if int8 { (0..t).map(|i| crate::scan::quantize_i8(&x[i * self.inn..(i + 1) * self.inn])).collect() } else { Vec::new() };
        let th = if t >= 48 { threads.max(1).min(self.out / 64).max(1) } else { 1 };
        if th == 1 {
            let loc = self.apply_range(x, t, &xq, 0, self.out, int8);
            y.copy_from_slice(&loc);
            return;
        }
        let step = (self.out.div_ceil(th) + 1) & !1;
        let parts: Vec<(usize, usize, Vec<f32>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..th)
                .map(|k| {
                    let (lo, hi) = ((k * step).min(self.out), ((k + 1) * step).min(self.out));
                    let xq = &xq;
                    sc.spawn(move || (lo, hi, self.apply_range(x, t, xq, lo, hi, int8)))
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("encoder thread")).collect()
        });
        for (lo, hi, loc) in parts {
            let w = hi - lo;
            for tt in 0..t {
                y[tt * self.out + lo..tt * self.out + hi].copy_from_slice(&loc[tt * w..(tt + 1) * w]);
            }
        }
    }
    /// outputs `o_lo..o_hi` (o_lo even) for all tokens, layout [t][o_hi - o_lo]
    fn apply_range(&self, x: &[f32], t: usize, xq: &[(Vec<i8>, f32)], o_lo: usize, o_hi: usize, int8: bool) -> Vec<f32> {
        let w = o_hi - o_lo;
        let mut y = vec![0f32; t * w];
        let simd = avx_ok();
        let use_vnni = int8 && simd && vnni_ok();
        let xu: Vec<Vec<u8>> = if use_vnni { xq.iter().map(|(c, _)| c.iter().map(|&v| (v as u8) ^ 0x80).collect()).collect() } else { Vec::new() };
        let mut i = 0;
        #[cfg(target_arch = "x86_64")]
        if simd {
            while i + 4 <= t {
                let mut o = o_lo;
                while o + 2 <= o_hi {
                    if int8 {
                        let (r0, r1) = (&self.q[o * self.inn..(o + 1) * self.inn], &self.q[(o + 1) * self.inn..(o + 2) * self.inn]);
                        let d = if use_vnni {
                            // SAFETY: avx512f/bw/vnni checked
                            unsafe { dot24_i8_vnni(r0, r1, self.sw[o], self.sw[o + 1], [&xu[i], &xu[i + 1], &xu[i + 2], &xu[i + 3]]) }
                        } else {
                            // SAFETY: avx2 checked
                            unsafe { dot24_i8_avx2(r0, r1, [&xq[i].0, &xq[i + 1].0, &xq[i + 2].0, &xq[i + 3].0]) }
                        };
                        for r in 0..2 {
                            for j in 0..4 {
                                y[(i + j) * w + o - o_lo + r] = d[r][j] as f32 * self.qs[o + r] * xq[i + j].1 + self.b[o + r];
                            }
                        }
                    } else {
                        let (r0, r1) = (&self.w[o * self.inn..(o + 1) * self.inn], &self.w[(o + 1) * self.inn..(o + 2) * self.inn]);
                        let xs = |k: usize| &x[(i + k) * self.inn..(i + k + 1) * self.inn];
                        // SAFETY: avx2+fma checked
                        let d = unsafe { dot24_f32_avx2(r0, r1, [xs(0), xs(1), xs(2), xs(3)]) };
                        for r in 0..2 {
                            for j in 0..4 {
                                y[(i + j) * w + o - o_lo + r] = d[r][j] + self.b[o + r];
                            }
                        }
                    }
                    o += 2;
                }
                i += 4;
            }
        }
        for tt in i..t {
            for o in o_lo..o_hi {
                y[tt * w + o - o_lo] = if int8 {
                    crate::scan::dot_i8_fast(&self.q[o * self.inn..(o + 1) * self.inn], &xq[tt].0) as f32 * self.qs[o] * xq[tt].1 + self.b[o]
                } else {
                    dot_f32(&self.w[o * self.inn..(o + 1) * self.inn], &x[tt * self.inn..(tt + 1) * self.inn]) + self.b[o]
                };
            }
        }
        y
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
        Ok(Encoder { vocab, word, pos, typ, eln, layers, int8: false, threads: 1 })
    }
    /// R105 (opt-in): split each linear layer's output rows over `threads` threads when a batch has >= 48 tokens
    pub fn with_threads(mut self, threads: usize) -> Encoder {
        self.threads = threads;
        self
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
        self.encode_batch(&[text]).pop().unwrap()
    }
    /// R104: encode several texts in ONE pass: all tokens of all sequences go through each linear layer together (the weights are streamed once per layer for the whole batch),
    /// attention stays per sequence. Same numbers as calling [`Encoder::encode`] per text (up to f32 summation order in the 4-token register blocks).
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        let seqs: Vec<Vec<u32>> = texts.iter().map(|t| self.tokenize(t)).collect();
        let t: usize = seqs.iter().map(|s| s.len()).sum();
        let mut starts = Vec::with_capacity(seqs.len() + 1);
        let mut acc = 0usize;
        for s in &seqs {
            starts.push(acc);
            acc += s.len();
        }
        starts.push(acc);
        let mut x = vec![0f32; t * D];
        for (si, ids) in seqs.iter().enumerate() {
            for (i, &id) in ids.iter().enumerate() {
                let r = starts[si] + i;
                let w = &self.word[id as usize * D..(id as usize + 1) * D];
                let p = &self.pos[i * D..(i + 1) * D];
                for j in 0..D {
                    x[r * D + j] = w[j] + p[j] + self.typ[j];
                }
                layer_norm(&mut x[r * D..(r + 1) * D], &self.eln.0, &self.eln.1);
            }
        }
        let (mut q, mut k, mut v, mut o, mut h1, mut h2) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let scale = 1.0 / (DH as f32).sqrt();
        let mut sc = vec![0f32; MAXLEN];
        for l in &self.layers {
            l.q.apply(&x, t, &mut q, self.int8, self.threads);
            l.k.apply(&x, t, &mut k, self.int8, self.threads);
            l.v.apply(&x, t, &mut v, self.int8, self.threads);
            let mut ctx = vec![0f32; t * D];
            for si in 0..seqs.len() {
                let (a0, a1) = (starts[si], starts[si + 1]);
                for h in 0..H {
                    for i in a0..a1 {
                        let qi = &q[i * D + h * DH..i * D + (h + 1) * DH];
                        let mut mx = f32::NEG_INFINITY;
                        for j in a0..a1 {
                            sc[j - a0] = dot_f32(qi, &k[j * D + h * DH..j * D + (h + 1) * DH]) * scale;
                            mx = mx.max(sc[j - a0]);
                        }
                        let mut z = 0.0;
                        for s in sc[..a1 - a0].iter_mut() {
                            *s = fast_exp(*s - mx);
                            z += *s;
                        }
                        for j in a0..a1 {
                            let a = sc[j - a0] / z;
                            for d in 0..DH {
                                ctx[i * D + h * DH + d] += a * v[j * D + h * DH + d];
                            }
                        }
                    }
                }
            }
            l.o.apply(&ctx, t, &mut o, self.int8, self.threads);
            for i in 0..t {
                for j in 0..D {
                    x[i * D + j] += o[i * D + j];
                }
                layer_norm(&mut x[i * D..(i + 1) * D], &l.ln1.0, &l.ln1.1);
            }
            l.f1.apply(&x, t, &mut h1, self.int8, self.threads);
            gelu_all(&mut h1);
            l.f2.apply(&h1, t, &mut h2, self.int8, self.threads);
            for i in 0..t {
                for j in 0..D {
                    x[i * D + j] += h2[i * D + j];
                }
                layer_norm(&mut x[i * D..(i + 1) * D], &l.ln2.0, &l.ln2.1);
            }
        }
        (0..seqs.len())
            .map(|si| {
                let (a0, a1) = (starts[si], starts[si + 1]);
                let mut out = vec![0f32; D];
                for i in a0..a1 {
                    for j in 0..D {
                        out[j] += x[i * D + j] / (a1 - a0) as f32;
                    }
                }
                let n = out.iter().map(|z| z * z).sum::<f32>().sqrt().max(1e-12);
                out.iter_mut().for_each(|z| *z /= n);
                out
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    //! Kernel tests need no files. The encoder tests need the exported weight file (`py/b1/r101_tools/export_weights.py`, 90 MB, not in the repo):
    //! set NM_ENCODER_WEIGHTS=<path>.f32 (tokenizer: NM_ENCODER_TOKENIZER or ~/.ruvector/models/all-MiniLM-L6-v2/tokenizer.json). Without it they print a SKIP line and pass.
    //! Golden vectors (`testdata/encoder_golden.json`) are ORT fp32 embeddings of the same strings.
    use super::*;
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn cos(a: &[f32], b: &[f32]) -> f32 {
        let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        d / (a.iter().map(|x| x * x).sum::<f32>().sqrt() * b.iter().map(|x| x * x).sum::<f32>().sqrt())
    }
    #[test]
    fn dot_kernels_match_scalar() {
        let mut s = 7u64;
        for n in [384usize, 1536] {
            let w: Vec<f32> = (0..n).map(|_| lcg(&mut s)).collect();
            let xs: Vec<Vec<f32>> = (0..4).map(|_| (0..n).map(|_| lcg(&mut s)).collect()).collect();
            for x in &xs {
                let want: f32 = w.iter().zip(x).map(|(a, b)| a * b).sum();
                assert!((dot_f32(&w, x) - want).abs() < 1e-3, "dot_f32 n={n}");
            }
            if avx_ok() {
                let got = unsafe { dot4_f32_avx2(&w, [&xs[0], &xs[1], &xs[2], &xs[3]]) };
                for k in 0..4 {
                    let want: f32 = w.iter().zip(&xs[k]).map(|(a, b)| a * b).sum();
                    assert!((got[k] - want).abs() < 1e-3, "dot4_f32 n={n} k={k}");
                }
                let wi: Vec<i8> = w.iter().map(|v| (v * 127.0) as i8).collect();
                let xi: Vec<Vec<i8>> = xs.iter().map(|x| x.iter().map(|v| (v * 127.0) as i8).collect()).collect();
                let goti = unsafe { dot4_i8_avx2(&wi, [&xi[0], &xi[1], &xi[2], &xi[3]]) };
                for k in 0..4 {
                    let want: i32 = wi.iter().zip(&xi[k]).map(|(a, b)| *a as i32 * *b as i32).sum();
                    assert_eq!(goti[k], want, "dot4_i8 is exact integer arithmetic (n={n}, k={k})");
                }
            }
        }
    }
    #[test]
    fn vnni_kernel_is_bit_exact_vs_avx2() {
        if !(avx_ok() && vnni_ok()) { eprintln!("SKIP vnni test: no AVX512-VNNI"); return; }
        let mut s = 99u64;
        for n in [64usize, 200, 384, 1536] {
            let rnd = |s: &mut u64| (lcg(s) * 127.0) as i8; // includes -127..127, never -128 (as scan::quantize_i8)
            let w0: Vec<i8> = (0..n).map(|_| rnd(&mut s)).collect();
            let w1: Vec<i8> = (0..n).map(|_| rnd(&mut s)).collect();
            let xs: Vec<Vec<i8>> = (0..4).map(|_| (0..n).map(|_| rnd(&mut s)).collect()).collect();
            let xu: Vec<Vec<u8>> = xs.iter().map(|x| x.iter().map(|&v| (v as u8) ^ 0x80).collect()).collect();
            let (sw0, sw1): (i32, i32) = (w0.iter().map(|&v| v as i32).sum(), w1.iter().map(|&v| v as i32).sum());
            let a = unsafe { dot24_i8_avx2(&w0, &w1, [&xs[0], &xs[1], &xs[2], &xs[3]]) };
            let b = unsafe { dot24_i8_vnni(&w0, &w1, sw0, sw1, [&xu[0], &xu[1], &xu[2], &xu[3]]) };
            assert_eq!(a, b, "n={n}");
        }
        // extreme values: all +127 weights x all +127 / -127 activations
        let w = vec![127i8; 384];
        for xv in [127i8, -127] {
            let x = vec![xv; 384]; let xu: Vec<u8> = x.iter().map(|&v| (v as u8) ^ 0x80).collect();
            let a = unsafe { dot24_i8_avx2(&w, &w, [&x, &x, &x, &x]) };
            let b = unsafe { dot24_i8_vnni(&w, &w, 127 * 384, 127 * 384, [&xu, &xu, &xu, &xu]) };
            assert_eq!(a, b);
        }
    }
    #[test]
    fn erf_gelu_accuracy() {
        for (x, want) in [(0.0f32, 0.0f32), (0.5, 0.5204999), (1.0, 0.8427008), (-1.0, -0.8427008), (2.0, 0.9953223), (4.0, 1.0)] {
            assert!((erf(x) - want).abs() < 2e-6, "erf({x}) = {} want {want}", erf(x));
        }
        assert!((gelu(1.0) - 0.8413447).abs() < 2e-6);
        assert!(gelu(-6.0).abs() < 1e-6 && (gelu(6.0) - 6.0).abs() < 1e-5);
        assert!(((fast_exp(1.0)) / std::f32::consts::E - 1.0).abs() < 1e-5);
    }
    fn files() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let w = std::path::PathBuf::from(std::env::var("NM_ENCODER_WEIGHTS").ok()?);
        let t = match std::env::var("NM_ENCODER_TOKENIZER") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                let h = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).ok()?;
                std::path::Path::new(&h).join(".ruvector/models/all-MiniLM-L6-v2/tokenizer.json")
            }
        };
        if w.exists() && t.exists() { Some((w, t)) } else { None }
    }
    #[test]
    fn load_rejects_wrong_weights_file() {
        let p = std::env::temp_dir().join("nm_enc_bad_weights.f32");
        std::fs::write(&p, [0u8; 64]).unwrap();
        let e = Encoder::load(&p, &p).err().expect("must fail");
        assert!(e.0.contains("expected"), "{}", e.0);
        let _ = std::fs::remove_file(&p);
    }
    #[test]
    fn encoder_matches_ort_golden_and_is_consistent() {
        let Some((w, t)) = files() else { eprintln!("SKIP encoder test: set NM_ENCODER_WEIGHTS"); return };
        let g: serde_json::Value = serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/encoder_golden.json")).unwrap()).unwrap();
        let texts: Vec<String> = g["texts"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
        let gold: Vec<Vec<f32>> = g["vectors"].as_array().unwrap().iter().map(|v| v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()).collect();
        let e = Encoder::load(&w, &t).unwrap();
        let e8 = Encoder::load(&w, &t).unwrap().with_int8(true);
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let batch = e.encode_batch(&refs);
        for (i, tx) in texts.iter().enumerate() {
            let v = e.encode(tx);
            assert_eq!(v.len(), 384);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "unit norm: {norm}");
            // accent text: folding is approximate (no full NFD), so a looser bound there
            let bound = if tx.contains('é') { 0.99 } else { 0.9999 };
            assert!(cos(&v, &gold[i]) > bound, "text {i} cos to ORT = {}", cos(&v, &gold[i]));
            assert!(cos(&v, &batch[i]) > 0.99999, "batch vs single, text {i}");
            assert_eq!(v, e.encode(tx), "deterministic");
            assert!(cos(&e8.encode(tx), &gold[i]) > 0.95, "int8 text {i}: {}", cos(&e8.encode(tx), &gold[i]));
        }
        // semantic sanity: related sentences closer than unrelated
        let a = e.encode("how to parse json in rust");
        let b = e.encode("deserialize JSON with serde in Rust");
        let c = e.encode("recipe for chocolate cake with butter");
        assert!(cos(&a, &b) > cos(&a, &c) + 0.2);
        // threads and empty/unknown inputs do not change or break results
        let et = Encoder::load(&w, &t).unwrap().with_threads(4);
        let many: Vec<&str> = (0..24).map(|i| refs[i % refs.len()]).collect();
        let bt = et.encode_batch(&many);
        for (i, v) in bt.iter().enumerate() { assert!(cos(v, &batch[i % refs.len()]) > 0.99999, "threaded batch {i}"); }
        assert_eq!(e.encode("").len(), 384);
        assert!(e.encode("zzzzqqqq xxxjjj 12345 \u{4e2d}\u{6587}").iter().all(|x| x.is_finite()));
    }
}
