"""
core.py - logika inti (tanpa antarmuka): metrik S-Box, Genetic Algorithm,
AES-128 termodifikasi (S-Box GA, CBC dua arah), serta metrik gambar (entropi, NPCR, UACI).
Dikonversi dari notebook Colab Projek Kriptografi Kelompok 9.
"""
import io
import json
import random
import time
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.stats import norm

# =====================================================================
# 1. KONSTANTA & METRIK S-BOX
# =====================================================================
N = 256
INPUTS = np.arange(N, dtype=np.uint16)
PARITY = np.array([bin(i).count("1") & 1 for i in range(256)], dtype=np.uint8)


def is_valid_sbox(sbox):
    s = np.asarray(sbox)
    return len(s) == 256 and np.array_equal(np.sort(s), np.arange(256))


def fwht(values):
    """Walsh-Hadamard transform untuk array panjang 256."""
    a = np.asarray(values, dtype=np.int32).copy()
    h = 1
    while h < len(a):
        for start in range(0, len(a), 2 * h):
            x = a[start:start + h].copy()
            y = a[start + h:start + 2 * h].copy()
            a[start:start + h] = x + y
            a[start + h:start + 2 * h] = x - y
        h *= 2
    return a


def component_nonlinearity(truth):
    signs = 1 - 2 * np.asarray(truth, dtype=np.int32)
    walsh = fwht(signs)
    return 128 - int(np.max(np.abs(walsh))) // 2


def calculate_metrics(sbox):
    s = np.asarray(sbox, dtype=np.uint8)
    if not is_valid_sbox(s):
        raise ValueError("S-Box harus berupa permutasi lengkap 0..255.")

    # Nonlinearity: minimum dari 255 kombinasi XOR bit output
    nl_values, component_truths = [], {}
    for mask in range(1, 256):
        truth = PARITY[np.bitwise_and(s, mask)]
        component_truths[mask] = truth
        nl_values.append(component_nonlinearity(truth))
    nl = int(min(nl_values))

    # SAC
    sac_matrix = np.zeros((8, 8), dtype=float)
    for i in range(8):
        changed = np.bitwise_xor(s, s[np.bitwise_xor(INPUTS, 1 << i)])
        for j in range(8):
            sac_matrix[i, j] = np.mean((changed >> j) & 1)
    sac_deviation = float(np.mean(np.abs(sac_matrix - 0.5)))

    # BIC-NL
    bic_nl_values = []
    for i in range(8):
        for j in range(i + 1, 8):
            mask = (1 << i) | (1 << j)
            bic_nl_values.append(component_nonlinearity(component_truths[mask]))
    bic_nl = int(min(bic_nl_values))

    # BIC-SAC
    bic_sac_values = []
    for i in range(8):
        changed = np.bitwise_xor(s, s[np.bitwise_xor(INPUTS, 1 << i)])
        for j in range(8):
            for k in range(j + 1, 8):
                pair_change = ((changed >> j) ^ (changed >> k)) & 1
                bic_sac_values.append(abs(float(np.mean(pair_change)) - 0.5))
    bic_sac_deviation = float(np.mean(bic_sac_values))

    # LAP
    max_corr = 0
    for out_mask in range(1, 256):
        out_parity = PARITY[np.bitwise_and(s, out_mask)]
        corr = fwht(1 - 2 * out_parity.astype(np.int32))
        max_corr = max(max_corr, int(np.max(np.abs(corr[1:]))))
    lap = max_corr / 512.0

    # DAP
    max_diff_count = 0
    for dx in range(1, 256):
        dy = np.bitwise_xor(s, s[np.bitwise_xor(INPUTS, dx)])
        counts = np.bincount(dy, minlength=256)
        max_diff_count = max(max_diff_count, int(counts.max()))
    dap = max_diff_count / 256.0

    return {
        "NL": nl,
        "SAC_deviation": sac_deviation,
        "BIC_NL": bic_nl,
        "BIC_SAC_deviation": bic_sac_deviation,
        "LAP": lap,
        "DAP": dap,
        "SAC_matrix": sac_matrix,
    }


# =====================================================================
# 2. GENETIC ALGORITHM (kromosom = parameter ekuivalen-affine fungsi invers GF(2^8))
# =====================================================================
def gf_mul_poly(a, b, poly):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        if a & 0x100:
            a ^= poly
        b >>= 1
    return r


def _pmod(a, b):
    while a.bit_length() >= b.bit_length():
        a ^= b << (a.bit_length() - b.bit_length())
    return a


def _irreducible(poly):
    return all(_pmod(poly, d) != 0 for d in range(2, 32))


def _gf_inverse(a, poly):
    """a^254 = a^-1 di GF(2^8), dihitung dengan square-and-multiply (hasil sama
    dengan 254x perkalian di notebook, tetapi jauh lebih cepat)."""
    result, base, e = 1, a, 254
    while e:
        if e & 1:
            result = gf_mul_poly(result, base, poly)
        base = gf_mul_poly(base, base, poly)
        e >>= 1
    return result


@st.cache_resource(show_spinner="Menyiapkan tabel invers GF(2^8)...")
def get_inverse_tables():
    polys = [p for p in range(0x100, 0x200) if _irreducible(p)]
    assert len(polys) == 30
    tables = []
    for p in polys:
        t = np.zeros(256, dtype=np.int64)
        for a in range(1, 256):
            t[a] = _gf_inverse(a, p)
        tables.append(t)
    return polys, tables


_PAR = [bin(i).count("1") & 1 for i in range(256)]


def mat_rank(rows):
    rows = list(rows)
    r = 0
    for bit in range(8):
        piv = next((i for i in range(r, 8) if rows[i] >> bit & 1), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(8):
            if i != r and rows[i] >> bit & 1:
                rows[i] ^= rows[r]
        r += 1
    return r


def random_invertible_matrix():
    while True:
        m = [random.randrange(1, 256) for _ in range(8)]
        if mat_rank(m) == 8:
            return m


def linear_table(m):
    return np.array(
        [sum(_PAR[m[i] & v] << i for i in range(8)) for v in range(256)], dtype=np.int64
    )


def build_sbox(g, inv_tables):
    poly, A, B, c, d = g
    inv = inv_tables[poly]
    return (linear_table(A)[inv[linear_table(B)[INPUTS.astype(int)] ^ d]] ^ c).astype(np.uint8)


_X = np.arange(256)


def sac_bic_fast(s):
    s = s.astype(np.int64)
    sac = np.zeros((8, 8))
    bic = []
    for i in range(8):
        ch = s ^ s[_X ^ (1 << i)]
        bits = [(ch >> j) & 1 for j in range(8)]
        for j in range(8):
            sac[i, j] = bits[j].mean()
            for k in range(j + 1, 8):
                bic.append((bits[j] ^ bits[k]).mean())
    return sac, np.array(bic)


def fitness_function(g, inv_tables):
    sac, bic = sac_bic_fast(build_sbox(g, inv_tables))
    strict = (np.abs(sac - 0.5).mean() + np.abs(bic - 0.5).mean()) / 2
    paper = abs(sac.mean() - 0.5) + abs(bic.mean() - 0.5)
    worst = np.abs(sac - 0.5).max()
    return float(strict + 0.5 * paper + 0.1 * worst)  # makin kecil makin baik


def random_chromosome():
    return [random.randrange(30), random_invertible_matrix(), random_invertible_matrix(),
            random.randrange(256), random.randrange(256)]


def copy_g(g):
    return [g[0], list(g[1]), list(g[2]), g[3], g[4]]


def crossover(a, b):
    child = [random.choice([a[0], b[0]]),
             [random.choice(p) for p in zip(a[1], b[1])],
             [random.choice(p) for p in zip(a[2], b[2])],
             random.choice([a[3], b[3]]), random.choice([a[4], b[4]])]
    if mat_rank(child[1]) < 8 or mat_rank(child[2]) < 8:
        return copy_g(a)
    return child


def _flip_bit_keep_invertible(m):
    for _ in range(20):
        n = list(m)
        n[random.randrange(8)] ^= 1 << random.randrange(8)
        if mat_rank(n) == 8:
            return n
    return m


def mutate(g):
    g = copy_g(g)
    for _ in range(random.randint(1, 3)):
        t = random.random()
        if t < .10:
            g[0] = random.randrange(30)
        elif t < .45:
            g[1] = _flip_bit_keep_invertible(g[1])
        elif t < .80:
            g[2] = _flip_bit_keep_invertible(g[2])
        elif t < .90:
            g[3] ^= 1 << random.randrange(8)
        else:
            g[4] ^= 1 << random.randrange(8)
    return g


def tournament(pop, scores, k):
    ids = random.sample(range(len(pop)), k)
    return pop[min(ids, key=lambda i: scores[i])]


def run_ga_once(seed, P, inv_tables, on_gen=None):
    random.seed(seed)
    np.random.seed(seed)
    pop = [random_chromosome() for _ in range(P["pop"])]
    best, best_score, hist = None, float("inf"), []
    for gen in range(P["gens"]):
        scores = [fitness_function(g, inv_tables) for g in pop]
        order = np.argsort(scores)
        pop = [pop[i] for i in order]
        scores = [scores[i] for i in order]
        if scores[0] < best_score:
            best_score, best = scores[0], copy_g(pop[0])
        hist.append({"restart": seed, "generation": gen + 1,
                     "best_fitness": float(scores[0]), "mean_fitness": float(np.mean(scores))})
        new = [copy_g(pop[i]) for i in range(P["elite"])]
        while len(new) < P["pop"]:
            p1 = tournament(pop, scores, P["tour"])
            p2 = tournament(pop, scores, P["tour"])
            child = crossover(p1, p2) if random.random() < P["cx"] else copy_g(p1)
            if random.random() < P["mut"]:
                child = mutate(child)
            new.append(child)
        pop = new
        if on_gen:
            on_gen(gen + 1)
    return best, best_score, hist


def run_ga(P, progress=None):
    polys, inv_tables = get_inverse_tables()
    overall_best, overall_score, all_hist, best_seed = None, float("inf"), [], 1
    for idx, seed in enumerate(range(1, P["restarts"] + 1)):
        def on_gen(g, idx=idx, seed=seed):
            if progress:
                frac = (idx * P["gens"] + g) / (P["restarts"] * P["gens"])
                progress.progress(frac, text=f"Restart {seed}/{P['restarts']} - generasi {g}/{P['gens']}")
        g, sc, h = run_ga_once(seed, P, inv_tables, on_gen)
        all_hist += h
        if sc < overall_score:
            overall_score, overall_best, best_seed = sc, g, seed
    sbox = build_sbox(overall_best, inv_tables)
    assert is_valid_sbox(sbox)
    history = pd.DataFrame([r for r in all_hist if r["restart"] == best_seed])
    return sbox, history, overall_score, best_seed, bin(polys[overall_best[0]])


# =====================================================================
# 3. AES-128 TERMODIFIKASI (S-Box hasil GA) + MODE "CBC DUA ARAH"
# =====================================================================
def xtime(a):
    a = int(a)
    return ((a << 1) ^ (0x11B if a & 0x80 else 0)) & 0xFF


def gf_mul(a, b):
    result, a, b = 0, int(a), int(b)
    while b:
        if b & 1:
            result ^= a
        a = xtime(a)
        b >>= 1
    return result


_MUL = {m: np.array([gf_mul(i, m) for i in range(256)], dtype=np.uint8) for m in (9, 11, 13, 14)}
_ISR_IDX = np.array([0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3])


class SBoxAES:
    def __init__(self, sbox):
        self.SBOX = np.asarray(sbox, dtype=np.uint8)
        self.INV = np.empty(256, dtype=np.uint8)
        self.INV[self.SBOX] = np.arange(256, dtype=np.uint8)
        S = [int(v) for v in self.SBOX]
        mul2 = [gf_mul(i, 2) for i in range(256)]
        mul3 = [gf_mul(i, 3) for i in range(256)]
        T0, T1, T2, T3 = [], [], [], []
        for x in range(256):
            v = S[x]
            v2, v3 = mul2[v], mul3[v]
            T0.append((v2 << 24) | (v << 16) | (v << 8) | v3)
            T1.append((v3 << 24) | (v2 << 16) | (v << 8) | v)
            T2.append((v << 24) | (v3 << 16) | (v2 << 8) | v)
            T3.append((v << 24) | (v << 16) | (v3 << 8) | v2)
        self.S, self.T0, self.T1, self.T2, self.T3 = S, T0, T1, T2, T3

    # ---- key schedule (memakai S-Box GA) ----
    def expand_key(self, key):
        if len(key) != 16:
            raise ValueError("Kunci harus 16 byte (AES-128).")
        words = [list(key[i:i + 4]) for i in range(0, 16, 4)]
        rcon = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]
        for i in range(4, 44):
            temp = words[i - 1].copy()
            if i % 4 == 0:
                temp = temp[1:] + temp[:1]                 # RotWord
                temp = [self.S[x] for x in temp]           # SubWord
                temp[0] ^= rcon[i // 4]
            words.append([words[i - 4][j] ^ temp[j] for j in range(4)])
        return [sum(words[4 * r:4 * r + 4], []) for r in range(11)]

    def _round_key_words(self, key):
        return [[(rk[4 * c] << 24) | (rk[4 * c + 1] << 16) | (rk[4 * c + 2] << 8) | rk[4 * c + 3]
                 for c in range(4)] for rk in self.expand_key(key)]

    # ---- enkripsi 1 blok (4 word 32-bit) dengan T-table ----
    def _enc_block_words(self, w0, w1, w2, w3, rk):
        S, T0, T1, T2, T3 = self.S, self.T0, self.T1, self.T2, self.T3
        k = rk[0]
        w0 ^= k[0]; w1 ^= k[1]; w2 ^= k[2]; w3 ^= k[3]
        for r in range(1, 10):
            k = rk[r]
            n0 = T0[w0 >> 24] ^ T1[(w1 >> 16) & 255] ^ T2[(w2 >> 8) & 255] ^ T3[w3 & 255] ^ k[0]
            n1 = T0[w1 >> 24] ^ T1[(w2 >> 16) & 255] ^ T2[(w3 >> 8) & 255] ^ T3[w0 & 255] ^ k[1]
            n2 = T0[w2 >> 24] ^ T1[(w3 >> 16) & 255] ^ T2[(w0 >> 8) & 255] ^ T3[w1 & 255] ^ k[2]
            n3 = T0[w3 >> 24] ^ T1[(w0 >> 16) & 255] ^ T2[(w1 >> 8) & 255] ^ T3[w2 & 255] ^ k[3]
            w0, w1, w2, w3 = n0, n1, n2, n3
        k = rk[10]
        o0 = ((S[w0 >> 24] << 24) | (S[(w1 >> 16) & 255] << 16) | (S[(w2 >> 8) & 255] << 8) | S[w3 & 255]) ^ k[0]
        o1 = ((S[w1 >> 24] << 24) | (S[(w2 >> 16) & 255] << 16) | (S[(w3 >> 8) & 255] << 8) | S[w0 & 255]) ^ k[1]
        o2 = ((S[w2 >> 24] << 24) | (S[(w3 >> 16) & 255] << 16) | (S[(w0 >> 8) & 255] << 8) | S[w1 & 255]) ^ k[2]
        o3 = ((S[w3 >> 24] << 24) | (S[(w0 >> 16) & 255] << 16) | (S[(w1 >> 8) & 255] << 8) | S[w2 & 255]) ^ k[3]
        return o0, o1, o2, o3

    # ---- dekripsi semua blok sekaligus (NumPy) ----
    def _decrypt_blocks_np(self, blocks, rks):
        rk = [np.array(r, dtype=np.uint8) for r in rks]
        s = blocks ^ rk[10]
        s = s[:, _ISR_IDX]
        s = self.INV[s]
        for r in range(9, 0, -1):
            s = s ^ rk[r]
            out = np.empty_like(s)
            for c in range(4):
                a0, a1, a2, a3 = (s[:, 4 * c + i] for i in range(4))
                out[:, 4 * c + 0] = _MUL[14][a0] ^ _MUL[11][a1] ^ _MUL[13][a2] ^ _MUL[9][a3]
                out[:, 4 * c + 1] = _MUL[9][a0] ^ _MUL[14][a1] ^ _MUL[11][a2] ^ _MUL[13][a3]
                out[:, 4 * c + 2] = _MUL[13][a0] ^ _MUL[9][a1] ^ _MUL[14][a2] ^ _MUL[11][a3]
                out[:, 4 * c + 3] = _MUL[11][a0] ^ _MUL[13][a1] ^ _MUL[9][a2] ^ _MUL[14][a3]
            s = out[:, _ISR_IDX]
            s = self.INV[s]
        return s ^ rk[0]

    @staticmethod
    def _pad(data):
        p = 16 - (len(data) % 16)
        return data + bytes([p]) * p

    def _chain_encrypt(self, blocks, rk, iv_words):
        p0, p1, p2, p3 = iv_words
        out = []
        for b0, b1, b2, b3 in blocks:
            p0, p1, p2, p3 = self._enc_block_words(b0 ^ p0, b1 ^ p1, b2 ^ p2, b3 ^ p3, rk)
            out.append((p0, p1, p2, p3))
        return out

    def encrypt_bytes(self, data, key, iv=None):
        """CBC dua arah: pass maju lalu pass mundur -> difusi ke seluruh ciphertext."""
        if iv is None:
            iv = bytes(16)
        if len(iv) != 16:
            raise ValueError("IV harus 16 byte.")
        rk = self._round_key_words(key)
        padded = self._pad(data)
        w = np.frombuffer(padded, dtype=">u4").astype(np.int64).reshape(-1, 4)
        blocks = [tuple(int(x) for x in row) for row in w]
        iv1 = tuple(int(x) for x in np.frombuffer(iv, dtype=">u4"))
        iv2 = self._enc_block_words(*iv1, rk)
        c = self._chain_encrypt(blocks, rk, iv1)
        d = self._chain_encrypt(c[::-1], rk, iv2)[::-1]
        return np.array(d, dtype=">u4").tobytes()

    def decrypt_bytes(self, ciphertext, key, iv=None):
        if iv is None:
            iv = bytes(16)
        if len(iv) != 16 or len(ciphertext) % 16:
            raise ValueError("IV harus 16 byte dan ciphertext kelipatan 16 byte.")
        rk = self._round_key_words(key)
        rk_bytes = self.expand_key(key)
        iv1 = tuple(int(x) for x in np.frombuffer(iv, dtype=">u4"))
        iv2_b = np.array(self._enc_block_words(*iv1, rk), dtype=">u4").tobytes()
        D = np.frombuffer(ciphertext, dtype=np.uint8).reshape(-1, 16)
        dec = self._decrypt_blocks_np(D, rk_bytes)
        prev = np.vstack([D[1:], np.frombuffer(iv2_b, dtype=np.uint8)[None, :]])
        C = dec ^ prev
        dec = self._decrypt_blocks_np(C, rk_bytes)
        prev = np.vstack([np.frombuffer(iv, dtype=np.uint8)[None, :], C[:-1]])
        padded = (dec ^ prev).tobytes()
        pad = padded[-1]
        if pad < 1 or pad > 16 or padded[-pad:] != bytes([pad]) * pad:
            raise ValueError("Padding tidak valid; periksa kunci atau ciphertext.")
        return padded[:-pad]


# =====================================================================
# 4. METRIK GAMBAR: ENTROPY, NPCR, UACI
# =====================================================================
def entropy(data):
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / len(data)
    return float(-np.sum(p * np.log2(p)))


def npcr_uaci(original, modified):
    a = np.frombuffer(original, dtype=np.uint8).astype(np.int16)
    b = np.frombuffer(modified, dtype=np.uint8).astype(np.int16)
    if len(a) != len(b):
        raise ValueError("Data harus memiliki panjang sama.")
    return float(np.mean(a != b) * 100.0), float(np.mean(np.abs(a - b) / 255.0) * 100.0)


NPCR_IDEAL = 255 / 256 * 100
UACI_IDEAL = (255 + 2) / (3 * 255 + 3) * 100


def critical_values(L, alpha=0.05, F=255):
    """Batas kritis Wu et al. (2011)."""
    npcr_min = (F - norm.ppf(1 - alpha) * np.sqrt(F / L)) / (F + 1) * 100
    mu = (F + 2) / (3 * F + 3)
    sd = np.sqrt((F + 2) * (F ** 2 + 2 * F + 3) / (18 * (F + 1) ** 2 * F * L))
    z = norm.ppf(1 - alpha / 2)
    return npcr_min, (mu - z * sd) * 100, (mu + z * sd) * 100


def encrypt_one_image(file, cipher, key, iv, max_side, n_trials):
    img = Image.open(file).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    raw = img.tobytes()
    t0 = time.time()
    enc = cipher.encrypt_bytes(raw, key, iv)
    rec = cipher.decrypt_bytes(enc, key, iv)
    ok = rec == raw
    cipher_pixels = enc[:len(raw)]
    cipher_img = Image.frombytes("RGB", img.size, cipher_pixels)
    dec_img = Image.frombytes("RGB", img.size, rec)

    positions = [0, len(raw) - 1] + [random.randrange(len(raw)) for _ in range(max(n_trials - 2, 0))]
    npcrs, uacis = [], []
    for pos in positions:
        mod = bytearray(raw)
        mod[pos] ^= 1
        c2 = cipher.encrypt_bytes(bytes(mod), key, iv)
        n, u = npcr_uaci(cipher_pixels, c2[:len(raw)])
        npcrs.append(n)
        uacis.append(u)
    n_min, u_lo, u_hi = critical_values(len(raw))
    return {
        "name": getattr(file, "name", "image"),
        "original": img, "cipher_img": cipher_img, "dec_img": dec_img, "enc_bytes": enc,
        "row": {
            "image": getattr(file, "name", "image"), "width": img.width, "height": img.height,
            "original_entropy": entropy(raw), "cipher_entropy": entropy(cipher_pixels),
            "decryption_success": ok, "ciphertext_bytes": len(enc),
            "NPCR mean (%)": float(np.mean(npcrs)), "UACI mean (%)": float(np.mean(uacis)),
            "NPCR lulus": f"{sum(n >= n_min for n in npcrs)}/{len(npcrs)}",
            "UACI lulus": f"{sum(u_lo <= u <= u_hi for u in uacis)}/{len(uacis)}",
            "NPCR batas bawah (%)": n_min, "UACI batas bawah (%)": u_lo, "UACI batas atas (%)": u_hi,
            "waktu (detik)": time.time() - t0,
        },
    }


def hist_figure(original, encrypted, title):
    o, e = np.asarray(original), np.asarray(encrypted)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    for ch, name in enumerate(["Red", "Green", "Blue"]):
        axes[0, ch].hist(o[:, :, ch].ravel(), bins=256, range=(0, 255))
        axes[0, ch].set_title(f"Original - {name}")
        axes[1, ch].hist(e[:, :, ch].ravel(), bins=256, range=(0, 255))
        axes[1, ch].set_title(f"Cipher - {name}")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()




def decrypt_to_image(data, cipher, key, iv, width, height):
    """Dekripsi ciphertext (.bin) menjadi gambar RGB. Melempar ValueError dengan pesan yang jelas."""
    if not data or len(data) % 16:
        raise ValueError("Ukuran berkas bukan kelipatan 16 byte. Pastikan ini berkas .bin hasil enkripsi yang utuh.")
    try:
        raw = cipher.decrypt_bytes(data, key, iv)
    except ValueError:
        raise ValueError("Dekripsi gagal (padding tidak valid). Periksa kunci, IV, dan S-Box; "
                         "semuanya harus sama dengan saat enkripsi.")
    if len(raw) != width * height * 3:
        msg = f"Ukuran tidak cocok: data berisi {len(raw)} byte, sedangkan {width} x {height} x 3 = {width * height * 3} byte."
        if len(raw) % 3 == 0:
            p = len(raw) // 3
            pairs = [f"{a} x {p // a}" for a in range(int(p ** 0.5), 0, -1) if p % a == 0][:3]
            msg += f" Gambar ini punya {p} piksel; kemungkinan ukuran (lebar x tinggi atau sebaliknya): {', '.join(pairs)}."
        raise ValueError(msg)
    return Image.frombytes("RGB", (width, height), raw)