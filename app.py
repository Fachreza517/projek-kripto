"""Antarmuka Streamlit - S-Box Genetic Algorithm & Enkripsi Gambar (Kelompok 9)."""
import io
import random
import string
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import core as C

DEFAULT_SBOX = Path(__file__).parent / "best_sbox.npy"
INK, COBALT, GRID = "#101B33", "#8E3FE9", "#DBDE38"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');
.stApp, .stApp p, .stApp label, .stApp li { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Bricolage Grotesque', sans-serif !important; letter-spacing: -0.02em; }
.block-container { max-width: 1120px; padding-top: 2.2rem; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { border-right: 1px solid #D5DBE8; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #D5DBE8; }
.stTabs [data-baseweb="tab"] { height: 46px; padding: 0 14px; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.02rem; }
.stTabs [aria-selected="true"] { color: #2D4BEA; }
.stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 500; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; }
[data-testid="stMetricLabel"] p { color: #55627f; }
.hero { display: flex; gap: 40px; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 1.4rem; }
.hero h1 { font-size: 2.5rem; line-height: 1.1; font-weight: 800; margin: 0 0 .7rem 0; padding: 0; max-width: 560px; }
.hero p { color: #3c4a6b; max-width: 520px; margin: 0; line-height: 1.55; }
.hero small { color: #6b7794; display: block; margin-top: .9rem; }
.sb { border-collapse: separate; border-spacing: 2px; }
.sb td { text-align: center; font-family: 'IBM Plex Mono', monospace; border-radius: 3px; }
.sb td.h { background: none; color: #6b7794; font-size: 11px; }
.scroll { overflow-x: auto; padding-bottom: 4px; }
.panel { background: #fff; border: 1px solid #D5DBE8; border-radius: 10px; }
.vrow { display: grid; grid-template-columns: 1.3fr 1fr 1.6fr auto; gap: 12px; align-items: center; padding: 11px 16px; border-bottom: 1px solid #E6EAF3; }
.vrow:last-child { border-bottom: 0; }
.vrow b { font-weight: 500; }
.vrow .v { font-family: 'IBM Plex Mono', monospace; }
.vrow .t { color: #6b7794; font-size: .9rem; }
.pill { padding: 2px 11px; border-radius: 99px; font-size: .82rem; white-space: nowrap; }
.ok { background: #DDF3EC; color: #0E5E4C; } .bad { background: #F8E1DB; color: #8E2F1E; } .info { background: #E4E8F5; color: #3A4A78; }
@media (max-width: 640px) { .hero h1 { font-size: 1.9rem; } .vrow { grid-template-columns: 1fr 1fr; } }
</style>
"""


# ------------------------------------------------------------------ elemen HTML
def _shade(v):
    t = v / 255
    a, b = (232, 237, 255), (45, 75, 234)
    r, g, bl = (round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return f"rgb({r},{g},{bl})", "#fff" if t > 0.55 else INK


def sbox_grid(sbox, mini=False):
    """Tabel 16x16: tiap sel diwarnai menurut nilainya (makin tua = nilai makin besar)."""
    w, h, fs = (14, 14, 0) if mini else (34, 30, 12)
    out = ['<div class="scroll"><table class="sb">']
    if not mini:
        out.append("<tr><td class='h'></td>" + "".join(f"<td class='h'>{c:X}</td>" for c in range(16)) + "</tr>")
    for r in range(16):
        out.append("<tr>" + ("" if mini else f"<td class='h'>{r:X}</td>"))
        for c in range(16):
            if sbox is None:
                bg, fg, txt, tip = "#F3A751", INK, "", ""
            else:
                v = int(sbox[r * 16 + c])
                (bg, fg), txt = _shade(v), ("" if mini else f"{v:02X}")
                tip = f"S[{r * 16 + c:02X}] = {v:02X} ({v})"
            out.append(f"<td title='{tip}' style='width:{w}px;height:{h}px;background:{bg};color:{fg};font-size:{fs}px'>{txt}</td>")
        out.append("</tr>")
    out.append("</table></div>")
    return "".join(out)


def sac_grid(mat):
    out = ["<div class='scroll'><table class='sb'><tr><td class='h'></td>"]
    out += [f"<td class='h'>out {j}</td>" for j in range(8)] + ["</tr>"]
    for i in range(8):
        out.append(f"<tr><td class='h'>in {i}</td>")
        for j in range(8):
            v = float(mat[i][j])
            dev = min(abs(v - 0.5) / 0.08, 1)
            bg = f"rgb({round(221 + (248 - 221) * dev)},{round(243 + (225 - 243) * dev)},{round(236 + (219 - 236) * dev)})"
            out.append(f"<td style='width:54px;height:30px;background:{bg};font-size:12px'>{v:.3f}</td>")
        out.append("</tr>")
    out.append("</table></div>")
    return "".join(out)


def verdict_rows(rows):
    html = ["<div class='panel'>"]
    for name, value, target, status in rows:
        cls, label = {"ok": ("ok", "memenuhi"), "bad": ("bad", "belum"), "info": ("info", "ukuran")}[status]
        html.append(f"<div class='vrow'><b>{name}</b><span class='v'>{value}</span>"
                    f"<span class='t'>{target}</span><span class='pill {cls}'>{label}</span></div>")
    html.append("</div>")
    return "".join(html)


# ------------------------------------------------------------------ state
def set_sbox(arr, info, src, history=None):
    st.session_state.update(sbox=np.asarray(arr, dtype=np.uint8), info=info, src=src, history=history)
    for k in ("metrics", "enc_results"):
        st.session_state.pop(k, None)


def random_key():
    st.session_state["key_text"] = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(16))


# ------------------------------------------------------------------ halaman
def main():
    st.set_page_config(page_title="S-Box GA dan Enkripsi Gambar", page_icon="🔐", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("key_text", "GA-SBox-Demo-123")

    # ---------- Sidebar
    with st.sidebar:
        st.markdown("### Pengaturan")
        options = (["S-Box bawaan"] if DEFAULT_SBOX.exists() else []) + ["Buat dengan GA", "Unggah file .npy"]
        source = st.radio("Sumber S-Box", options)
        P = {}
        if source == "Buat dengan GA":
            with st.expander("Parameter GA", expanded=True):
                P = dict(pop=st.slider("Ukuran populasi", 10, 120, 40, 10),
                         gens=st.slider("Generasi per restart", 10, 200, 50, 10),
                         restarts=st.slider("Jumlah restart", 1, 8, 2),
                         cx=st.slider("Peluang crossover", 0.0, 1.0, 0.8, 0.05),
                         mut=st.slider("Peluang mutasi", 0.0, 1.0, 0.6, 0.05),
                         elite=st.slider("Jumlah elit", 1, 8, 4),
                         tour=st.slider("Ukuran turnamen", 2, 6, 3))
                st.caption("Parameter notebook asli: 80 / 150 / 8 restart. Di server gratis akan lama.")
        st.markdown("### Kunci dan IV")
        st.text_input("Kunci AES (16 karakter)", key="key_text", max_chars=16)
        st.button("Acak kunci baru", on_click=random_key)
        iv_hex = st.text_input("IV (32 karakter hex)", "00112233445566778899aabbccddeeff", max_chars=32)
        st.markdown("### Uji gambar")
        max_side = st.select_slider("Sisi terpanjang gambar (px)", [64, 128, 192, 256, 384, 512], 128)
        n_trials = st.slider("Percobaan NPCR/UACI per gambar", 2, 20, 4)
        st.caption("Enkripsi memakai Python murni, jadi gambar besar jauh lebih lama.")

    if source == "S-Box bawaan" and st.session_state.get("src") != "bawaan":
        set_sbox(np.load(DEFAULT_SBOX), "S-Box bawaan dari repositori (parameter dan seed notebook).", "bawaan")
    sbox = st.session_state.get("sbox")

    # ---------- Hero
    status = "Belum ada S-Box. Pilih sumbernya di sidebar." if sbox is None else st.session_state["info"]
    st.markdown(
        f"<div class='hero'><div><h1>Rancang S-Box sendiri, lalu uji pada gambar.</h1>"
        f"<p>Algoritma genetika mencari S-Box 8-bit yang kuat. S-Box itu dipasang pada AES-128 "
        f"termodifikasi untuk mengenkripsi gambar, lalu hasilnya diukur.</p><small>{status}</small></div>"
        f"<div title='Setiap kotak adalah satu entri S-Box; warna makin tua berarti nilai makin besar'>"
        f"{sbox_grid(sbox, mini=True)}</div></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["1. S-Box", "2. Evaluasi", "3. Enkripsi gambar"])

    # ---------- Tab 1
    with tab1:
        if source == "Buat dengan GA":
            if st.button("Jalankan algoritma genetika", type="primary"):
                bar = st.progress(0.0, text="Memulai...")
                t0 = time.time()
                s, hist, score, seed, poly = C.run_ga(P, bar)
                bar.empty()
                set_sbox(s, f"S-Box hasil GA: restart terbaik {seed}, fitness {score:.5f}, "
                            f"polinomial {poly}, {time.time() - t0:.1f} detik.", "ga", hist)
                st.rerun()
        elif source == "Unggah file .npy":
            up = st.file_uploader("Pilih best_sbox.npy dari notebook", type=["npy"])
            if up is not None:
                try:
                    arr = np.load(io.BytesIO(up.read()), allow_pickle=False).astype(np.uint8)
                    if C.is_valid_sbox(arr):
                        if st.session_state.get("src") != f"npy:{up.name}:{up.size}":
                            set_sbox(arr, f"S-Box dimuat dari {up.name}.", f"npy:{up.name}:{up.size}")
                            st.rerun()
                    else:
                        st.error("File ini bukan permutasi lengkap 0 sampai 255. Pastikan memakai best_sbox.npy.")
                except Exception as exc:
                    st.error(f"File tidak bisa dibaca: {exc}")
        if sbox is None:
            st.info("Pilih S-Box bawaan, jalankan GA, atau unggah file .npy di sidebar.")
        else:
            st.markdown("#### Tabel S-Box (heksadesimal)")
            st.caption("Baris = digit pertama input, kolom = digit kedua. Arahkan kursor ke kotak untuk melihat nilai desimal.")
            st.markdown(sbox_grid(sbox), unsafe_allow_html=True)
            hist = st.session_state.get("history")
            if hist is not None:
                st.markdown("#### Perkembangan fitness")
                st.line_chart(hist.set_index("generation")[["best_fitness", "mean_fitness"]], color=[COBALT, "#9AA5BF"])
            inv = np.empty(256, dtype=np.uint8)
            inv[sbox] = np.arange(256, dtype=np.uint8)
            c1, c2, _ = st.columns([1, 1, 2])
            b = io.BytesIO(); np.save(b, sbox)
            c1.download_button("Unduh best_sbox.npy", b.getvalue(), "best_sbox.npy")
            b = io.BytesIO(); np.save(b, inv)
            c2.download_button("Unduh inverse_sbox.npy", b.getvalue(), "inverse_sbox.npy")

    # ---------- Tab 2
    with tab2:
        if sbox is None:
            st.info("Siapkan S-Box di tab 1 terlebih dahulu.")
        else:
            if st.button("Hitung semua metrik", type="primary"):
                with st.spinner("Menghitung NL, SAC, BIC, LAP, dan DAP..."):
                    st.session_state["metrics"] = C.calculate_metrics(sbox)
            m = st.session_state.get("metrics")
            if m is None:
                st.caption("Perhitungan memakan beberapa detik.")
            else:
                bal = all(int(np.sum((sbox >> k) & 1)) == 128 for k in range(8))
                uniq = len(np.unique(sbox))
                rows = [
                    ("Balance", "128 : 128" if bal else "tidak seimbang", "Setiap bit output berisi 128 nol dan 128 satu", "ok" if bal else "bad"),
                    ("Bijective", f"{uniq} / 256", "Semua nilai berbeda, tidak ada yang hilang", "ok" if uniq == 256 else "bad"),
                    ("NL", m["NL"], "Target 112 (makin besar makin baik)", "ok" if m["NL"] >= 112 else "bad"),
                    ("SAC", f"{m['SAC_deviation']:.5f}", "Deviasi dari 0,5; makin kecil makin baik", "info"),
                    ("BIC-NL", m["BIC_NL"], "Target 112", "ok" if m["BIC_NL"] >= 112 else "bad"),
                    ("BIC-SAC", f"{m['BIC_SAC_deviation']:.5f}", "Deviasi dari 0,5; makin kecil makin baik", "info"),
                    ("LAP", f"{m['LAP']:.5f}", "Batas praktis 8-bit: 0,0625", "ok" if m["LAP"] <= 0.0625 else "bad"),
                    ("DAP", f"{m['DAP']:.5f}", "Batas praktis 4/256 = 0,015625 (batas bawah teori 2/256)", "ok" if m["DAP"] <= 4 / 256 else "bad"),
                ]
                st.markdown(verdict_rows(rows), unsafe_allow_html=True)
                st.markdown("#### Matriks SAC")
                st.caption("Peluang bit output berubah saat satu bit input dibalik. Ideal 0,5; warna kemerahan berarti jauh dari 0,5.")
                st.markdown(sac_grid(m["SAC_matrix"]), unsafe_allow_html=True)

    # ---------- Tab 3
    with tab3:
        if sbox is None:
            st.info("Siapkan S-Box di tab 1 terlebih dahulu.")
        else:
            key = st.session_state["key_text"].encode()
            try:
                iv = bytes.fromhex(iv_hex)
            except ValueError:
                iv = b""
            if len(key) != 16:
                st.warning("Kunci harus tepat 16 karakter (huruf latin, angka, atau simbol ASCII).")
            if len(iv) != 16:
                st.warning("IV harus 32 karakter heksadesimal (0-9, a-f).")
            files = st.file_uploader("Unggah gambar (boleh lebih dari satu)", accept_multiple_files=True,
                                     type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
            if files and len(key) == 16 and len(iv) == 16 and st.button("Enkripsi dan uji", type="primary"):
                cipher, res, bar = C.SBoxAES(sbox), [], st.progress(0.0)
                for i, f in enumerate(files):
                    bar.progress(i / len(files), text=f"Memproses {f.name}...")
                    res.append(C.encrypt_one_image(f, cipher, key, iv, max_side, n_trials))
                bar.empty()
                st.session_state["enc_results"] = res

            res = st.session_state.get("enc_results")
            if res:
                st.caption(f"Nilai ideal: NPCR {C.NPCR_IDEAL:.4f}%, UACI {C.UACI_IDEAL:.4f}%, entropi 8 bit/byte.")
                for r in res:
                    row = r["row"]
                    with st.container(border=True):
                        st.markdown(f"#### {r['name']}  \n<span style='color:#6b7794'>{row['width']} x {row['height']} px</span>",
                                    unsafe_allow_html=True)
                        a, b, c = st.columns(3)
                        a.image(r["original"], caption="Asli", width="stretch")
                        b.image(r["cipher_img"], caption="Terenkripsi", width="stretch")
                        c.image(r["dec_img"], caption="Hasil dekripsi", width="stretch")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Dekripsi", "Identik" if row["decryption_success"] else "Gagal")
                        m2.metric("Entropi cipher", f"{row['cipher_entropy']:.4f}", f"asli {row['original_entropy']:.4f}", delta_color="off")
                        m3.metric("NPCR", f"{row['NPCR mean (%)']:.3f}%", f"lulus {row['NPCR lulus']}", delta_color="off")
                        m4.metric("UACI", f"{row['UACI mean (%)']:.3f}%", f"lulus {row['UACI lulus']}", delta_color="off")
                        with st.expander("Histogram RGB"):
                            fig = C.hist_figure(r["original"], r["cipher_img"], r["name"])
                            st.pyplot(fig)
                            C.plt.close(fig)
                df = pd.DataFrame([r["row"] for r in res])
                with st.expander("Tabel hasil lengkap"):
                    st.dataframe(df, width="stretch")
                zb = io.BytesIO()
                with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as z:
                    for r in res:
                        stem = r["name"].rsplit(".", 1)[0]
                        z.writestr(f"{stem}_encrypted.png", C.png_bytes(r["cipher_img"]))
                        z.writestr(f"{stem}_decrypted.png", C.png_bytes(r["dec_img"]))
                        z.writestr(f"{stem}.bin", r["enc_bytes"])
                    z.writestr("image_results.csv", df.to_csv(index=False))
                    nb = io.BytesIO(); np.save(nb, sbox); z.writestr("best_sbox.npy", nb.getvalue())
                st.download_button("Unduh hasil (ZIP)", zb.getvalue(), "hasil_enkripsi.zip", mime="application/zip")


main()
