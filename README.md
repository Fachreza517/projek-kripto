# S-Box Genetic Algorithm dan Enkripsi Gambar

Projek Kriptografi Kelompok 9. Aplikasi Streamlit yang:

1. mencari S-Box 8-bit dengan Genetic Algorithm (atau memakai S-Box bawaan),
2. mengevaluasi S-Box (Balance, Bijective, NL, SAC, BIC-NL, BIC-SAC, LAP, DAP),
3. mengenkripsi dan mendekripsi gambar dengan AES-128 termodifikasi (S-Box GA, CBC dua arah),
   lalu mengukur entropi, NPCR, UACI, dan histogram.

## Isi repositori

| File | Fungsi |
|---|---|
| `app.py` | Antarmuka Streamlit (file utama untuk deploy) |
| `core.py` | Logika inti: metrik S-Box, GA, AES termodifikasi, metrik gambar |
| `best_sbox.npy` | S-Box bawaan (dibuat dengan parameter dan seed notebook) |
| `requirements.txt` | Daftar library |
| `.streamlit/config.toml` | Tema dan konfigurasi server |

## Menjalankan di komputer

```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Unggah semua file di atas (termasuk folder `.streamlit`) ke repositori GitHub.
2. Buka share.streamlit.io, pilih **Create app**, lalu isi repositori, branch `main`, dan Main file path `app.py`.
3. Klik **Deploy**.

## Catatan

- Enkripsi ditulis dengan Python murni, jadi gambar diperkecil otomatis (bisa diubah di sidebar).
- Ini AES termodifikasi dan mode CBC dua arah bukan standar. Untuk pemakaian nyata gunakan AES-GCM.
- Kunci bawaan hanya untuk demo; gunakan tombol **Acak kunci baru** untuk eksperimen.
