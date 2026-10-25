# Tugas Mandiri Pandas – Cara Pengerjaan

Lanjutan **Hands-On Session 3 (Praktik Pandas)**. Anda akan berlatih membersihkan data kotor dan merekayasa fitur pada tujuh dataset baru (A–G). Soal lengkap ada di dokumen tugas; README ini menjelaskan cara mengerjakannya.

## 1. Isi paket

```
paket_mahasiswa_tugas_pandas/
├── README.md             <- file ini
├── data_dictionary.md    <- penjelasan kolom tiap dataset
└── datasets/
    ├── A_orders.csv, A_customers.csv, A_products.csv, A_promo.csv
    ├── B_hr_karyawan.csv
    ├── C_sensor_iot.csv
    ├── D_ulasan_produk.csv
    ├── E_properti_jabodetabek.csv
    ├── F_churn_telekomunikasi.csv
    └── G_nilai_mahasiswa.csv
```

Semua data bersifat sintetis (bukan data nyata atau pribadi) dan **sengaja kotor**.

## 2. Persiapan

Python 3.10+ dan Jupyter. Pasang library:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Letakkan folder `datasets/` sejajar dengan notebook Anda, lalu jalankan `jupyter notebook`.

## 3. Paket pengerjaan

| Paket | Dataset wajib | Perkiraan waktu |
|---|---|---|
| Inti | A, B, F | 6–8 jam |
| Lengkap | A sampai G | 14–18 jam |
| Proyek akhir | Satu dataset B, E, atau F | +4 jam |

Tanyakan kepada dosen paket mana yang berlaku untuk kelas Anda.

Penanda tingkat soal: **[D]** Dasar, **[M]** Menengah, **[T]** Tantangan. Kerjakan berurutan dalam satu dataset karena soal berikutnya memakai hasil soal sebelumnya.

## 4. Alur kerja per dataset

Buat satu notebook per dataset, misalnya `A_nusashop.ipynb`, dengan enam bagian ini:

1. **Inspeksi.** Muat data tanpa parameter tambahan, lalu jalankan `info()`, `describe(include='all')`, `isnull().sum()`, `duplicated().sum()`, dan `nunique()` untuk kolom kategori. Catat semua keanehan sebelum mengubah apa pun.
2. **Pembersihan.** Selesaikan satu masalah per sel. Setiap selesai, buktikan hasilnya (mis. `nunique()` turun dari 40 menjadi 8).
3. **Rekayasa fitur.** Buat kolom baru sesuai soal.
4. **Analisis.** Agregasi, pivot, dan grafik, masing-masing dengan satu kalimat interpretasi.
5. **Pipeline.** Rangkum langkah pembersihan dan fitur menjadi fungsi (pola Langkah 14).
6. **Kesimpulan.** Tiga sampai lima kalimat tentang temuan dan keterbatasan.

Sel pembuka yang disarankan:

```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import warnings; warnings.filterwarnings('ignore')

df_raw = pd.read_csv('datasets/A_orders.csv')   # tanpa parameter tambahan dulu
df = df_raw.copy()                              # jangan menimpa data mentah
df.info()
```

## 5. Tabel keputusan (wajib)

Pada bagian Pembersihan, buat satu sel Markdown berisi tabel ini dan isi setiap masalah yang Anda temukan:

| Kolom | Masalah | Baris terdampak | Tindakan | Alasan |
|---|---|---|---|---|
| contoh: `qty` | bernilai 0 dan negatif | (isi hitungan Anda) | dihapus | tidak mungkin pada transaksi nyata |

Banyak keputusan sah selama beralasan (menghapus, mengganti NaN, atau imputasi). Jawaban tanpa alasan tidak mendapat nilai penuh.

## 6. Aturan penting

- **Jangan menimpa data mentah.** Selalu bekerja pada `.copy()` dan simpan hasil ke file baru, mis. `A_orders_bersih.csv`.
- **Jalankan berurutan.** Banyak sel memakai kolom dari sel sebelumnya. Sebelum mengumpulkan, pilih `Kernel > Restart & Run All` dan pastikan tidak ada error.
- **Cegah kebocoran data.** Pada dataset B, E, dan F, lakukan `train_test_split(..., random_state=42)` lebih dulu, lalu `fit` scaler, encoder, dan nilai imputasi **hanya pada data latih**. Waspadai kolom yang baru diketahui setelah hasil terjadi.
- **Setelah `merge`, periksa jumlah baris.** Dengan `how='left'`, jumlahnya tidak boleh bertambah.
- **Grafik** harus berjudul, berlabel sumbu, dan diinterpretasi.
- **Kerja mandiri.** Diskusi konsep boleh, tetapi kode dan interpretasi harus milik sendiri. Bila memakai bantuan AI atau sumber lain, sebutkan di akhir notebook.

## 7. Tips debugging

| Gejala | Kemungkinan penyebab |
|---|---|
| `KeyError` | kolom dibuat di sel yang belum dijalankan, atau nama kolom salah ketik |
| `to_datetime` gagal atau hasil aneh | format tanggal campuran; periksa `dayfirst` dan `format='mixed'` |
| `ValueError: could not convert string to float` | masih ada `Rp`, titik ribuan, koma desimal, spasi, atau teks seperti `absen` |
| Jumlah baris bertambah setelah `merge` | kunci duplikat pada tabel kanan |
| `str` accessor error | kolom bertipe campuran; ubah dulu dengan `.astype(str)` atau tangani NaN |
| Hasil `groupby` berbeda dari teman | ejaan kategori belum dinormalisasi atau duplikat belum dihapus |

## 8. Pengumpulan

Kumpulkan dalam satu folder atau zip bernama `NIM_Nama_TugasPandas`:

- [ ] Notebook per dataset yang dikerjakan (`NIM_Nama_TugasPandas.ipynb` bila satu file)
- [ ] File hasil bersih (`*_bersih.csv`) bila diminta
- [ ] `pipeline.py` dan laporan 1–2 halaman (hanya untuk Proyek Akhir)
- [ ] Notebook lolos `Restart & Run All`
- [ ] Setiap keputusan pembersihan dijelaskan
- [ ] Scaler dan encoder hanya di-`fit` pada data latih
- [ ] Semua grafik berjudul, berlabel, dan diinterpretasi

## 9. Penilaian (100 poin)

| Komponen | Bobot |
|---|---|
| Inspeksi dan dokumentasi masalah | 15 |
| Kebenaran pembersihan | 25 |
| Kualitas rekayasa fitur | 20 |
| Penerapan teknik materi | 15 |
| Pipeline dan reproduksibilitas | 15 |
| Interpretasi dan komunikasi | 10 |

Pengurangan: kebocoran data (−10), notebook tidak bisa dijalankan ulang (−10), data mentah ditimpa (−5), jawaban tanpa penjelasan keputusan (−5 per dataset).

## 10. Referensi cepat

Rujuk kembali materi Hands-On Session 3 di Artificial-Intelegent-Course-Bank. Fungsi yang paling sering dipakai: `str.strip/lower/title/extract/replace/split`, `pd.to_datetime`, `.dt.*`, `pd.cut/qcut`, `groupby().agg/transform`, `pd.merge`, `pd.pivot_table`, `pd.melt`, `pd.get_dummies`, `MinMaxScaler`, `StandardScaler`, `resample`, `interpolate`.

Selamat mengerjakan.
