# Tugas Hands-On 3: Pandas Lanjutan

Tugas mandiri lanjutan dari **Hands-On Session 3** (Part 1: manipulasi data dan feature engineering, Part 2: data cleaning, time series, RFM, dan preprocessing). Pilih **satu** dari sepuluh dataset di bawah, lalu ulangi alur Part 2 pada dataset tersebut.

## Tujuan

Setelah mengerjakan tugas ini, Anda mampu:

- mengaudit kualitas data (duplikat, missing value, outlier, tipe data),
- membersihkan data dengan kode yang bisa diulang,
- menganalisis data dengan `groupby` lanjutan, reshape, dan time series (bila ada kolom tanggal),
- menyiapkan data untuk model tanpa data leakage (`train_test_split` + `ColumnTransformer`),
- mendokumentasikan hasil dalam CSV bersih dan kamus data.

## Prasyarat

- Part 1 dan Part 2 sudah dikerjakan (`Hands-On-Session3.md`, `Hands-On-Session3-Part2.md`).
- Python 3.10+ dengan `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (≥ 1.2).
- Google Colab atau Jupyter Notebook.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Pilihan Dataset

Pilih **1 dari 10** studi kasus. Buka tautan, lalu unduh lewat **File → Download → CSV** dan simpan dengan nama file pada tabel.

| # | Kasus | Sektor | File | Baris | Jenis tugas | Dataset |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | Credit Risk Prediction | Fintech | `case1_credit_risk.csv` | 2.500 | Klasifikasi biner | [Buka](https://docs.google.com/spreadsheets/d/16BvrUSi30QUc7AbzcVVuQyEiu0PZpr2BGK-ipSPQyGQ/edit?usp=sharing) |
| 02 | Customer Churn Analysis | Telekomunikasi | `case2_customer_churn.csv` | 5.000 | Klasifikasi biner | [Buka](https://docs.google.com/spreadsheets/d/1LVa38KCj7suilukuu3QuEenYDwgqsICnSwCWv4eIf3A/edit?usp=sharing) |
| 03 | Product Recommendation | E-Commerce | `case3_ecommerce_transactions.csv` | 10.000 | Klasifikasi biner + clustering | [Buka](https://docs.google.com/spreadsheets/d/1hLVdH6LyQWzUprUonRxpQB27ech7jQ4iZbi2_ZYtZVg/edit?usp=sharing) |
| 04 | Hospital Readmission | Healthcare | `case4_hospital_readmission.csv` | 1.500 | Klasifikasi biner | [Buka](https://docs.google.com/spreadsheets/d/1hLVdH6LyQWzUprUonRxpQB27ech7jQ4iZbi2_ZYtZVg/edit?usp=sharing) |
| 05 | Employee Attrition | HR Analytics | `case5_employee_attrition.csv` | 1.200 | Klasifikasi biner | [Buka](https://docs.google.com/spreadsheets/d/1otMROchKlVchMtEOIfcbhHGyeoPbliTRm4y9wA9mel0/edit?usp=sharing) |
| 06 | Traffic Accident Severity | Smart City | `case6_traffic_accident.csv` | 3.000 | Klasifikasi multi-kelas | [Buka](https://docs.google.com/spreadsheets/d/1zz6-KTh2kBeexOeAjFpByb-ez_z0FN1bhabecwpPbr0/edit?usp=sharing) |
| 07 | Fraud Detection | Banking | `case8_fraud_detection.csv` | 8.000 | Klasifikasi biner (imbalanced) | [Buka](https://docs.google.com/spreadsheets/d/1uZold6CUfeMdkUjjnintOM3DHqo0OQovYPO5V1RmAN0/edit?usp=sharing) |
| 08 | Student Performance | EdTech | `case9_student_performance.csv` | 800 | Klasifikasi biner | [Buka](https://docs.google.com/spreadsheets/d/1-7RVtr_yZdilG4nurXmSjvvTyDqBLgM7TWo2kmFe-Gg/edit?usp=sharing) |


Ini sebagai contoh jangan pilih ini
| 09 | House Price Prediction | Properti | `case7_house_price.csv` | 2.000 | Regresi | [Buka](https://docs.google.com/spreadsheets/d/1hIO4vVxjf4-K2YHq9PF9NHsY-7VUyOgqijhuntxj1Kk/edit?usp=sharing) |
| 11 | Sentiment Analysis UMKM | Retail | `case10_product_review.csv` | 6.000 | NLP multi-kelas | [Buka](https://docs.google.com/spreadsheets/d/1fqmF4T_j7K50HUVYb80pa2w4sRt_l7-5oxRfjAlJwIA/edit?usp=sharing) |

Alternatif: bangkitkan dataset NusaShop baru seperti pada Part 1 dengan `np.random.seed(2026)` dan `n = 2000`.

## Yang Dikerjakan

| Tahap | Tugas | Bobot* |
| --- | --- | --- |
| 1. Data | Muat dataset pilihan, tampilkan `shape`, `info()`, dan `describe()` | — |
| 2. Kualitas | Audit dengan fungsi `cek_kualitas`, lalu bersihkan: duplikat, missing value, outlier per grup, tipe data | 30% |
| 3. Analisis | Sesuaikan dengan data: `groupby` lanjutan, tren waktu bila ada kolom tanggal, RFM bila ada pelanggan dan transaksi. Tulis **3 temuan** yang bisa ditindaklanjuti | 30% |
| 4. Persiapan model | Split dengan `stratify` (klasifikasi), `ColumnTransformer`, tanpa leakage. Simpan CSV bersih dan kamus data | 25% |
| 5. Kerapian | Sel berurutan, ada penjelasan singkat tiap langkah, hasil dapat direproduksi | 15% |

\*Bobot adalah usulan dan dapat disesuaikan pengajar.

### Kerangka `cek_kualitas`

Sesuaikan nama kolom grup dan kunci dengan dataset Anda.

```python
def cek_kualitas(d, kolom_num, kolom_grup, kunci):
    """Ringkasan kualitas data: baris, duplikat, missing, dan outlier IQR per grup."""
    g  = d.groupby(kolom_grup, observed=True)[kolom_num]
    q1 = g.transform(lambda s: s.quantile(0.25))
    q3 = g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    outlier = (d[kolom_num] < q1 - 1.5 * iqr) | (d[kolom_num] > q3 + 1.5 * iqr)
    return pd.Series({
        'n_baris':    len(d),
        'n_duplikat': d.duplicated(subset=kunci).sum(),
        'n_missing':  d.isnull().sum().sum(),
        'n_outlier':  outlier.sum(),
    })

laporan = pd.DataFrame({
    'sebelum': cek_kualitas(df_mentah, 'kolom_numerik', 'kolom_grup', 'kolom_kunci'),
    'sesudah': cek_kualitas(df_bersih, 'kolom_numerik', 'kolom_grup', 'kolom_kunci'),
})
laporan
```

### Aturan Penting

1. **Urutan pembersihan:** duplikat → missing value → outlier → tipe data.
2. **Jangan menyunting data mentah secara manual.** Semua perubahan harus lewat kode.
3. **Split dulu, baru** imputasi, scaling, dan encoding. Imputer, scaler, dan encoder hanya di-`fit` pada data train.
4. Untuk target yang tidak seimbang (misalnya kasus 08), gunakan `stratify=y`.
5. Kasus 10 (teks): bagian yang tidak relevan, seperti RFM atau scaling pada kolom teks, boleh dilewati dengan alasan tertulis.

## Yang Dikumpulkan

```text
tugas-hands-on-3/
├── README.md                  # ringkasan: kasus yang dipilih, temuan, cara menjalankan
├── notebook.ipynb             # dapat dijalankan dari atas ke bawah
├── data/
│   └── <nama_file>.csv        # dataset asli
├── output/
│   ├── data_bersih.csv        # hasil pembersihan
│   └── kamus_data.csv         # kolom, tipe, n_missing, n_unik
└── laporan_kualitas.csv       # tabel sebelum vs sesudah
```

## Checklist Sebelum Mengumpulkan

- [ ] Notebook berjalan dari atas ke bawah tanpa error (*Restart & Run All*).
- [ ] Tabel kualitas sebelum vs sesudah ada dan diinterpretasikan.
- [ ] Setiap keputusan pembersihan (buang, isi, clip) disertai alasan singkat.
- [ ] Tidak ada data leakage pada tahap split dan preprocessing.
- [ ] Tiga temuan ditulis dalam kalimat yang bisa ditindaklanjuti.
- [ ] `data_bersih.csv` dan `kamus_data.csv` tersimpan.

## Referensi

- Repo materi: https://github.com/Muhammad-Ikhwan-Fathulloh/Artificial-Intelegent-Course-Bank
- Dokumentasi pandas: https://pandas.pydata.org/docs/user_guide/index.html
- Dokumentasi scikit-learn (ColumnTransformer): https://scikit-learn.org/stable/modules/compose.html
