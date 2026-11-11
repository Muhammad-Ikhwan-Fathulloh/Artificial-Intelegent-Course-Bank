# UTS Machine Learning: Klasifikasi

**Nama:** ______________________________
**NIM:** ______________________________
**Kelas:** ______________________________

---

## Ketentuan Umum

### Perhitungan kasus dari NIM

Langkah menentukan kasus dari NIM:

1. Jumlahkan seluruh digit NIM.
2. Hitung sisa bagi jumlah tersebut dengan 10 (mod 10).
3. Jika sisanya 0, kasus = 10. Selain itu, kasus = sisanya.
4. Kerjakan hanya kasus dengan nomor tersebut.

**Contoh untuk NIM 19552011024:**

1. Jumlah digit = 1+9+5+5+2+0+1+1+0+2+4 = 30
2. 30 mod 10 = 0
3. Sisa 0, maka kasus = 10
4. Kerjakan **Kasus 10** (`kasus10_titanic.csv`, target `survived`).

### Pemilihan dataset berdasarkan NIM

Dataset yang dikerjakan ditentukan oleh **jumlah seluruh digit NIM, lalu sisa bagi 10 (mod 10)**. Jika sisanya 0, gunakan Kasus 10.

| Sisa jumlah digit NIM mod 10 | Kasus    | Dataset                       | Target                            |
| ---------------------------- | -------- | ----------------------------- | --------------------------------- |
| 1                            | Kasus 1  | `kasus01_wine.csv`            | `kultivar`                        |
| 2                            | Kasus 2  | `kasus02_breast_cancer.csv`   | `diagnosis`                       |
| 3                            | Kasus 3  | `kasus03_penguins.csv`        | `species`                         |
| 4                            | Kasus 4  | `kasus04_birthwt.csv`         | `low`                             |
| 5                            | Kasus 5  | `kasus05_seattle_weather.csv` | `weather`                         |
| 6                            | Kasus 6  | `kasus06_cars.csv`            | `Origin`                          |
| 7                            | Kasus 7  | `kasus07_pima_diabetes.csv`   | `type`                            |
| 8                            | Kasus 8  | `kasus08_infert.csv`          | `case`                            |
| 9                            | Kasus 9  | `kasus09_mode_choice.csv`     | `mode` (diturunkan dari `choice`) |
| 0                            | Kasus 10 | `kasus10_titanic.csv`         | `survived`                        |

**NIM 19552011024**: jumlah digitnya 1+9+5+5+2+0+1+1+0+2+4 = 30, dan 30 mod 10 = 0, sehingga mengerjakan **Kasus 10** (`kasus10_titanic.csv`).

Soal untuk kesepuluh kasus disertakan di bawah. Hanya kerjakan kasus yang sesuai dengan NIM Anda.

### Tahapan wajib

Setiap kasus harus mencakup keenam tahapan berikut:

1. **Feature engineering.** Audit data (duplikat, missing value, nilai tidak wajar, tipe data), lalu buat fitur baru dan ubah fitur yang ada. Jelaskan setiap keputusan.
2. **Model klasifikasi.** Bandingkan minimal dua algoritma (misalnya Logistic Regression dan Random Forest) menggunakan pipeline scikit-learn.
3. **Metrik evaluasi.** Laporkan metrik yang sesuai dengan jenis klasifikasi dan jelaskan maknanya.
4. **Deploy dengan joblib.** Simpan pipeline lengkap beserta metadata yang dibutuhkan aplikasi.
5. **Aplikasi Streamlit.** Buat aplikasi yang menerima input mentah, menampilkan prediksi beserta probabilitasnya, dan menampilkan metrik evaluasi. File `app.py` wajib ada.
6. **Notebook Google Colab.** Notebook yang melakukan training dan menjalankan aplikasi Streamlit di Colab wajib ada.

### Aturan yang wajib dipatuhi

- Lakukan pembagian data train dan test **sebelum** imputasi, scaling, atau encoding. Semua statistik preprocessing harus dihitung dari data train saja.
- Tentukan sendiri kolom yang layak dijadikan fitur, dan jelaskan kolom mana yang dibuang beserta alasannya (misalnya ID, kolom yang hanya diketahui setelah kejadian, atau kolom yang redundan).
- Evaluasi model pada data test sebanyak satu kali. Pemilihan model dilakukan dengan cross-validation di data train.
- Simpan model di folder `models/` dan jangan menulis kredensial atau path absolut di dalam kode.

### Yang dikumpulkan

- `fitur.py`: fungsi feature engineering.
- `latih.py`: skrip pelatihan, evaluasi, dan penyimpanan model.
- `models/*.joblib`: hasil penyimpanan pipeline (dibuat ulang saat `latih.py` dijalankan).
- `app.py` (**wajib**): aplikasi Streamlit.
- Notebook Google Colab untuk training dan aplikasi (`notebook/<kasus>_colab.ipynb`) (**wajib**).
- Laporan singkat (1–2 halaman) berisi temuan feature engineering, perbandingan model, hasil metrik, dan interpretasi.

---

## Kasus 1: Jenis Anggur dari Analisis Kimia

**File:** `kasus01_wine.csv` · **Jumlah data:** 178 · **Target:** `kultivar` (Kultivar A, B, C)

**Kondisi NIM:** dikerjakan jika jumlah digit NIM mod 10 = 1.

Contoh: NIM 19552011025 → kasus 1.

NIM 19552011024 (jumlah digit 30, mod 10 = 0): tidak dikerjakan.

**Konteks:** Anggur dari tiga kultivar di satu wilayah dianalisis kadar 13 komponen kimianya. Tentukan kultivar sebuah sampel anggur berdasarkan hasil analisis tersebut.

| Kolom                                                                    | Keterangan                                          |
| ------------------------------------------------------------------------ | --------------------------------------------------- |
| `alcohol`                                                                | Kadar alkohol                                       |
| `malic_acid`                                                             | Asam malat                                          |
| `ash`, `alcalinity_of_ash`, `magnesium`                                  | Kadar abu, alkalinitas abu, dan magnesium           |
| `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins` | Komponen fenolik                                    |
| `color_intensity`, `hue`, `od280_od315_of_diluted_wines`                 | Intensitas dan warna anggur, serta rasio absorbansi |
| `proline`                                                                | Kadar prolin                                        |

**Tugas khusus:**
- Periksa korelasi antarfitur. Buat fitur rasio dari pasangan fitur yang saling terkait, lalu putuskan apakah fitur asli perlu dibuang.
- Terapkan scaling dengan benar, karena skala fitur sangat berbeda.

---

## Kasus 2: Diagnosis Kanker Payudara

**File:** `kasus02_breast_cancer.csv` · **Jumlah data:** 569 · **Target:** `diagnosis` (Ganas, Jinak)

**Kondisi NIM:** dikerjakan jika jumlah digit NIM mod 10 = 2.

Contoh: NIM 19552011026 → kasus 2.

NIM 19552011024 (jumlah digit 30, mod 10 = 0): tidak dikerjakan.

**Konteks:** Setiap baris adalah hasil pengukuran sel dari sampel jaringan payudara. Tentukan apakah sampel tersebut ganas atau jinak.

**Kamus kolom:** 30 fitur numerik. Setiap ukuran tersedia dalam tiga bentuk: `mean_` (rata-rata sel), `radius_error`/`texture_error`/dan seterusnya (galat standar), dan `worst_` (nilai terburuk). Ukuran dasarnya adalah radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, dan fractal_dimension.

**Tugas khusus:**
- Banyak fitur saling berkorelasi tinggi. Jelaskan strategi yang Anda pilih: seleksi fitur, reduksi dimensi, atau keduanya.
- Pada konteks medis, kesalahan jenis apa yang lebih mahal? Pilih metrik utama dan jelaskan alasannya.

---

## Kasus 3: Spesies Penguin

**File:** `kasus03_penguins.csv` · **Jumlah data:** 344 · **Target:** `species` (Adelie, Chinstrap, Gentoo)

**Kondisi NIM:** dikerjakan jika jumlah digit NIM mod 10 = 3.

Contoh: NIM 19552011027 → kasus 3.

NIM 19552011024 (jumlah digit 30, mod 10 = 0): tidak dikerjakan.

**Konteks:** Data pengukuran penguin dari tiga pulau di Antarktika. Tentukan spesies penguin berdasarkan ciri fisik dan kondisi pengamatan.

| Kolom                             | Keterangan                       |
| --------------------------------- | -------------------------------- |
| `island`                          | Pulau tempat pengamatan          |
| `bill_length_mm`, `bill_depth_mm` | Panjang dan kedalaman paruh (mm) |
| `flipper_length_mm`               | Panjang sirip (mm)               |
| `body_mass_g`                     | Massa tubuh (gram)               |
| `sex`                             | Jenis kelamin (ada nilai kosong) |
| `year`                            | Tahun pengamatan                 |

**Tugas khusus:**
- Tangani nilai kosong pada `sex` dan kolom numerik. Jelaskan pilihan Anda.
- Buat fitur rasio dari panjang dan kedalaman paruh. Putuskan apakah kolom `year` layak dijadikan fitur.

---

## Kasus 4: Bayi dengan Berat Lahir Rendah

**File:** `kasus04_birthwt.csv` · **Jumlah data:** 189 · **Target:** `low` (1 = berat lahir rendah, kurang dari 2.500 gram; 0 = tidak)

**Kondisi NIM:** dikerjakan jika jumlah digit NIM mod 10 = 4.

Contoh: NIM 19552011028 → kasus 4.

NIM 19552011024 (jumlah digit 30, mod 10 = 0): tidak dikerjakan.

**Konteks:** Data 189 kelahiran di sebuah rumah sakit. Tentukan apakah bayi akan lahir dengan berat rendah berdasarkan kondisi ibu sebelum atau selama kehamilan.

| Kolom   | Keterangan                                             |
| ------- | ------------------------------------------------------ |
| `age`   | Usia ibu (tahun)                                       |
| `lwt`   | Berat badan ibu saat haid terakhir (pon)               |
| `race`  | Ras ibu: 1 = kulit putih, 2 = kulit hitam, 3 = lainnya |
| `smoke` | Merokok saat hamil (0 atau 1)                          |
| `ptl`   | Jumlah kelahiran prematur sebelumnya                   |
| `ht`    | Riwayat hipertensi (0 atau 1)                          |
| `ui`    | Adanya iritabilitas uterus (0 atau 1)                  |
| `ftv`   | Jumlah kunjungan dokter pada trimester pertama         |
| `bwt`   | Berat lahir bayi dalam gram                            |

**Tugas khusus:**
- Periksa setiap kolom dan tentukan apakah kolom tersebut tersedia sebelum target diketahui. Jelaskan kolom yang Anda buang.
- Kolom `race` berisi kode numerik. Tentukan apakah kolom ini harus diperlakukan sebagai numerik atau kategorik, lalu jelaskan.
- Pada konteks ini, kesalahan jenis apa yang lebih mahal? Pilih metrik utama dan threshold, lalu jelaskan alasannya.

---

## Kasus 5: Kondisi Cuaca Harian di Seattle

**File:** `kasus05_seattle_weather.csv` · **Jumlah data:** 1.461 · **Target:** `weather` (drizzle, fog, rain, snow, sun)

**Kondisi NIM:** dikerjakan jika jumlah digit NIM mod 10 = 5.

Contoh: NIM 19552011029 → kasus 5.

NIM 19552011024 (jumlah digit 30, mod 10 = 0): tidak dikerjakan.

**Konteks:** Catatan cuaca harian di Seattle beberapa tahun. Tentukan kategori cuaca suatu hari berdasarkan pengukurannya.

| Kolom                  | Keterangan                     |
| ---------------------- | ------------------------------ |
| `date`                 | Tanggal (format `YYYY-MM-DD`)  |
| `precipitation`        | Curah hujan (mm)               |
| `temp_max`, `temp_min` | Suhu maksimum dan minimum (°C) |
| `wind`                 | Kecepatan angin (km/jam)       |

**Tugas khusus:**
- Dari `date`, buat fitur bulan, hari dalam setahun, dan musim. Jelaskan mengapa fitur ini dapat membantu.
- Kelas `snow` dan `drizzle` jarang. Tentukan cara menangani ketidakseimbangan kelas, dan pilih metrik yang tepat.

---

## Kasus 6: Negara Asal Mobil

**File:** `kasus06_cars.csv` · **Jumlah data:** 406 · **Target:** `Origin` (USA, Europe, Japan)

**Kondisi NIM:** dikerjakan jika jumlah digit NIM mod 10 = 6.

Contoh: NIM 19552011020 → kasus 6.

NIM 19552011024 (jumlah digit 30, mod 10 = 0): tidak dikerjakan.

**Konteks:** Data spesifikasi mobil dari tahun 1970 hingga 1982. Tentukan negara asal mobil berdasarkan spesifikasinya.

| Kolom              | Keterangan                                                   |
| ------------------ | ------------------------------------------------------------ |
| `Name`             | Nama lengkap mobil, misalnya "chevrolet chevelle malibu"     |
| `Miles_per_Gallon` | Konsumsi bahan bakar (mil per galon)                         |
| `Cylinders`        | Jumlah silinder                                              |
| `Displacement`     | Volume silinder                                              |
| `Horsepower`       | Tenaga mesin (ada nilai kosong)                              |
| `Weight_in_lbs`    | Berat (pon)                                                  |
| `Acceleration`     | Waktu akselerasi                                             |
| `Year`             | Tanggal dalam format `YYYY-MM-DD`, yang mewakili tahun model |

**Tugas khusus:**
- Dari `Name`, ekstrak merek mobil. Jelaskan apakah merek bisa menjadi fitur yang valid.
- Dari `Year`, buat fitur tahun. Tangani nilai kosong pada `Horsepower`.

---

## Kasus 7: Diabetes pada Wanita Pima

**File:** `kasus07_pima_diabetes.csv` · **Jumlah data:** 200 · **Target:** `type` (Yes, No)

**Kondisi NIM:** dikerjakan jika jumlah digit NIM mod 10 = 7.

Contoh: NIM 19552011021 → kasus 7.

NIM 19552011024 (jumlah digit 30, mod 10 = 0): tidak dikerjakan.

**Konteks:** Data wanita keturunan Pima yang diperiksa untuk diabetes. Tentukan apakah seorang wanita menderita diabetes berdasarkan hasil pemeriksaan.

| Kolom   | Keterangan                                                 |
| ------- | ---------------------------------------------------------- |
| `npreg` | Jumlah kehamilan                                           |
| `glu`   | Konsentrasi glukosa plasma pada tes toleransi glukosa oral |
| `bp`    | Tekanan darah diastolik                                    |
| `skin`  | Ketebalan lipatan kulit triceps                            |
| `bmi`   | Indeks massa tubuh                                         |
| `ped`   | Fungsi silsilah diabetes (riwayat diabetes dalam keluarga) |
| `age`   | Usia                                                       |

**Tugas khusus:**
- Periksa apakah ada nilai yang tidak wajar secara medis. Jelaskan langkah yang Anda ambil.
- Buat fitur turunan dari dua atau lebih kolom yang relevan, lalu jelaskan dasar pemilihannya.

---

## Kasus 8: Studi Infertilitas

**File:** `kasus08_infert.csv` · **Jumlah data:** 248 · **Target:** `case` (1 = kasus, 0 = kontrol)

**Kondisi NIM:** dikerjakan jika jumlah digit NIM mod 10 = 8.

Contoh: NIM 19552011022 → kasus 8.

NIM 19552011024 (jumlah digit 30, mod 10 = 0): tidak dikerjakan.

**Konteks:** Studi kasus-kontrol tentang faktor risiko infertilitas. Tentukan apakah seorang wanita termasuk kelompok kasus berdasarkan riwayat kehamilannya.

| Kolom                       | Keterangan                                                  |
| --------------------------- | ----------------------------------------------------------- |
| `education`                 | Tingkat pendidikan: 0-5 tahun, 6-11 tahun, 12 tahun ke atas |
| `age`                       | Usia                                                        |
| `parity`                    | Jumlah kehamilan yang pernah dilahirkan                     |
| `induced`                   | Jumlah aborsi yang diinduksi                                |
| `spontaneous`               | Jumlah keguguran spontan                                    |
| `stratum`, `pooled.stratum` | Nomor kelompok pencocokan (matching)                        |

**Tugas khusus:**
- Jelaskan apakah `stratum` dan `pooled.stratum` boleh dijadikan fitur, dan mengapa.
- Buat fitur turunan dari riwayat kehamilan, misalnya total kehamilan atau total kehilangan kehamilan. Jelaskan manfaatnya.

---

## Kasus 9: Pilihan Moda Transportasi

**File:** `kasus09_mode_choice.csv` · **Jumlah data:** 840 baris (210 individu × 4 alternatif) · **Target:** kolom `mode` dari baris yang `choice = 1`

**Kondisi NIM:** dikerjakan jika jumlah digit NIM mod 10 = 9.

Contoh: NIM 19552011023 → kasus 9.

NIM 19552011024 (jumlah digit 30, mod 10 = 0): tidak dikerjakan.

**Konteks:** Survei perjalanan non-bisnis antara Sydney, Canberra, dan Melbourne pada 1987. Setiap individu memilih satu dari empat moda: pesawat, mobil, bus, atau kereta. Dokumentasi sumber tidak menjelaskan urutan kode numerik moda (1–4).

Format data bersifat **panjang**: setiap individu memiliki empat baris, satu per moda. Target tidak tersedia langsung sebagai satu kolom per individu.

| Kolom        | Keterangan                            |
| ------------ | ------------------------------------- |
| `individual` | ID individu                           |
| `mode`       | Kode moda (1–4)                       |
| `choice`     | 1 jika moda ini dipilih, 0 jika tidak |
| `ttme`       | Waktu tempuh terminal                 |
| `invc`       | Biaya dalam kendaraan                 |
| `invt`       | Waktu dalam kendaraan                 |
| `gc`         | Biaya gabungan (generalized cost)     |
| `hinc`       | Pendapatan rumah tangga               |
| `psize`      | Ukuran rombongan                      |

**Tugas khusus:**
- Ubah data dari format panjang menjadi satu baris per individu. Atribut moda (`ttme`, `invc`, `invt`, `gc`) menjadi kolom terpisah untuk setiap moda. Jelaskan langkah Anda.
- Pastikan setiap individu hanya memiliki satu target, lalu periksa keseimbangan kelasnya.
- Jelaskan mengapa perubahan format ini diperlukan sebelum model dibuat, dan bagaimana aplikasi Streamlit akan menerima input dalam format ini.

---

## Kasus 10: Penumpang Titanic

**File:** `kasus10_titanic.csv` · **Jumlah data:** 2.201 · **Target:** `survived` (Yes, No)

**Kondisi NIM:** dikerjakan jika jumlah digit NIM mod 10 = 0 (sisa 0 menjadi Kasus 10).

Contoh: NIM 19552011024 → kasus 10.

NIM 19552011024 (jumlah digit 30, mod 10 = 0): dikerjakan.

**Konteks:** Data penumpang kapal Titanic. Tentukan apakah seorang penumpang selamat berdasarkan kelas, jenis kelamin, dan usianya.

| Kolom      | Keterangan                                |
| ---------- | ----------------------------------------- |
| `class`    | Kelas penumpang: 1st, 2nd, 3rd, atau Crew |
| `sex`      | Jenis kelamin                             |
| `age`      | Child atau Adult                          |
| `survived` | Yes jika selamat, No jika tidak           |

**Tugas khusus:**
- Buat fitur interaksi dari kombinasi kelas, jenis kelamin, dan usia. Jelaskan apakah interaksi ini memberi informasi tambahan.
- Ubah data menjadi pilihan kategori yang sesuai untuk model, lalu jelaskan encoding yang Anda pakai.