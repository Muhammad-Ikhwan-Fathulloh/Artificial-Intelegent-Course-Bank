# Praktik Pandas: Manipulasi Data dan Feature Engineering

Panduan praktik langkah demi langkah. Ikuti urutan dari atas ke bawah, salin kode ke satu notebook (Google Colab atau Jupyter), jalankan, lalu cocokkan hasilnya dengan bagian **Output yang diharapkan** dan kerjakan bagian **Coba Sendiri**.

## Informasi Umum

| Item | Keterangan |
| --- | --- |
| Dataset | Transaksi e-commerce NusaShop (simulasi, 500 baris × 10 kolom) |
| Estimasi waktu | 2–3 jam |
| Prasyarat | Python dasar, serta dasar `pandas` dan `numpy` |

**Catatan:** Jalankan sel secara berurutan. Banyak langkah memakai kolom yang dibuat pada langkah sebelumnya.

## Daftar Langkah

- Langkah 0: Persiapan
- Langkah 1: Membuat dataset
- Langkah 2: Inspeksi data
- Langkah 3: Seleksi data
- Langkah 4: Manipulasi kolom
- Langkah 5: Groupby dan agregasi
- Langkah 6: Merge dan pivot
- Langkah 7: Transformasi string
- Langkah 8: Transformasi datetime
- Langkah 9: Binning
- Langkah 10: Encoding
- Langkah 11: Scaling
- Langkah 12: Feature extraction
- Langkah 13: Seleksi fitur dengan korelasi
- Langkah 14: Pipeline end-to-end
- Latihan mandiri

## Langkah 0: Persiapan

Instal pustaka yang diperlukan:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Impor pustaka dan atur tampilan grafik:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11
print('Library berhasil diimport')
```

**Output yang diharapkan:** `Library berhasil diimport`

## Langkah 1: Membuat Dataset

Dataset sengaja dibuat "kotor": nama kota tidak konsisten, tanggal masih berupa string, dan ada rating kosong.

| Kolom | Deskripsi |
| --- | --- |
| `order_id` | ID transaksi |
| `user_id` | ID pelanggan |
| `nama_produk` | Nama produk beserta kode (contoh: `Laptop-PRD001`) |
| `kategori` | Kategori produk |
| `harga` | Harga satuan (Rp) |
| `qty` | Jumlah unit |
| `kota` | Kota pengiriman (terdapat typo) |
| `tanggal_order` | Tanggal order (string) |
| `status` | Status pengiriman |
| `rating` | Rating 1–5 (terdapat NaN) |

```python
np.random.seed(42)
n = 500

kategori_list = ['Elektronik', 'Fashion', 'Makanan', 'Kecantikan', 'Olahraga']
kota_kotor    = ['Jakarta', 'jakarta', ' Bandung', 'Bandung ', 'Surabaya',
                 'SURABAYA', 'Medan', 'medan', 'Yogyakarta', 'yogyakarta']
status_list   = ['Selesai', 'Dikirim', 'Diproses', 'Dibatalkan']

produk_map = {
    'Elektronik': ['Laptop-PRD001', 'Smartphone-PRD002', 'Earbuds-PRD003', 'Tablet-PRD004'],
    'Fashion':    ['Kaos-PRD005', 'Celana-PRD006', 'Jaket-PRD007', 'Sepatu-PRD008'],
    'Makanan':    ['Kopi-PRD009', 'Snack-PRD010', 'Minyak-PRD011', 'Beras-PRD012'],
    'Kecantikan': ['Serum-PRD013', 'Lipstik-PRD014', 'Sunscreen-PRD015', 'Toner-PRD016'],
    'Olahraga':   ['Dumbell-PRD017', 'Matras-PRD018', 'Sepatu Lari-PRD019', 'Botol-PRD020'],
}
harga_map = {
    'Elektronik': (500000, 8000000), 'Fashion': (50000, 500000),
    'Makanan': (15000, 150000),      'Kecantikan': (50000, 600000),
    'Olahraga': (30000, 1500000),
}

kategori_col = np.random.choice(kategori_list, n)
nama_produk  = [np.random.choice(produk_map[k]) for k in kategori_col]
harga_col    = np.array([np.random.uniform(*harga_map[k]) for k in kategori_col]).round(-3)

tanggal_base = pd.date_range('2023-01-01', '2023-12-31', periods=n)
tanggal_acak = pd.to_datetime(
    np.random.choice(tanggal_base.astype(np.int64), n)
).strftime('%Y-%m-%d')

rating_col = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan], n,
                              p=[0.05, 0.07, 0.15, 0.38, 0.30, 0.05])

df = pd.DataFrame({
    'order_id':      [f'ORD-{str(i+1).zfill(4)}' for i in range(n)],
    'user_id':       [f'USR-{str(np.random.randint(1,101)).zfill(3)}' for _ in range(n)],
    'nama_produk':   nama_produk,
    'kategori':      kategori_col,
    'harga':         harga_col,
    'qty':           np.random.randint(1, 6, n),
    'kota':          np.random.choice(kota_kotor, n),
    'tanggal_order': tanggal_acak,
    'status':        np.random.choice(status_list, n, p=[0.6, 0.2, 0.12, 0.08]),
    'rating':        rating_col,
})

# Tabel diskon untuk latihan merge
df_diskon = pd.DataFrame({
    'kategori':      kategori_list,
    'persen_diskon': [0.10, 0.15, 0.05, 0.20, 0.08],
    'batas_min_qty': [1, 2, 3, 1, 2],
})

print(f'Dataset: {df.shape[0]} baris x {df.shape[1]} kolom')
df.head(8)
```

**Output yang diharapkan:** `Dataset: 500 baris x 10 kolom`. Baris pertama berisi `ORD-0001`, `USR-016`, `Lipstik-PRD014`, `Kecantikan`, dan harga `373000.0`.

## Langkah 2: Inspeksi Data

```python
df.info()
df.describe(include='all')
```

Temuan yang perlu Anda identifikasi:

| Temuan | Masalah |
| --- | --- |
| `rating` hanya 476 non-null | Ada 24 nilai kosong (missing value) |
| `tanggal_order` bertipe `object` | Belum bertipe datetime |
| `kota` memiliki 10 nilai unik | Seharusnya hanya 5 kota |

**Coba Sendiri:** tampilkan jumlah missing value per kolom dengan `df.isnull().sum()`.

## Langkah 3: Seleksi Data

### 3.1 `loc` dan `iloc`

|  | `loc` | `iloc` |
| --- | --- | --- |
| Dasar | Label | Posisi integer |
| Batas akhir slice | Inklusif | Eksklusif |

```python
print(df.loc[0:4, 'nama_produk':'qty'])
print(df.iloc[10:15, 0:4])
```

**Output yang diharapkan:** `loc` menampilkan 5 baris (index 0–4), `iloc` menampilkan 5 baris (index 10–14) dengan 4 kolom pertama.

### 3.2 Boolean Filtering

Gunakan `&` (AND), `|` (OR), dan `~` (NOT). Selalu beri tanda kurung pada setiap kondisi.

```python
# Elektronik dengan harga > 1 juta
mahal = df[(df['kategori'] == 'Elektronik') & (df['harga'] > 1_000_000)]
print(f'Elektronik > 1 juta: {len(mahal)}')

# Order Selesai atau Dikirim
aktif = df[df['status'].isin(['Selesai', 'Dikirim'])]
print(f'Order aktif: {len(aktif)}')

# Rating di bawah 3
buruk = df[df['rating'] < 3]
print(f'Rating < 3: {len(buruk)}')
```

**Output yang diharapkan:** `104`, `401`, `56`.

**Coba Sendiri:** tampilkan order berstatus `Dibatalkan` dengan `qty` ≥ 4.

## Langkah 4: Manipulasi Kolom

### 4.1 Menambah Kolom

```python
df['total_harga'] = df['harga'] * df['qty']
df['ppn']         = df['total_harga'] * 0.11
df['harga_final'] = df['total_harga'] + df['ppn']

df[['harga', 'qty', 'total_harga', 'ppn', 'harga_final']].head()
```

**Output yang diharapkan:** baris 0 memiliki `total_harga = 1492000.0`, `ppn = 164120.0`, dan `harga_final = 1656120.0`.

### 4.2 Sorting

```python
df.sort_values('total_harga', ascending=False)[
    ['order_id', 'nama_produk', 'harga', 'qty', 'total_harga']
].head()
```

**Output yang diharapkan:** transaksi terbesar adalah `ORD-0024` (Smartphone-PRD002) senilai `39340000.0`.

### 4.3 `apply` dan Fungsi Custom

Hindari loop `for` manual pada DataFrame; gunakan `apply`.

```python
def label_transaksi(total):
    if total >= 1_000_000:
        return 'Besar'
    elif total >= 200_000:
        return 'Sedang'
    else:
        return 'Kecil'

df['label_transaksi'] = df['total_harga'].apply(label_transaksi)
print(df['label_transaksi'].value_counts())
```

**Output yang diharapkan:** Besar 252, Sedang 185, Kecil 63.

**Coba Sendiri:** buat kolom `is_mahal` (True jika `harga` > 1 juta) menggunakan `apply(lambda ...)`.

## Langkah 5: Groupby dan Agregasi

Konsep: **Split** (pecah per grup), **Apply** (hitung), **Combine** (gabungkan hasil).

### 5.1 Statistik per Kategori

```python
penjualan_kategori = df.groupby('kategori').agg(
    total_revenue = ('total_harga', 'sum'),
    rata_harga    = ('harga', 'mean'),
    jumlah_order  = ('order_id', 'count'),
    rata_rating   = ('rating', 'mean'),
).round(0)

print(penjualan_kategori.sort_values('total_revenue', ascending=False))
```

**Output yang diharapkan:** Elektronik teratas (sekitar 1,34 miliar), Makanan terendah (sekitar 23 juta).

### 5.2 Visualisasi

```python
ax = penjualan_kategori['total_revenue'].sort_values().plot(
    kind='barh', color='#FF4500', edgecolor='white'
)
ax.set_title('Total Revenue per Kategori', fontsize=14, fontweight='bold')
ax.set_xlabel('Total Revenue (Rp)')
for bar in ax.patches:
    ax.text(bar.get_width() + 1e6, bar.get_y() + bar.get_height()/2,
            f'Rp {bar.get_width()/1e6:.0f}jt', va='center', fontsize=10)
plt.tight_layout()
plt.show()
```

### 5.3 Statistik per User

```python
user_stats = df.groupby('user_id').agg(
    total_belanja = ('total_harga', 'sum'),
    jumlah_order  = ('order_id', 'count'),
    avg_order     = ('total_harga', 'mean'),
    rata_rating   = ('rating', 'mean'),
).round(0)

print(f'Jumlah user unik: {len(user_stats)}')
user_stats.sort_values('total_belanja', ascending=False).head()
```

**Output yang diharapkan:** 98 user unik; teratas `USR-058` dengan total sekitar 62,8 juta.

**Coba Sendiri:** hitung total revenue per `status`.

## Langkah 6: Merge dan Pivot

### 6.1 Merge dengan Tabel Diskon

`how='left'` mempertahankan semua baris transaksi.

```python
df_merged = pd.merge(df, df_diskon, on='kategori', how='left')

df_merged['dapat_diskon'] = df_merged['qty'] >= df_merged['batas_min_qty']
df_merged['nilai_diskon'] = np.where(
    df_merged['dapat_diskon'],
    df_merged['total_harga'] * df_merged['persen_diskon'],
    0
)
df_merged['harga_setelah_diskon'] = df_merged['harga_final'] - df_merged['nilai_diskon']

print(f'Jumlah baris setelah merge: {len(df_merged)}')
df_merged[['order_id', 'kategori', 'total_harga', 'persen_diskon',
           'dapat_diskon', 'nilai_diskon', 'harga_setelah_diskon']].head(6)
```

**Checkpoint:** jumlah baris harus tetap 500. Jika bertambah, ada kunci duplikat pada tabel diskon.

### 6.2 Pivot Table

```python
pivot = pd.pivot_table(
    df, values='total_harga',
    index='kategori', columns='status',
    aggfunc='sum', fill_value=0
).round(0)
print(pivot)

plt.figure(figsize=(10, 4))
sns.heatmap(pivot/1e6, annot=True, fmt='.0f', cmap='Oranges',
            linewidths=0.5, cbar_kws={'label': 'Revenue (juta Rp)'})
plt.title('Heatmap Revenue per Kategori x Status (juta Rp)', fontweight='bold')
plt.tight_layout()
plt.show()
```

**Coba Sendiri:** buat pivot rata-rata `rating` per `kategori` × `status`.

## Langkah 7: Transformasi String

### 7.1 Temukan Masalahnya

```python
print(sorted(df['kota'].unique()))
print(f"Jumlah nilai unik: {df['kota'].nunique()}")
```

**Output yang diharapkan:** 10 nilai unik, termasuk `' Bandung'`, `'Bandung '`, `'SURABAYA'`, dan `'medan'`.

### 7.2 Bersihkan

```python
df['kota_bersih'] = df['kota'].str.strip().str.title()
print(sorted(df['kota_bersih'].unique()))
```

**Output yang diharapkan:** `['Bandung', 'Jakarta', 'Medan', 'Surabaya', 'Yogyakarta']`.

### 7.3 Ekstraksi dengan Regex

```python
df['kode_produk'] = df['nama_produk'].str.extract(r'(PRD\d+)')
df['nama_bersih'] = df['nama_produk'].str.replace(r'-PRD\d+', '', regex=True)
df[['nama_produk', 'nama_bersih', 'kode_produk']].head(6)
```

### 7.4 Pencarian Kata

```python
df['is_elektronik_premium'] = df['nama_bersih'].str.contains(
    'Laptop|Smartphone|Tablet', case=False
)
print(df['is_elektronik_premium'].value_counts())
```

**Output yang diharapkan:** False 416, True 84.

**Coba Sendiri:** buat kolom `nama_huruf_besar` berisi `nama_bersih` dalam huruf kapital (`.str.upper()`).

## Langkah 8: Transformasi Datetime

### 8.1 Konversi Tipe

```python
print(df['tanggal_order'].dtype)               # object
df['tanggal_order'] = pd.to_datetime(df['tanggal_order'])
print(df['tanggal_order'].dtype)               # datetime64[ns]
```

### 8.2 Ekstraksi Komponen

```python
df['tahun']       = df['tanggal_order'].dt.year
df['bulan']       = df['tanggal_order'].dt.month
df['hari']        = df['tanggal_order'].dt.day
df['nama_bulan']  = df['tanggal_order'].dt.month_name()
df['hari_minggu'] = df['tanggal_order'].dt.day_name()
df['kuartal']     = df['tanggal_order'].dt.quarter
df['is_weekend']  = df['tanggal_order'].dt.dayofweek >= 5   # Sabtu=5, Minggu=6

df[['tanggal_order', 'tahun', 'bulan', 'nama_bulan', 'hari_minggu', 'kuartal', 'is_weekend']].head(6)
```

### 8.3 Visualisasi Tren

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

df.groupby('bulan')['order_id'].count().plot(
    kind='bar', ax=axes[0], color='#FF4500', edgecolor='white')
axes[0].set_title('Jumlah Order per Bulan', fontweight='bold')
axes[0].tick_params(axis='x', rotation=0)

df.groupby('is_weekend')['total_harga'].mean().plot(
    kind='bar', ax=axes[1], color=['#033047', '#FF4500'], edgecolor='white')
axes[1].set_title('Rata-rata Transaksi: Weekday vs Weekend', fontweight='bold')
axes[1].set_xticklabels(['Weekday', 'Weekend'], rotation=0)

plt.tight_layout()
plt.show()
```

**Coba Sendiri:** hitung jumlah order per `hari_minggu`. Hari apa yang paling ramai?

## Langkah 9: Binning

`pd.cut` memakai batas manual, sedangkan `pd.qcut` membagi berdasarkan kuantil sehingga jumlah data per kelompok relatif sama.

```python
# Segmen harga
df['segmen_harga'] = pd.cut(
    df['harga'],
    bins=[0, 100_000, 500_000, 2_000_000, float('inf')],
    labels=['Budget', 'Mid-Range', 'Premium', 'Luxury']
)
print(df['segmen_harga'].value_counts().sort_index())

# Volume order
df['volume_order'] = pd.cut(
    df['qty'], bins=[0, 1, 3, 5], labels=['Eceran', 'Normal', 'Grosir']
)
print(df['volume_order'].value_counts().sort_index())

# Periode kuartal
df['periode'] = pd.cut(
    df['bulan'], bins=[0, 3, 6, 9, 12],
    labels=['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Okt-Des)']
)
```

**Output yang diharapkan (segmen harga):** Budget 88, Mid-Range 212, Premium 110, Luxury 90.

**Coba Sendiri:** bagi `harga` menjadi 4 kelompok dengan `pd.qcut`, lalu bandingkan jumlah datanya dengan `pd.cut`.

## Langkah 10: Encoding

| Jenis data | Teknik |
| --- | --- |
| Ordinal (ada urutan) | Map manual atau Label Encoding |
| Nominal (tanpa urutan) | One-Hot Encoding |

```python
# Ordinal: status
status_order = {'Dibatalkan': 0, 'Diproses': 1, 'Dikirim': 2, 'Selesai': 3}
df['status_enc'] = df['status'].map(status_order)
print(df['status_enc'].value_counts().sort_index())

# Nominal: kategori
ohe_kategori = pd.get_dummies(df['kategori'], prefix='kat')
print(ohe_kategori.head(6))

# Nominal: kota dengan drop_first
ohe_kota = pd.get_dummies(df['kota_bersih'], prefix='kota', drop_first=True)
print(list(ohe_kota.columns))
```

**Output yang diharapkan:** `status_enc` bernilai 0: 38, 1: 61, 2: 114, 3: 287. Kolom kota: `['kota_Jakarta', 'kota_Medan', 'kota_Surabaya', 'kota_Yogyakarta']`.

**Dummy variable trap:** saat dipakai untuk model, gunakan `drop_first=True` agar kolom tidak saling bergantung sempurna.

**Coba Sendiri:** lakukan One-Hot Encoding pada `hari_minggu`.

## Langkah 11: Scaling

|  | Min-Max | Standard (Z-score) |
| --- | --- | --- |
| Rumus | (x − min) / (max − min) | (x − μ) / σ |
| Hasil | Rentang 0–1 | Rata-rata 0, standar deviasi 1 |

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

fitur_numerik = ['harga', 'qty', 'total_harga']

df_mm = pd.DataFrame(
    MinMaxScaler().fit_transform(df[fitur_numerik]),
    columns=[f'{c}_minmax' for c in fitur_numerik]
)
df_std = pd.DataFrame(
    StandardScaler().fit_transform(df[fitur_numerik]),
    columns=[f'{c}_zscore' for c in fitur_numerik]
)

pd.concat([df[fitur_numerik].head(), df_mm.head(), df_std.head()], axis=1).round(3)
```

Visualisasi sebelum dan sesudah scaling:

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, fitur_numerik):
    ax.hist(df[col], bins=30, alpha=0.5, color='#033047', label='Original')
    ax.hist(df_mm[f'{col}_minmax'], bins=30, alpha=0.5, color='#FF4500', label='Min-Max')
    ax.set_title(f'Distribusi: {col}', fontweight='bold')
    ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
```

**Checkpoint:** semua kolom `_minmax` bernilai 0–1; kolom `_zscore` memiliki rata-rata mendekati 0 dan standar deviasi mendekati 1.

**Data leakage:** pada proyek nyata, lakukan `fit` scaler hanya pada data train, lalu `transform` pada data test.

**Coba Sendiri:** periksa `df_mm.min()` dan `df_mm.max()`, serta `df_std.mean()` dan `df_std.std()`.

## Langkah 12: Feature Extraction

Membuat fitur baru dari kombinasi kolom yang sudah ada.

```python
df['margin_estimasi']  = (df['total_harga'] * 0.25).round(0)
df['is_repeat_buyer']  = df.groupby('user_id')['order_id'].transform('count') > 3
df['total_spend_user'] = df.groupby('user_id')['total_harga'].transform('sum')
df['avg_spend_user']   = df.groupby('user_id')['total_harga'].transform('mean').round(0)

df[['user_id', 'total_harga', 'total_spend_user', 'avg_spend_user', 'is_repeat_buyer']].head(8)
```

Segmentasi user berdasarkan total belanja:

```python
df['segmen_user'] = pd.qcut(df['total_spend_user'], q=3,
                            labels=['Bronze', 'Silver', 'Gold'])

print(df['segmen_user'].value_counts())
print(df.groupby('segmen_user')['total_harga'].mean().round(0))
```

**Output yang diharapkan:** rata-rata transaksi naik dari Bronze (sekitar 1,25 juta) ke Silver (sekitar 3,38 juta) ke Gold (sekitar 6,13 juta).

**Catatan:** `transform` mengembalikan hasil agregasi dengan jumlah baris yang sama seperti data asli, sehingga bisa langsung disimpan sebagai kolom baru. Ini berbeda dengan `agg`.

**Coba Sendiri:** buat fitur `selisih_dari_rata_user` = `total_harga` − `avg_spend_user`.

## Langkah 13: Seleksi Fitur dengan Korelasi

```python
numerik_cols = ['harga', 'qty', 'total_harga', 'ppn', 'harga_final',
                'rating', 'status_enc', 'bulan', 'is_weekend',
                'total_spend_user', 'avg_spend_user']

corr_matrix = df[numerik_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, square=True)
plt.title('Heatmap Korelasi Fitur Numerik', fontweight='bold')
plt.tight_layout()
plt.show()

korelasi_target = (corr_matrix['total_harga']
                   .drop('total_harga').abs().sort_values(ascending=False))
print(korelasi_target)
print('\nKorelasi > 0.3:', korelasi_target[korelasi_target > 0.3].index.tolist())
```

**Output yang diharapkan:** `ppn` dan `harga_final` bernilai 1.0, `harga` sekitar 0.87, `avg_spend_user` sekitar 0.40, `total_spend_user` sekitar 0.31.

**Diskusi:** mengapa `ppn` dan `harga_final` berkorelasi 1.0 dengan `total_harga`? Apakah keduanya layak dipakai bersamaan sebagai fitur?

## Langkah 14: Pipeline End-to-End

Gabungkan semua langkah menjadi satu fungsi agar konsisten dan dapat dipakai ulang pada data baru.

```python
def fe_pipeline(df_raw):
    df_out = df_raw.copy()

    # 1. Type casting
    df_out['tanggal_order'] = pd.to_datetime(df_out['tanggal_order'])

    # 2. String cleaning
    df_out['kota'] = df_out['kota'].str.strip().str.title()

    # 3. Ekstraksi datetime
    df_out['bulan']      = df_out['tanggal_order'].dt.month
    df_out['kuartal']    = df_out['tanggal_order'].dt.quarter
    df_out['is_weekend'] = (df_out['tanggal_order'].dt.dayofweek >= 5).astype(int)

    # 4. Kombinasi kolom
    df_out['total_harga']      = df_out['harga'] * df_out['qty']
    df_out['total_spend_user'] = df_out.groupby('user_id')['total_harga'].transform('sum')
    df_out['is_repeat_buyer']  = (df_out.groupby('user_id')['order_id'].transform('count') > 3).astype(int)

    # 5. Binning
    df_out['segmen_harga'] = pd.cut(
        df_out['harga'], bins=[0, 100_000, 500_000, 2_000_000, float('inf')],
        labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])

    # 6. Encoding
    df_out['status_enc'] = df_out['status'].map(
        {'Dibatalkan': 0, 'Diproses': 1, 'Dikirim': 2, 'Selesai': 3})
    ohe_cols   = pd.get_dummies(df_out[['kategori', 'kota']], drop_first=True).astype(int)
    ohe_segmen = pd.get_dummies(df_out['segmen_harga'], prefix='seg', drop_first=True).astype(int)
    df_out = pd.concat([df_out, ohe_cols, ohe_segmen], axis=1)

    # 7. Scaling
    scale_cols = ['harga', 'qty', 'total_harga', 'total_spend_user']
    df_out[[f'{c}_scaled' for c in scale_cols]] = MinMaxScaler().fit_transform(df_out[scale_cols])

    # 8. Pilih fitur akhir
    fitur_akhir = (
        ['harga_scaled', 'qty_scaled', 'total_harga_scaled', 'total_spend_user_scaled',
         'bulan', 'kuartal', 'is_weekend', 'is_repeat_buyer', 'status_enc', 'rating']
        + list(ohe_cols.columns) + list(ohe_segmen.columns)
    )
    return df_out[fitur_akhir]


df_ml = fe_pipeline(df)
print(f'Shape: {df_ml.shape}')
print(list(df_ml.columns))
print(df_ml.isnull().sum()[lambda s: s > 0])
df_ml.head()
```

**Output yang diharapkan:** shape `(500, 21)`; hanya `rating` yang masih memiliki NaN (24 baris).

Tangani missing value sebelum data masuk ke model:

```python
df_ml['rating'] = df_ml['rating'].fillna(df_ml['rating'].median())
print(df_ml.isnull().sum().sum())   # harus 0
```

**Checkpoint akhir:** `df_ml` berukuran 500 × 21, semua fitur numerik, dan tanpa missing value.

## Latihan Mandiri

Kerjakan latihan berikut menggunakan dataset dan kolom yang telah dibuat pada langkah-langkah di atas.

### Soal 1: Seleksi dan Filtering

Tampilkan 10 produk **Fashion** dengan harga 100.000–400.000, status **Selesai**, dan rating ≥ 4. Urutkan `harga` dari tertinggi ke terendah.

### Soal 2: Groupby dan Agregasi

Per `kota_bersih`, hitung total revenue, rata-rata rating, jumlah order, dan persentase order berstatus `Selesai`. Urutkan dari revenue tertinggi.

### Soal 3: String dan Datetime

1. Buat kolom `jam_kategori`: Senin–Jumat diberi nilai `Weekday`, Sabtu–Minggu diberi nilai `Weekend`.
2. Buat pivot table rata-rata `total_harga` per `kategori` × `jam_kategori`.

### Soal 4: Feature Engineering

1. Buat `harga_per_unit` = `harga` / `qty`.
2. Buat `is_high_value` bernilai True jika `total_harga` di atas rata-rata keseluruhan.
3. Lakukan One-Hot Encoding pada `segmen_user`.
4. Terapkan `StandardScaler` pada `harga`, `qty`, dan `total_harga`.

### Tantangan: Analisis Korelasi Fitur

1. Dari `df_ml`, hitung korelasi semua fitur terhadap `total_harga_scaled`.
2. Tampilkan 5 fitur dengan korelasi absolut tertinggi.
3. Buat bar chart horizontal dari kelima fitur tersebut.
4. Interpretasikan: fitur apa yang paling berpengaruh, dan mengapa?

## Rangkuman

| Topik | Teknik | Fungsi Python |
| --- | --- | --- |
| Seleksi | loc, iloc, Boolean | `df.loc[]`, `df.iloc[]`, `df[kondisi]` |
| Kolom | Tambah, sort, custom | `df['baru']=`, `.sort_values()`, `.apply()` |
| Agregasi | Split-Apply-Combine | `.groupby().agg()` |
| Gabung dan reshape | Join, pivot | `pd.merge()`, `pd.pivot_table()` |
| String | Cleaning, regex | `.str.strip()`, `.str.extract()`, `.str.contains()` |
| Datetime | Komponen waktu | `.dt.year`, `.dt.month`, `.dt.dayofweek` |
| Binning | Numerik ke kategori | `pd.cut()`, `pd.qcut()` |
| Encoding | Kategori ke numerik | `.map()`, `pd.get_dummies()` |
| Scaling | Normalisasi | `MinMaxScaler`, `StandardScaler` |
| Extraction | Fitur baru | `.transform()`, kombinasi kolom |

### Poin Penting

1. **Bersihkan data terlebih dahulu.** String yang tidak konsisten dan tipe data yang salah adalah sumber error nomor satu.
2. **Fit scaler hanya pada data train** untuk menghindari data leakage.
3. Gunakan **`drop_first=True`** pada One-Hot Encoding.
4. Waspadai **fitur redundan**, misalnya `ppn` dan `harga_final` terhadap `total_harga`.
5. Fitur yang baik lahir dari **domain knowledge dan kreativitas**.
6. **Pipeline** yang konsisten membuat hasil reproducible dan siap produksi.

## Referensi

- Repo materi: https://github.com/Muhammad-Ikhwan-Fathulloh/Artificial-Intelegent-Course-Bank
- Dokumentasi Pandas: https://pandas.pydata.org/docs/
- Dokumentasi scikit-learn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
