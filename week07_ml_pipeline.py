# =============================================================
# SESI 7 — KONSEP ML, DATASET, PIPELINE & FEATURE ENGINEERING
# Mata Kuliah: Intelegensi Buatan (IF-030)
# Install: pip install scikit-learn pandas seaborn matplotlib
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                    PolynomialFeatures, LabelEncoder)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      GridSearchCV, StratifiedKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, confusion_matrix,
                               silhouette_score, accuracy_score)

# ─────────────────────────────────────────────
# BAGIAN 1: DATA SINTETIS — MAHASISWA
# ─────────────────────────────────────────────
print("=" * 58)
print(" BAGIAN 1: DATASET SINTETIS — MAHASISWA")
print("=" * 58)

np.random.seed(42)
N = 150

data_mhs = {
    'ipk':       np.random.uniform(1.5, 4.0, N).round(2),
    'absensi':   np.random.randint(0, 25, N),
    'tugas':     np.random.randint(40, 100, N),
    'kota':      np.random.choice(['Jakarta','Bandung','Surabaya','Medan'], N),
    'angkatan':  np.random.choice([2020, 2021, 2022, 2023], N),
}
df_mhs = pd.DataFrame(data_mhs)
df_mhs['lulus'] = (
    (df_mhs['ipk'] >= 2.75) &
    (df_mhs['absensi'] <= 12) &
    (df_mhs['tugas'] >= 60)
).astype(int)

# Tambah missing values acak
idx_nan = np.random.choice(N, 8, replace=False)
df_mhs.loc[idx_nan[:4], 'ipk']     = np.nan
df_mhs.loc[idx_nan[4:], 'absensi'] = np.nan

print(f"\n  Shape       : {df_mhs.shape}")
print(f"  Missing     : {df_mhs.isnull().sum().to_dict()}")
print(f"  Label dist  : {df_mhs['lulus'].value_counts().to_dict()}")
print(f"\n  5 baris pertama:")
print(df_mhs.head().to_string(index=False))


# ─────────────────────────────────────────────
# BAGIAN 2: PREPROCESSING MANUAL
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print(" BAGIAN 2: PREPROCESSING MANUAL (STEP BY STEP)")
print("=" * 58)

df_proc = df_mhs.copy()

# 2a. Imputasi
print(f"\n  ── Sebelum Imputasi ──")
print(f"  IPK missing   : {df_proc['ipk'].isnull().sum()}")
print(f"  Absensi missing: {df_proc['absensi'].isnull().sum()}")

df_proc['ipk']     = df_proc['ipk'].fillna(df_proc['ipk'].median())
df_proc['absensi'] = df_proc['absensi'].fillna(df_proc['absensi'].median())

print(f"\n  ── Setelah Imputasi (median) ──")
print(f"  IPK missing   : {df_proc['ipk'].isnull().sum()}")
print(f"  Absensi missing: {df_proc['absensi'].isnull().sum()}")

# 2b. Feature Engineering
df_proc['ipk_kat']    = pd.cut(df_proc['ipk'],
                                bins=[0, 2.0, 2.75, 3.5, 4.0],
                                labels=['Buruk','Cukup','Baik','Sangat Baik'])
df_proc['kehadiran']  = 100 - (df_proc['absensi'] / 14 * 100).round(1)
df_proc['skor_total'] = (df_proc['ipk'] * 25 +
                          df_proc['tugas'] * 0.5 +
                          df_proc['kehadiran'] * 0.25).round(2)

print(f"\n  ── Feature Engineering ──")
print(df_proc[['ipk','ipk_kat','kehadiran','skor_total','lulus']].head(8).to_string(index=False))

# 2c. Encoding
le = LabelEncoder()
df_proc['kota_enc'] = le.fit_transform(df_proc['kota'])
ohe_kota = pd.get_dummies(df_proc['kota'], prefix='kota')
df_proc  = pd.concat([df_proc, ohe_kota], axis=1)

print(f"\n  ── One-Hot Encoding 'kota' ──")
print(df_proc[['kota'] + list(ohe_kota.columns)].head(5).to_string(index=False))

# 2d. Standarisasi
scaler = StandardScaler()
num_cols = ['ipk', 'absensi', 'tugas', 'kehadiran', 'skor_total']
print(f"\n  ── Standarisasi (z-score) ──")
print(f"  Sebelum: IPK mean={df_proc['ipk'].mean():.2f}, std={df_proc['ipk'].std():.2f}")
df_proc[num_cols] = scaler.fit_transform(df_proc[num_cols])
print(f"  Sesudah: IPK mean={df_proc['ipk'].mean():.4f}, std={df_proc['ipk'].std():.4f}")


# ─────────────────────────────────────────────
# BAGIAN 3: SKLEARN PIPELINE (CARA BENAR)
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print(" BAGIAN 3: SKLEARN PIPELINE (ANTI DATA LEAKAGE)")
print("=" * 58)

num_feats = ['ipk', 'absensi', 'tugas']
cat_feats = ['kota', 'angkatan']
X_raw = df_mhs[num_feats + cat_feats]
y_raw = df_mhs['lulus']

# BENAR: split SEBELUM preprocessing!
X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y_raw, test_size=0.25, random_state=42, stratify=y_raw)

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_feats),
    ('cat', cat_pipe, cat_feats),
])
pipe_knn = Pipeline([
    ('prep',  preprocessor),
    ('model', KNeighborsClassifier(n_neighbors=5)),
])

pipe_knn.fit(X_tr, y_tr)
acc_knn = pipe_knn.score(X_te, y_te)
cv_knn  = cross_val_score(pipe_knn, X_raw, y_raw, cv=5,
                            scoring='accuracy').mean()

print(f"\n  ✅ Pipeline KNN(k=5):")
print(f"     Test accuracy    : {acc_knn:.2%}")
print(f"     Cross-val (5-fold): {cv_knn:.2%}")

# Prediksi mahasiswa baru
baru = pd.DataFrame([{
    'ipk': 3.5, 'absensi': 4, 'tugas': 82,
    'kota': 'Bandung', 'angkatan': 2022
}])
pred_baru = pipe_knn.predict(baru)
prob_baru = pipe_knn.predict_proba(baru) if hasattr(pipe_knn, 'predict_proba') else None
print(f"\n  Mahasiswa baru (IPK=3.5, Absensi=4, Tugas=82):")
print(f"  → {'Lulus ✅' if pred_baru[0] else 'Tidak Lulus ❌'}")


# ─────────────────────────────────────────────
# BAGIAN 4: K-MEANS CLUSTERING
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print(" BAGIAN 4: K-MEANS CLUSTERING — SEGMENTASI PELANGGAN")
print("=" * 58)

# Data Pelanggan E-Commerce
np.random.seed(0)
n_cust = 250
cluster_profiles = [
    (25, 300,  15),   # muda, hemat, sering beli
    (40, 650,   8),   # dewasa, boros, sesekali
    (58, 420,   4),   # senior, moderat, jarang
]
usia, pengeluaran, frekuensi = [], [], []
for (mu_u, mu_p, mu_f) in cluster_profiles:
    n_i = n_cust // 3
    usia.extend(np.random.normal(mu_u, 4, n_i))
    pengeluaran.extend(np.random.normal(mu_p, 60, n_i))
    frekuensi.extend(np.random.normal(mu_f, 2, n_i))

df_cust = pd.DataFrame({
    'usia': np.array(usia).round(1),
    'pengeluaran_bulanan': np.array(pengeluaran).round(0),
    'frekuensi_beli': np.array(frekuensi).round(1),
}).clip(lower=0)

X_cust = StandardScaler().fit_transform(df_cust)

# Elbow + Silhouette
print("\n  ── Elbow Method & Silhouette ──")
inertia_list, sil_list = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cust)
    inertia_list.append(km.inertia_)
    sil_list.append(silhouette_score(X_cust, km.labels_))
    bar  = "█" * int(sil_list[-1] * 20)
    best = " ← OPTIMAL" if k == 3 else ""
    print(f"  k={k}  inertia={inertia_list[-1]:>7.0f}  "
          f"silhouette={sil_list[-1]:.3f}  {bar}{best}")

# Fit k=3
km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cust['cluster'] = km_final.fit_predict(X_cust)

# Label segmen
SEGMENT_NAMES = {
    0: "High Spender Dewasa",
    1: "Frequent Buyer Muda",
    2: "Senior Low Frequency",
}
df_cust['segmen'] = df_cust['cluster'].map(SEGMENT_NAMES)

print(f"\n  ── Profil Klaster (k=3) ──")
print(f"  {'Segmen':<24} {'n':>4} {'Usia':>6} {'Pengeluaran':>12} {'Frekuensi':>10}")
print(f"  {'─'*60}")
for c in range(3):
    sub  = df_cust[df_cust['cluster'] == c]
    name = SEGMENT_NAMES.get(c, f"Cluster {c}")
    print(f"  {name:<24} {len(sub):>4} {sub['usia'].mean():>6.1f}"
          f" {sub['pengeluaran_bulanan'].mean():>12.0f}"
          f" {sub['frekuensi_beli'].mean():>10.1f}")

print("\n  ── Rekomendasi Bisnis ──")
rekomendasi = {
    "High Spender Dewasa":    "Promo premium, exclusive offers",
    "Frequent Buyer Muda":    "Loyalty points, flash sale",
    "Senior Low Frequency":   "Reminder email, easy checkout",
}
for seg, rek in rekomendasi.items():
    print(f"  {seg:<24} → {rek}")


# ─────────────────────────────────────────────
# BAGIAN 5: FULL PIPELINE — TITANIC
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print(" BAGIAN 5: FULL PIPELINE — TITANIC DATASET")
print("=" * 58)

df_titanic = sns.load_dataset('titanic')

# Feature Engineering
df_titanic['family_size'] = df_titanic['sibsp'] + df_titanic['parch'] + 1
df_titanic['alone']       = (df_titanic['family_size'] == 1).astype(int)
df_titanic['fare_log']    = np.log1p(df_titanic['fare'])

fitur_num = ['age', 'fare_log', 'family_size', 'pclass']
fitur_cat = ['sex', 'embarked', 'alone']
target    = 'survived'

df_t = df_titanic[fitur_num + fitur_cat + [target]].dropna(subset=['embarked'])
X_t, y_t = df_t[fitur_num + fitur_cat], df_t[target]
X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(
    X_t, y_t, test_size=0.2, random_state=42, stratify=y_t)

pre_t = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
    ]), fitur_num),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), fitur_cat),
])

# Bandingkan beberapa model
models = {
    'KNN(k=5)':      KNeighborsClassifier(n_neighbors=5),
    'DecisionTree':  DecisionTreeClassifier(max_depth=5, random_state=42),
    'RandomForest':  RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':           SVC(kernel='rbf', C=1.0, random_state=42),
}

print(f"\n  {'─'*52}")
print(f"  {'Model':<16} {'Test Acc':>9} {'CV-5 Acc':>9} {'CV Std':>8}")
print(f"  {'─'*52}")

resultados = {}
for nama, model in models.items():
    pipe_t = Pipeline([('prep', pre_t), ('model', model)])
    pipe_t.fit(X_tr_t, y_tr_t)
    acc_t  = pipe_t.score(X_te_t, y_te_t)
    cv_t   = cross_val_score(pipe_t, X_t, y_t, cv=5)
    resultados[nama] = {'acc': acc_t, 'cv': cv_t.mean(), 'std': cv_t.std()}
    print(f"  {nama:<16} {acc_t:>8.2%}  {cv_t.mean():>8.2%}  {cv_t.std():>7.3f}")

print(f"  {'─'*52}")
best = max(resultados, key=lambda k: resultados[k]['cv'])
print(f"  🏆 Terbaik: {best} (CV={resultados[best]['cv']:.2%})")

# Feature Importance dari Random Forest
pipe_rf = Pipeline([
    ('prep',  pre_t),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42)),
])
pipe_rf.fit(X_tr_t, y_tr_t)
rf_model   = pipe_rf.named_steps['model']
ohe_names  = (pipe_rf.named_steps['prep']
              .named_transformers_['cat']
              .named_steps['ohe']
              .get_feature_names_out(fitur_cat))
feat_names = fitur_num + list(ohe_names)
importances = rf_model.feature_importances_

print(f"\n  ── Feature Importance (Top 8) ──")
idx_sorted = importances.argsort()[::-1][:8]
for i in idx_sorted:
    bar = "█" * int(importances[i] * 80)
    print(f"  {feat_names[i]:<22} {importances[i]:.4f}  {bar}")


# ─────────────────────────────────────────────
# BAGIAN 6: GRIDSEARCHCV — TUNING KNN
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print(" BAGIAN 6: HYPERPARAMETER TUNING — GRIDSEARCHCV")
print("=" * 58)

pipe_cv = Pipeline([
    ('prep',  pre_t),
    ('model', KNeighborsClassifier()),
])
param_grid = {
    'model__n_neighbors': [3, 5, 7, 9, 11, 15],
    'model__weights':     ['uniform', 'distance'],
    'model__metric':      ['euclidean', 'manhattan'],
}
gs = GridSearchCV(pipe_cv, param_grid, cv=5,
                  scoring='accuracy', n_jobs=-1, verbose=0)
gs.fit(X_tr_t, y_tr_t)

print(f"\n  Best params : {gs.best_params_}")
print(f"  Best CV acc : {gs.best_score_:.2%}")
print(f"  Test acc    : {gs.score(X_te_t, y_te_t):.2%}")


# ─────────────────────────────────────────────
# BAGIAN 7: VISUALISASI LENGKAP
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print(" BAGIAN 7: VISUALISASI")
print("=" * 58)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor("#141414")

# Plot 1: Distribusi label
ax = axes[0, 0]
ax.set_facecolor("#1c1c1c")
val_counts = df_mhs['lulus'].value_counts()
bars = ax.bar(['Tidak Lulus', 'Lulus'],
              [val_counts.get(0, 0), val_counts.get(1, 0)],
              color=['#FF6FA5', '#5BD68A'], edgecolor='#141414', linewidth=2)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
            str(int(b.get_height())), ha='center', color='white', fontsize=11)
ax.set_title("Distribusi Label — Mahasiswa", color='white', fontweight='bold')
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#444')

# Plot 2: Elbow Method
ax = axes[0, 1]
ax.set_facecolor("#1c1c1c")
ax.plot(list(K_range), inertia_list, 'o-',
        color='#5B9DFF', linewidth=2, markersize=7,
        markerfacecolor='#FFD93D', markeredgecolor='#141414')
ax.axvline(3, color='#5BD68A', linestyle='--', linewidth=1.5, label='k=3 optimal')
ax.set_title("Elbow Method — K-Means", color='white', fontweight='bold')
ax.set_xlabel("k", color='#aaa'); ax.set_ylabel("Inertia", color='#aaa')
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#444')
ax.legend(labelcolor='white', facecolor='#1c1c1c', edgecolor='#444')
ax.grid(True, color='#333', linewidth=0.5)

# Plot 3: Scatter cluster
ax = axes[0, 2]
ax.set_facecolor("#1c1c1c")
colors_clust = ['#FFD93D', '#FF6FA5', '#5B9DFF']
for c in range(3):
    mask = df_cust['cluster'] == c
    ax.scatter(df_cust.loc[mask, 'usia'],
               df_cust.loc[mask, 'pengeluaran_bulanan'],
               c=colors_clust[c], s=40, alpha=0.75,
               edgecolors='#141414', linewidths=0.5,
               label=SEGMENT_NAMES[c])
ax.set_title("K-Means Clustering Pelanggan", color='white', fontweight='bold')
ax.set_xlabel("Usia", color='#aaa'); ax.set_ylabel("Pengeluaran (Rp)", color='#aaa')
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#444')
ax.legend(labelcolor='white', facecolor='#1c1c1c',
          edgecolor='#444', fontsize=7)

# Plot 4: Model comparison
ax = axes[1, 0]
ax.set_facecolor("#1c1c1c")
model_names = list(resultados.keys())
acc_vals    = [resultados[m]['acc'] * 100 for m in model_names]
cv_vals     = [resultados[m]['cv']  * 100 for m in model_names]
x = np.arange(len(model_names))
w = 0.35
b1 = ax.bar(x - w/2, acc_vals, w, color='#5B9DFF',
            edgecolor='#141414', label='Test Acc')
b2 = ax.bar(x + w/2, cv_vals, w, color='#FFD93D',
            edgecolor='#141414', label='CV Acc')
ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=8)
ax.set_ylim(0, 110); ax.set_title("Perbandingan Model — Titanic",
                                   color='white', fontweight='bold')
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#444')
ax.legend(labelcolor='white', facecolor='#1c1c1c', edgecolor='#444')

# Plot 5: Feature Importance
ax = axes[1, 1]
ax.set_facecolor("#1c1c1c")
top8 = idx_sorted[:8]
top8_names = [feat_names[i][:14] for i in top8]
top8_vals  = [importances[i] for i in top8]
bars_fi = ax.barh(top8_names[::-1], top8_vals[::-1],
                   color='#5BD68A', edgecolor='#141414')
ax.set_title("Feature Importance (RF)", color='white', fontweight='bold')
ax.set_xlabel("Importance", color='#aaa')
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#444')

# Plot 6: Silhouette per k
ax = axes[1, 2]
ax.set_facecolor("#1c1c1c")
ax.plot(list(K_range), sil_list, 's-',
        color='#FF6FA5', linewidth=2, markersize=8,
        markerfacecolor='#FFD93D', markeredgecolor='#141414')
ax.axvline(3, color='#5BD68A', linestyle='--', linewidth=1.5, label='k=3 optimal')
ax.set_title("Silhouette Score vs k", color='white', fontweight='bold')
ax.set_xlabel("k", color='#aaa'); ax.set_ylabel("Silhouette", color='#aaa')
ax.tick_params(colors='white')
for s in ax.spines.values(): s.set_edgecolor('#444')
ax.legend(labelcolor='white', facecolor='#1c1c1c', edgecolor='#444')
ax.grid(True, color='#333', linewidth=0.5)

plt.suptitle("Sesi 7 — ML Pipeline, Clustering & Feature Engineering",
             color='white', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("week07_ml_pipeline.png", dpi=150,
            bbox_inches='tight', facecolor='#141414')
plt.show()
print("  📊 Grafik disimpan ke week07_ml_pipeline.png")

print("\n" + "=" * 58)
print("  ✅ SELESAI — Semua praktik Sesi 7 berhasil!")
print("=" * 58)
