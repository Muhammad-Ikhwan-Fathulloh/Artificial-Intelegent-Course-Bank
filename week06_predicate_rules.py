# =============================================================
# SESI 6 — LOGIKA PREDIKAT, INFERENSI & RULE-BASED SYSTEM
# Mata Kuliah: Intelegensi Buatan (IF-030)
# Install: pip install scikit-learn matplotlib
# =============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json, itertools

# ─────────────────────────────────────────────
# BAGIAN 1: MESIN INFERENSI (FORWARD + BACKWARD)
# ─────────────────────────────────────────────
print("=" * 55)
print(" BAGIAN 1: MESIN INFERENSI LENGKAP")
print("=" * 55)


class InferenceEngine:
    """Mesin inferensi dengan forward chaining dan backward chaining."""

    def __init__(self, name="KB"):
        self.name   = name
        self.fakta  = set()
        self.aturan = []     # [(kondisi_list, hasil, nama)]
        self.log    = []

    def assert_fact(self, *fakta):
        for f in fakta:
            self.fakta.add(f)

    def add_rule(self, kondisi: list, hasil: str, nama: str = ""):
        self.aturan.append((kondisi, hasil, nama))

    # ── Forward Chaining ──
    def forward_chain(self, verbose=True) -> list:
        """Terus terapkan aturan sampai stabil (fixed point)."""
        self.log.clear()
        changed, iterasi = True, 0
        while changed:
            changed = False
            iterasi += 1
            for cond, hasil, nama in self.aturan:
                if all(k in self.fakta for k in cond) \
                   and hasil not in self.fakta:
                    self.fakta.add(hasil)
                    entry = f"[iter{iterasi}] FIRE '{nama}': {cond} → '{hasil}'"
                    self.log.append(entry)
                    if verbose:
                        print(f"  {entry}")
                    changed = True
        if verbose:
            print(f"\n  Total iterasi: {iterasi}, Fakta baru: {len(self.log)}")
        return self.log

    # ── Backward Chaining ──
    def backward_chain(self, goal: str, depth=0,
                       visited=None, verbose=True) -> bool:
        """Buktikan 'goal' dari fakta yang ada menggunakan aturan."""
        if visited is None:
            visited = set()
        indent = "  " * depth

        if goal in self.fakta:
            if verbose:
                print(f"{indent}✅ FACT: '{goal}'")
            return True

        if goal in visited:
            if verbose:
                print(f"{indent}🔄 LOOP: '{goal}' (abaikan)")
            return False
        visited.add(goal)

        for cond, hasil, nama in self.aturan:
            if hasil == goal:
                if verbose:
                    print(f"{indent}→ TRY rule '{nama}': {cond} → '{hasil}'")
                if all(self.backward_chain(k, depth+1, visited, verbose)
                       for k in cond):
                    self.fakta.add(goal)
                    if verbose:
                        print(f"{indent}✅ PROVED: '{goal}' via rule '{nama}'")
                    return True
        if verbose:
            print(f"{indent}❌ FAIL: '{goal}' tidak dapat dibuktikan")
        return False

    def print_kb(self):
        print(f"\n  ── {self.name}: Fakta Aktif ──")
        for f in sorted(self.fakta):
            print(f"  • {f}")
        print(f"\n  ── {self.name}: Aturan ({len(self.aturan)}) ──")
        for i, (cond, hasil, nama) in enumerate(self.aturan, 1):
            print(f"  R{i} [{nama}]: IF {cond} THEN '{hasil}'")


# Demo: Diagnosis Medis
print("\n── Demo: Diagnosis Medis ──")
kb_medis = InferenceEngine("Medis")
kb_medis.assert_fact("demam", "batuk", "pilek", "badan_pegal")

kb_medis.add_rule(["demam", "batuk"],            "infeksi",   "R1-infeksi")
kb_medis.add_rule(["infeksi", "pilek"],          "flu",       "R2-flu")
kb_medis.add_rule(["flu", "badan_pegal"],        "flu_berat", "R3-flu_berat")
kb_medis.add_rule(["demam", "nyeri_dada"],       "pneumonia", "R4-pneumonia")
kb_medis.add_rule(["infeksi", "badan_pegal"],    "kelelahan", "R5-kelelahan")

print("\n  [Forward Chaining]")
kb_medis.forward_chain()
kb_medis.print_kb()

# Reset untuk backward chaining
kb_bc = InferenceEngine("Medis-BC")
kb_bc.assert_fact("demam", "batuk", "pilek", "badan_pegal")
for rule in kb_medis.aturan:
    kb_bc.aturan.append(rule)

print("\n  [Backward Chaining: Buktikan 'flu_berat']")
hasil_bc = kb_bc.backward_chain("flu_berat")
print(f"\n  Kesimpulan: flu_berat dapat dibuktikan = {hasil_bc}")


# ─────────────────────────────────────────────
# BAGIAN 2: CERTAINTY FACTOR (CF)
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 2: CERTAINTY FACTOR (CF)")
print("=" * 55)


class SistemPakarCF:
    """Sistem pakar dengan Certainty Factor (CF 0.0–1.0)."""

    def __init__(self):
        self.fakta  = {}   # {nama: cf}
        self.aturan = []   # [(cond, hasil, cf_rule, nama)]

    def assert_fact(self, nama: str, cf: float = 1.0):
        self.fakta[nama] = max(self.fakta.get(nama, 0), cf)

    def add_rule(self, cond: list, hasil: str,
                 cf: float, nama: str = ""):
        self.aturan.append((cond, hasil, cf, nama))

    def get_cf(self, nama: str) -> float:
        return self.fakta.get(nama, 0.0)

    def forward_cf(self, verbose=True) -> list:
        """Forward chaining dengan propagasi CF."""
        log, changed = [], True
        while changed:
            changed = False
            for cond, hasil, cf_rule, nama in self.aturan:
                cf_cond = min(self.get_cf(c) for c in cond)
                if cf_cond > 0:
                    cf_new = round(cf_rule * cf_cond, 4)
                    if cf_new > self.fakta.get(hasil, 0):
                        self.fakta[hasil] = cf_new
                        entry = (f"  FIRE '{nama}': "
                                 f"CF={cf_rule}×{cf_cond:.2f} = {cf_new:.3f}")
                        log.append(entry)
                        if verbose:
                            print(entry)
                        changed = True
        return log

    def diagnosis(self, threshold=0.3, exclude_keys=None) -> list:
        """Kembalikan diagnosis yang CF-nya melebihi threshold."""
        exclude = set(exclude_keys or [])
        hasil = [(k, v) for k, v in self.fakta.items()
                 if k not in exclude and v < 1.0 and v >= threshold]
        return sorted(hasil, key=lambda x: -x[1])


# Demo Diagnosis
print("\n── Demo: Sistem Pakar CF ──")
sp = SistemPakarCF()
sp.assert_fact("demam",       cf=0.90)
sp.assert_fact("batuk",       cf=0.80)
sp.assert_fact("pilek",       cf=0.70)
sp.assert_fact("badan_pegal", cf=0.60)

sp.add_rule(["demam", "batuk"],            "infeksi",   0.80, "R1")
sp.add_rule(["infeksi", "pilek"],          "flu",        0.90, "R2")
sp.add_rule(["flu", "badan_pegal"],        "flu_berat",  0.70, "R3")
sp.add_rule(["demam", "batuk", "badan_pegal"], "kelelahan", 0.60, "R4")

sp.forward_cf()
print("\n── Diagnosis dengan Keyakinan (threshold=0.3) ──")
gejala_input = {"demam", "batuk", "pilek", "badan_pegal"}
for d, cf in sp.diagnosis(threshold=0.3, exclude_keys=gejala_input):
    bar  = "█" * int(cf * 20)
    lvl  = "TINGGI" if cf > 0.7 else "SEDANG" if cf > 0.4 else "RENDAH"
    flag = "⚠️ " if lvl == "TINGGI" else "  "
    print(f"  {flag}{d:<14} CF={cf:.3f}  {bar:<20} [{lvl}]")


# ─────────────────────────────────────────────
# BAGIAN 3: SISTEM PAKAR KERUSAKAN SMARTPHONE
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 3: SISTEM PAKAR KERUSAKAN SMARTPHONE")
print("=" * 55)


class PakarSmartphone(SistemPakarCF):
    """Sistem pakar diagnosis kerusakan smartphone."""

    GEJALA_LIST = [
        "layar_tidak_nyala", "baterai_cepat_habis",
        "tidak_bisa_charge", "suara_tidak_ada",
        "kamera_buram", "sinyal_hilang",
        "hp_panas", "storage_penuh", "restart_sendiri",
        "touch_tidak_respon", "wifi_tidak_konek",
    ]

    def __init__(self):
        super().__init__()
        self._setup_rules()

    def _setup_rules(self):
        rules = [
            (["layar_tidak_nyala", "baterai_cepat_habis"], "masalah_daya", 0.85, "R1"),
            (["masalah_daya", "tidak_bisa_charge"],        "baterai_rusak", 0.90, "R2"),
            (["masalah_daya", "hp_panas"],                 "overheat",      0.80, "R3"),
            (["suara_tidak_ada"],                          "speaker_rusak", 0.75, "R4"),
            (["kamera_buram"],                             "lensa_kotor",   0.60, "R5"),
            (["sinyal_hilang", "wifi_tidak_konek"],        "antena_rusak",  0.85, "R6"),
            (["sinyal_hilang"],                            "coba_restart",  0.50, "R7"),
            (["storage_penuh", "restart_sendiri"],         "storage_issue", 0.80, "R8"),
            (["touch_tidak_respon", "layar_tidak_nyala"],  "lcd_rusak",     0.90, "R9"),
            (["hp_panas", "baterai_cepat_habis",
              "restart_sendiri"],                          "cpu_overheat",  0.75, "R10"),
        ]
        for rule in rules:
            self.add_rule(*rule)

    def diagnosa_interaktif(self, gejala_ada: list):
        for g in gejala_ada:
            self.assert_fact(g, cf=1.0)
        self.forward_cf(verbose=False)
        return self.diagnosis(threshold=0.4,
                              exclude_keys=set(gejala_ada))


# Test
sp2 = PakarSmartphone()
gejala = ["layar_tidak_nyala", "baterai_cepat_habis",
          "tidak_bisa_charge", "hp_panas"]
print(f"\n  Gejala masuk: {gejala}")
hasil = sp2.diagnosa_interaktif(gejala)
print(f"\n  ── Kemungkinan Kerusakan ──")
for d, cf in hasil:
    bar  = "█" * int(cf * 20)
    print(f"  {d:<16} CF={cf:.3f} {bar}")


# ─────────────────────────────────────────────
# BAGIAN 4: RULE-BASED vs MACHINE LEARNING
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 4: RULE-BASED vs DECISION TREE ML")
print("=" * 55)


class RuleBasedIris:
    """Klasifikasi Iris dengan aturan IF-THEN manual dari pakar."""

    RULES = [
        # (petal_len_max, petal_wid_max, kelas)
        (2.5, 99.0, "setosa"),
        (5.0, 1.8,  "versicolor"),
        (99.0, 99.0, "virginica"),
    ]

    def predict_one(self, petal_len, petal_wid):
        for pl_max, pw_max, kelas in self.RULES:
            if petal_len <= pl_max and petal_wid <= pw_max:
                return kelas
        return "virginica"

    def predict(self, X):
        return [self.predict_one(x[2], x[3]) for x in X]

    def score(self, X, y_true, target_names):
        preds = self.predict(X)
        benar = sum(p == target_names[t]
                    for p, t in zip(preds, y_true))
        return benar / len(y_true)


iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Rule-Based
rb = RuleBasedIris()
acc_rb   = rb.score(X_te, y_te, iris.target_names)
preds_rb = rb.predict(X_te)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_tr, y_tr)
acc_ml   = dt.score(X_te, y_te)
preds_ml = iris.target_names[dt.predict(X_te)]

print(f"\n  {'─'*48}")
print(f"  {'Metode':<22} {'Akurasi':<10} {'Keterangan'}")
print(f"  {'─'*48}")
print(f"  {'Rule-Based':<22} {acc_rb:.2%}     3 aturan IF-THEN manual")
print(f"  {'Decision Tree ML':<22} {acc_ml:.2%}     Belajar dari 105 sampel")
print(f"  {'─'*48}")

print(f"\n  ── Laporan Rule-Based ──")
print(classification_report(y_te, [iris.target_names.tolist().index(p)
                                   if hasattr(iris.target_names, 'tolist')
                                   else list(iris.target_names).index(p)
                                   for p in preds_rb],
                             target_names=iris.target_names))

print(f"  ── Decision Tree Rules (depth=4) ──")
tree_rules = export_text(dt, feature_names=iris.feature_names)
for line in tree_rules.split('\n')[:20]:
    print(f"  {line}")


# ─────────────────────────────────────────────
# BAGIAN 5: EKSPOR ATURAN KE JSON
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 5: EKSPOR ATURAN KE JSON")
print("=" * 55)

aturan_json = {
    "versi": "1.0",
    "domain": "Diagnosis Smartphone",
    "aturan": [
        {
            "id": f"R{i+1}",
            "kondisi": cond,
            "hasil": hasil,
            "cf": cf,
            "nama": nama,
        }
        for i, (cond, hasil, cf, nama)
        in enumerate(sp2.aturan)
    ]
}

with open("week06_aturan.json", "w", encoding="utf-8") as f:
    json.dump(aturan_json, f, indent=2, ensure_ascii=False)
print("  ✅ Aturan diekspor ke week06_aturan.json")

# Baca kembali
with open("week06_aturan.json") as f:
    loaded = json.load(f)
print(f"  ✅ Berhasil dimuat: {len(loaded['aturan'])} aturan")


# ─────────────────────────────────────────────
# BAGIAN 6: VISUALISASI PERBANDINGAN
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print(" BAGIAN 6: VISUALISASI")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#141414")

# — Plot 1: Akurasi RB vs ML —
ax = axes[0]
ax.set_facecolor("#1c1c1c")
metode  = ["Rule-Based\n(3 aturan)", "Decision Tree\n(ML)"]
akurasi = [acc_rb * 100, acc_ml * 100]
colors  = ["#FFD93D", "#5BD68A"]
bars = ax.bar(metode, akurasi, color=colors,
              edgecolor="#141414", linewidth=2, width=0.5)
for bar, val in zip(bars, akurasi):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{val:.1f}%", ha='center', color='white',
            fontweight='bold', fontsize=11)
ax.set_ylim(0, 110)
ax.set_title("Akurasi: Rule-Based vs ML", color='white',
             fontweight='bold', pad=10)
ax.set_ylabel("Akurasi (%)", color='#aaa')
ax.tick_params(colors='white')
for s in ax.spines.values():
    s.set_edgecolor('#444')
ax.grid(True, color='#333', linewidth=0.5, axis='y')

# — Plot 2: CF Diagnosis Smartphone —
ax2 = axes[1]
ax2.set_facecolor("#1c1c1c")
if hasil:
    names_sp = [h[0] for h in hasil]
    cf_vals  = [h[1] for h in hasil]
    bar_cols = ["#5BD68A" if c > 0.7 else
                "#FFD93D" if c > 0.4 else "#FF6FA5"
                for c in cf_vals]
    bars2 = ax2.barh(names_sp, cf_vals,
                     color=bar_cols, edgecolor="#141414",
                     linewidth=1.5)
    for bar, val in zip(bars2, cf_vals):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va='center', color='white', fontsize=9)
    ax2.set_xlim(0, 1.1)
    ax2.axvline(0.7, color='#5BD68A', linestyle='--',
                linewidth=1, label='Threshold tinggi (0.7)')
    ax2.axvline(0.4, color='#FFD93D', linestyle='--',
                linewidth=1, label='Threshold sedang (0.4)')
ax2.set_title("CF Diagnosis Smartphone", color='white',
              fontweight='bold', pad=10)
ax2.set_xlabel("Certainty Factor", color='#aaa')
ax2.tick_params(colors='white')
for s in ax2.spines.values():
    s.set_edgecolor('#444')
ax2.legend(labelcolor='white', facecolor='#1c1c1c',
           edgecolor='#444', fontsize=7)

plt.suptitle("Sesi 6 — Rule-Based System & Certainty Factor",
             color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("week06_hasil.png", dpi=150,
            bbox_inches='tight', facecolor='#141414')
plt.show()
print("  📊 Grafik disimpan ke week06_hasil.png")

print("\n" + "=" * 55)
print("  ✅ SELESAI — Semua praktik Sesi 6 berhasil!")
print("=" * 55)
