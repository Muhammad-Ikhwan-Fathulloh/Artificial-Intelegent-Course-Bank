# Intelegensi Buatan (IF-030)

Repositori ini berisi rencana, materi, dan notebook praktik mata kuliah Intelegensi Buatan. Topik mengikuti RPS. Praktik dan tambahan modern diatur agar alurnya:

**Python → Machine Learning → Deep Learning → Transformers & LLM → FastAPI → Streamlit**

---

## Peran FastAPI dan Streamlit

| Tool | Peran | Kapan dipakai |
|---|---|---|
| **Streamlit** | Implementasi cepat: dari notebook ke UI interaktif dalam hitungan menit | Demo cepat di minggu 9-13 (opsional), dibahas penuh di minggu 15 |
| **FastAPI** | Membuat API: model dibungkus sebagai layanan (`/predict`) yang bisa dipanggil aplikasi mana pun | Dibahas penuh di minggu 14 |

Keduanya dibahas di pertemuan berbeda. Di minggu 15 keduanya digabung: Streamlit sebagai tampilan yang memanggil FastAPI.

---

## Alur Belajar

| Fase | Pertemuan | Hasil |
|---|---|---|
| Python | 1-4 | Bisa coding, memakai NumPy dan Pandas, sudah mencoba model ML pertama |
| Machine Learning | 5-7, 9-11 | Pipeline ML lengkap dengan scikit-learn |
| Deep Learning | 12 | Neural network dan CNN (Computer Vision) |
| Transformers & LLM | 13 | NLP dengan Hugging Face dan pemanggilan LLM |
| FastAPI | 14 | Model menjadi API |
| Streamlit & Evaluasi | 15 | Aplikasi cepat, pengujian, dan presentasi proyek |

---

## Rencana per Pertemuan

| Mg | Topik RPS | Praktik Python | Tambahan Modern |
|---|---|---|---|
| 1 | Konsep dasar AI, sejarah, tujuan dan ruang lingkup, contoh aplikasi | Setup Colab, dasar Python, NumPy | Peta AI → ML → DL → Generative AI |
| 2 | Intelligent agent, rational agent, environment, problem formulation, state space | OOP agent sederhana, Pandas dasar, **model ML pertama** (`fit`, `predict`, `accuracy`) | Agent sebagai persepsi → keputusan → aksi; model ML sebagai "otak" agent |
| 3 | Problem solving, state space, BFS, DFS | `deque`, rekursi, pathfinding pada maze | Pencarian pada game dan navigasi |
| 4 | UCS, heuristic search, Greedy Best First, A* | `heapq`, A* pada grid | Rute peta dan logistik, perbandingan efisiensi |
| 5 | Knowledge representation, knowledge base, propositional logic | Fakta dan aturan dengan `dict`/`set`, tabel kebenaran | Knowledge graph, pengantar RAG |
| 6 | Predicate logic, inference, forward/backward chaining, rule-based system | Mesin inferensi mini, bandingkan rule-based dengan model ML minggu 2 | Mengapa ML menggantikan aturan manual |
| 7 | Konsep ML, dataset, feature, label, training/testing, supervised/unsupervised | Membedah model minggu 2, K-Means sekilas | Pipeline ML, *data-centric AI* |
| 8 | **UTS** | | |
| 9 | Dataset, preprocessing, cleaning, normalization, encoding, train-test split | Pandas, Matplotlib/Seaborn, `Pipeline` scikit-learn | EDA, *data leakage*; mulai proyek data science. Streamlit teaser: tampilkan DataFrame dan grafik |
| 10 | Classification, regression, Decision Tree, KNN, Naive Bayes, Linear/Logistic Regression | Latih dan bandingkan beberapa model | Random Forest, XGBoost, *feature importance* |
| 11 | K-Means, clustering, confusion matrix, accuracy, precision, recall, F1, evaluasi | Clustering, `classification_report`, cross-validation | PCA, metrik data tidak seimbang; jembatan ke DL |
| 12 | Analisis kebutuhan, perumusan masalah, pemilihan metode, desain arsitektur solusi AI | Neural network (PyTorch/Keras), CNN dengan *transfer learning* | **Deep Learning dan Computer Vision** |
| 13 | Python untuk AI, NumPy, Pandas, Scikit-learn, training dan testing model | Hugging Face `pipeline` (sentiment, NER), tokenizer, embedding, memanggil API LLM, prompt engineering | **Transformers, NLP, LLM** (attention secara konsep) |
| 14 | Integrasi/deployment model AI, **API sederhana** | **FastAPI**: endpoint `/predict`, validasi input dengan Pydantic, dokumentasi otomatis `/docs`, uji dengan `curl`/Postman | Model proyek (ML, CNN, atau LLM) dibungkus sebagai API |
| 15 | Aplikasi berbasis web, pengujian sistem, evaluasi model, optimasi | **Streamlit**: UI input → prediksi → output, memanggil FastAPI; analisis error, perbaikan model | Evaluasi LLM (halusinasi, bias), *responsible AI*, presentasi proyek |
| 16 | **UAS** | | |

---

## Proyek Sepanjang Semester

| Minggu | Tahap |
|---|---|
| 2-7 | Notebook latihan (search, rule-based, ML pertama) |
| 9-11 | Proyek data science: EDA dan model ML |
| 12-13 | Upgrade ke modul CV atau NLP/LLM |
| 14 | Model dibungkus FastAPI |
| 15 | Antarmuka Streamlit, pengujian, presentasi |

---

## Struktur Repositori (usulan)

```
intelegensi-buatan/
├── README.md
├── requirements.txt
├── week01-intro-python/
├── week02-agent-first-ml/
├── week03-bfs-dfs/
├── week04-ucs-astar/
├── week05-knowledge-propositional/
├── week06-predicate-rule-based/
├── week07-ml-concepts/
├── week09-preprocessing-eda/
├── week10-classification-regression/
├── week11-clustering-evaluation/
├── week12-deep-learning-cv/
├── week13-transformers-nlp-llm/
├── week14-fastapi/
├── week15-streamlit-evaluation/
└── project/
    ├── notebooks/
    ├── api/            # FastAPI
    └── app/            # Streamlit
```

Setiap folder minggu berisi: `slides/` (atau tautan), `notebook.ipynb`, dan `README.md` singkat (tujuan, prasyarat, tugas).

---

## Tools dan Lingkungan

- **Google Colab**: minggu 1-13 (tanpa instalasi).
- **VS Code lokal atau GitHub Codespaces**: minggu 14-15, karena server FastAPI dan Streamlit tidak cocok dijalankan di Colab.
- **Library utama**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `torch` atau `tensorflow/keras`, `transformers`, `fastapi`, `uvicorn`, `pydantic`, `streamlit`, `requests`.
- **Akses LLM**: gunakan model gratis dari Hugging Face atau API key bersama kelas.

```bash
# Setup lokal
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Minggu 14: jalankan API
uvicorn api.main:app --reload

# Minggu 15: jalankan UI
streamlit run app/app.py
```

---

## Penilaian (sesuai RPS)

| Komponen | Bobot |
|---|---|
| Kuis / Partisipasi | 7% |
| Tugas / Proyek / Praktikum | 53% |
| UTS | 10% |
| UAS | 30% |

Pretest dan posttest (15%) dipakai sebagai bagian kebijakan penilaian. Kuis minggu 2 dapat berfungsi sebagai pretest.

---

## Catatan Perancangan

- **Code first, theory later**: mahasiswa menjalankan model dulu (minggu 2), lalu konsepnya dibedah (minggu 7), algoritmanya dijelaskan (minggu 10).
- **Notebook fill-in-the-blank** untuk minggu 1-4 agar mahasiswa pemula tidak tertinggal.
- **Minggu 12 dan 13 paling padat.** Jika waktu kurang: pakai CNN pretrained saja di minggu 12, dan cukup `pipeline` Hugging Face plus satu panggilan API LLM di minggu 13.
- **Minggu 15 padat** (Streamlit, pengujian, presentasi). Streamlit teaser di minggu 9 dan seterusnya membantu meringankannya.

---

## Perbaikan RPS yang Disarankan

- Beri nomor Sub-CPMK `CPMK052.1` sampai `.5` (saat ini semuanya tertulis `CPMK052`).
- Rapikan nomor yang menempel di tabel minggu 5, 9, dan 12.
- Isi Deskripsi Singkat MK (saat ini kosong).
- Tambahkan bahan kajian: deep learning, computer vision, NLP, Transformers, LLM, FastAPI, Streamlit.
- Hapus *reinforcement learning* di subtopik minggu 7, atau tambahkan ke daftar topik.
- Tambahkan pustaka: *Dive into Deep Learning* (d2l.ai), dokumentasi Hugging Face, FastAPI, dan Streamlit, serta *Speech and Language Processing* (Jurafsky & Martin).

---

## Referensi

**Utama**
1. Russell, S., & Norvig, P. (2024). *Artificial Intelligence: A Modern Approach*, Global Edition (4th ed.). Pearson.
2. Luger, G. F. (2024). *Artificial Intelligence: Principles and Practice*.
3. Joshi, A. V. (2023). *Machine Learning and Artificial Intelligence*.

**Pendukung**
1. Bartz, E., & Bartz-Beielstein, T. (Eds.). (2024). *Online Machine Learning: A Practical Guide with Examples in Python*.
2. Sharma, D. K., Hota, H. S., & Rababaah, A. R. (Eds.). (2024). *Machine Learning for Real World Applications*.
3. Miao, F., & Holmes, W. (2023). *Guidance for Generative AI in Education and Research*.

**Dokumentasi online**
- scikit-learn, PyTorch, Hugging Face, FastAPI, Streamlit
- *Dive into Deep Learning*: https://d2l.ai
