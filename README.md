<div align="center">

# 🧠 VectorDB

**A Vector Database built from scratch in Python — no external ML libraries**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-brightgreen?style=for-the-badge)]()

Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side with a **RAG pipeline** powered by a local LLM via Ollama — all in pure Python, standard library only.

> Built to show how production vector databases like Pinecone, Weaviate, and Chroma actually work under the hood.

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [API Reference](#-rest-api-reference) • [Architecture](#-architecture) • [Algorithms](#-algorithm-deep-dive)

---

![VectorDB Screenshot](https://i.imgur.com/placeholder.png)
*← Replace this with a real screenshot of your UI after running the project*

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **3 Search Algorithms** | HNSW, KD-Tree, Brute Force — run all three and compare speed live |
| 📐 **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| 🎯 **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| 🗺️ **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| 📄 **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| 🤖 **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| 🌐 **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |
| 📦 **Zero Dependencies** | Pure Python standard library — no pip installs required |

---

## 🚀 Demo

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)     ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python)           ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search               ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)             ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

Type `binary tree` → finds CS vectors. Type `sushi` → finds Food vectors.  
The scatter plot shows semantic clusters forming in real time.

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **Python** | 3.9+ | 3.11+ |
| **RAM** | 4 GB | 8 GB |
| **Disk** | 3 GB free | 5 GB free |
| **OS** | Windows / macOS / Linux | Any |

> **Low RAM?** Use `llama3.2:1b` (~800MB) instead of `llama3.2` (~2GB). See [configuration](#configuration).

---

## 📦 Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/VectorDB.git
cd VectorDB
```

### Step 2 — Install Python (if not already installed)

Download from [python.org](https://www.python.org/downloads/) — version 3.9 or higher.

Verify:
```bash
python --version
```

### Step 3 — Install Ollama

1. Download from [ollama.com](https://ollama.com) for your OS
2. Pull the two required models:

```bash
ollama pull nomic-embed-text   # ~274 MB — embedding model
ollama pull llama3.2           # ~2 GB   — language model
```

3. Verify both models are ready:
```bash
ollama list
```

### Step 4 — Run the server

```bash
python main.py
```

You should see:
```
=== VectorDB Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
Server running on http://localhost:8080
```

**Open your browser:** [http://localhost:8080](http://localhost:8080)

---

## 🎮 Usage

### Tab 1 — Search (Demo Vectors)

- Type any concept: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **⚡ SEARCH** — results appear, matching point glows on scatter plot
- Click **▶ COMPARE ALL ALGOS** to benchmark all 3 algorithms side-by-side

The scatter plot projects all 20 vectors to 2D using PCA. The 4 semantic categories (CS, Math, Food, Sports) form distinct clusters — this is what semantic similarity looks like visually.

### Tab 2 — Documents (Real Embeddings)

1. Enter a title (e.g., `Operating Systems Notes`)
2. Paste any text — lecture notes, articles, textbook paragraphs
3. Click **⚡ EMBED & INSERT**

Long documents are automatically split into overlapping 250-word chunks. Each chunk gets its own 768D embedding stored in a separate HNSW index.

### Tab 3 — Ask AI (RAG Pipeline)

1. Insert documents in Tab 2 first
2. Type any question about your documents
3. Click **🤖 ASK AI**

Behind the scenes:
```
1. Your question  →  embedded with nomic-embed-text (768D vector)
2. HNSW search    →  finds 3 most semantically similar chunks
3. Retrieved chunks  →  sent as context to llama3.2
4. llama3.2       →  generates an answer grounded in your documents
```

---

## ⚙️ Configuration

### Use a smaller, faster LLM

If `llama3.2` is too slow on your machine:

```bash
ollama pull llama3.2:1b    # only ~800MB, much faster
```

Then edit `main.py`:
```python
self.gen_model = "llama3.2:1b"   # line ~270 in OllamaClient.__init__
```

Restart the server.

### Change the port

At the bottom of `main.py`:
```python
server = HTTPServer(("0.0.0.0", 8080), Handler)   # change 8080 to any free port
```

---

## 🌐 REST API Reference

The server exposes a full REST API at `http://localhost:8080`.

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all stored documents |
| `DELETE` | `/doc/delete/:id` | — | Delete document chunk |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status and model info |

### Examples

**Search:**
```bash
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

**Insert a vector:**
```bash
curl -X POST http://localhost:8080/insert \
  -H "Content-Type: application/json" \
  -d '{"metadata":"My vector","category":"cs","embedding":[0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]}'
```

**Ask a question (RAG):**
```bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## 🏗️ Architecture

```
VectorDB/
├── main.py       ← Python backend (all algorithms + HTTP server + RAG)
├── index.html    ← Frontend (PCA scatter plot, chat UI, benchmark)
└── README.md     ← This file
```

### Backend Components (`main.py`)

```
BruteForce      O(N·d)      Exact, linear scan baseline
KDTree          O(log N)    Exact, binary space partitioning
HNSW            O(log N)    Approximate, multilayer small-world graph

VectorDB        Unified interface over all 3  (16D demo vectors)
DocumentDB      HNSW-only index for Ollama embeddings (768D)
OllamaClient    HTTP client → /api/embeddings + /api/generate
```

---

## 📐 Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)

The same algorithm used by **Pinecone, Weaviate, Chroma, and Milvus**.

Nodes are inserted into a multilayer graph. Each node is randomly assigned a maximum layer. Layer 0 has all nodes with many short-range connections; higher layers have exponentially fewer nodes with longer-range connections.

- **Insert:** Greedy descent from the top layer. At each layer, run a beam search (`ef_construction=200`) and connect to the M nearest neighbors bidirectionally.
- **Search:** Same greedy descent. At layer 0, expand to `ef` nearest candidates using a priority queue.
- **Why it's fast:** Upper layers act like a highway — you quickly reach the right neighborhood, then zoom in at layer 0. Achieves **O(log N)** complexity.

### KD-Tree (K-Dimensional Tree)

Binary space partitioning. Each node splits space along one dimension (cycling through all dimensions). Search prunes entire subtrees when the closest possible point in that subtree can't beat the current best.

**Weakness:** Degrades with high dimensions (curse of dimensionality). Works well for ≤20D, becomes close to brute force at 768D.

### Why HNSW Wins at High Dimensions

KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions, almost all space is near the hypersphere boundary — no subtrees get pruned. HNSW's graph-based navigation doesn't have this problem, maintaining efficiency even at 768D.

---

## 🗂️ Storage Usage

| Component | Size |
|---|---|
| `nomic-embed-text` model | ~274 MB |
| `llama3.2` model | ~2.0 GB |
| Project files | ~100 KB |
| **Total** | **~2.3 GB** |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` in header | Run `ollama serve` in a terminal |
| Embedding takes forever | Ollama is downloading on first use — wait ~2 min |
| `python: command not found` | Ensure Python 3.9+ is installed and in PATH |
| Port 8080 already in use | Change port in `main.py` (see [Configuration](#configuration)) |
| LLM answer is slow | Switch to `llama3.2:1b` (see [Configuration](#configuration)) |
| `ModuleNotFoundError` | This project uses zero external modules — check your Python version |

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- [ ] Persist vectors to disk (JSON / SQLite)
- [ ] Add more distance metrics (dot product, Hamming)
- [ ] Support for streaming LLM responses
- [ ] Docker support
- [ ] Unit tests

---

## 📄 License

MIT — use this however you want.

---

<div align="center">

Made with Python 🐍 and zero dependencies

⭐ Star this repo if you found it useful!

</div>