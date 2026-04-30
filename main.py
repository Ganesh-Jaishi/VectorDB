"""
VectorDB — Python port of the C++ VectorDB project
Implements HNSW, KD-Tree, and Brute Force search algorithms
with a RAG pipeline powered by local Ollama LLM.
"""

from __future__ import annotations
import math
import random
import threading
import time
import json
import re
import os
import urllib.request
import urllib.error
from typing import Callable, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# =====================================================================
#  CONSTANTS
# =====================================================================

DIMS = 16   # demo vectors dimension

# =====================================================================
#  DATA TYPES
# =====================================================================

class VectorItem:
    def __init__(self, id: int, metadata: str, category: str, emb: list[float]):
        self.id = id
        self.metadata = metadata
        self.category = category
        self.emb = emb

DistFn = Callable[[list[float], list[float]], float]

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a)
    nb  = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))

def manhattan(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":    return cosine
    if metric == "manhattan": return manhattan
    return euclidean

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: list[VectorItem] = []

    def insert(self, v: VectorItem):
        self.items.append(v)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        scored = [(dist(q, v.emb), v.id) for v in self.items]
        scored.sort()
        return scored[:k]

    def remove(self, id: int):
        self.items = [v for v in self.items if v.id != id]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    def __init__(self, item: VectorItem):
        self.item = item
        self.left:  Optional[KDNode] = None
        self.right: Optional[KDNode] = None

class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[KDNode] = None

    def _insert(self, node: Optional[KDNode], v: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(v)
        ax = depth % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left  = self._insert(node.left,  v, depth + 1)
        else:
            node.right = self._insert(node.right, v, depth + 1)
        return node

    def insert(self, v: VectorItem):
        self.root = self._insert(self.root, v, 0)

    def _knn(self, node: Optional[KDNode], q: list[float], k: int, depth: int,
             dist: DistFn, heap: list[tuple[float, int]]):
        if node is None:
            return
        import heapq
        dn = dist(q, node.item.emb)
        # Max-heap stored as negatives
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)
        ax   = depth % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._knn(closer,  q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        import heapq
        heap: list[tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist, heap)
        result = [(-d, id) for d, id in heap]
        result.sort()
        return result

    def rebuild(self, items: list[VectorItem]):
        self.root = None
        for v in items:
            self.insert(v)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSWNode:
    def __init__(self, item: VectorItem, max_lyr: int):
        self.item    = item
        self.max_lyr = max_lyr
        self.nbrs: list[list[int]] = [[] for _ in range(max_lyr + 1)]

class HNSW:
    def __init__(self, M: int = 16, ef_build: int = 200):
        self.M        = M
        self.M0       = 2 * M
        self.ef_build = ef_build
        self.mL       = 1.0 / math.log(M)
        self.G: dict[int, HNSWNode] = {}
        self.top_layer = -1
        self.entry_pt  = -1
        self._rng      = random.Random(42)

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, q: list[float], ep: int, ef: int, lyr: int,
                       dist: DistFn) -> list[tuple[float, int]]:
        import heapq
        vis   = {ep}
        d0    = dist(q, self.G[ep].item.emb)
        cands = [(d0, ep)]          # min-heap
        found = [(-d0, ep)]         # max-heap (negated)

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            nd = self.G.get(cid)
            if nd is None or lyr >= len(nd.nbrs):
                continue
            for nid in nd.nbrs[lyr]:
                if nid in vis or nid not in self.G:
                    continue
                vis.add(nid)
                nd2 = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd2 < -found[0][0]:
                    heapq.heappush(cands, (nd2, nid))
                    heapq.heappush(found, (-nd2, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-d, id) for d, id in found]
        result.sort()
        return result

    def _select_nbrs(self, cands: list[tuple[float, int]], max_m: int) -> list[int]:
        return [id for _, id in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn):
        id  = item.id
        lvl = self._rand_level()
        self.G[id] = HNSWNode(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt  = id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            nd = self.G.get(ep)
            if nd and lc < len(nd.nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W    = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            maxM = self.M0 if lc == 0 else self.M
            sel  = self._select_nbrs(W, maxM)
            self.G[id].nbrs[lc] = sel

            for nid in sel:
                nd = self.G.get(nid)
                if nd is None:
                    continue
                while lc >= len(nd.nbrs):
                    nd.nbrs.append([])
                conn = nd.nbrs[lc]
                conn.append(id)
                if len(conn) > maxM:
                    ds = []
                    for c in conn:
                        if c in self.G:
                            ds.append((dist(self.G[nid].item.emb, self.G[c].item.emb), c))
                    ds.sort()
                    nd.nbrs[lc] = [c for _, c in ds[:maxM]]
            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt  = id

    def knn(self, q: list[float], k: int, ef: int, dist: DistFn) -> list[tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            nd = self.G.get(ep)
            if nd and lc < len(nd.nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id: int):
        if id not in self.G:
            return
        for nd in self.G.values():
            for layer in nd.nbrs:
                if id in layer:
                    layer.remove(id)
        if self.entry_pt == id:
            self.entry_pt = next((nid for nid in self.G if nid != id), -1)
        del self.G[id]

    def get_info(self) -> dict:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer  = [0] * max_l
        edges_per_layer  = [0] * max_l
        nodes = []
        edges = []
        for id, nd in self.G.items():
            nodes.append({"id": id, "metadata": nd.item.metadata,
                          "category": nd.item.category, "maxLyr": nd.max_lyr})
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id, "dst": nid, "lyr": lc})
        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }

    def __len__(self):
        return len(self.G)

# =====================================================================
#  VECTOR DATABASE  (demo 16D index)
# =====================================================================

class Hit:
    def __init__(self, id, meta, cat, emb, dist_val):
        self.id   = id
        self.meta = meta
        self.cat  = cat
        self.emb  = emb
        self.dist = dist_val

class SearchOut:
    def __init__(self, hits, us, algo, metric):
        self.hits   = hits
        self.us     = us
        self.algo   = algo
        self.metric = metric

class BenchOut:
    def __init__(self, bf_us, kd_us, hnsw_us, n):
        self.bf_us   = bf_us
        self.kd_us   = kd_us
        self.hnsw_us = hnsw_us
        self.n       = n

class VectorDB:
    def __init__(self, dims: int):
        self.dims   = dims
        self.store: dict[int, VectorItem] = {}
        self.bf     = BruteForce()
        self.kdt    = KDTree(dims)
        self.hnsw   = HNSW(16, 200)
        self.lock   = threading.Lock()
        self._next  = 1

    def insert(self, meta: str, cat: str, emb: list[float], dist: DistFn) -> int:
        with self.lock:
            v = VectorItem(self._next, meta, cat, emb)
            self._next += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, id: int) -> bool:
        with self.lock:
            if id not in self.store:
                return False
            del self.store[id]
            self.bf.remove(id)
            self.hnsw.remove(id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: list[float], k: int, metric: str, algo: str) -> SearchOut:
        with self.lock:
            dfn = get_dist_fn(metric)
            t0  = time.perf_counter_ns()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            us = (time.perf_counter_ns() - t0) // 1000
            hits = [
                Hit(id, self.store[id].metadata, self.store[id].category,
                    self.store[id].emb, d)
                for d, id in raw if id in self.store
            ]
            return SearchOut(hits, us, algo, metric)

    def benchmark(self, q: list[float], k: int, metric: str) -> BenchOut:
        with self.lock:
            dfn = get_dist_fn(metric)
            def time_fn(fn):
                t = time.perf_counter_ns()
                fn()
                return (time.perf_counter_ns() - t) // 1000
            return BenchOut(
                time_fn(lambda: self.bf.knn(q, k, dfn)),
                time_fn(lambda: self.kdt.knn(q, k, dfn)),
                time_fn(lambda: self.hnsw.knn(q, k, 50, dfn)),
                len(self.store),
            )

    def all(self) -> list[VectorItem]:
        with self.lock:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self.lock:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.lock:
            return len(self.store)

# =====================================================================
#  DOCUMENT DATABASE  — HNSW over real Ollama embeddings
# =====================================================================

class DocItem:
    def __init__(self, id, title, text, emb):
        self.id    = id
        self.title = title
        self.text  = text
        self.emb   = emb

class DocumentDB:
    def __init__(self):
        self.store: dict[int, DocItem] = {}
        self.hnsw  = HNSW(16, 200)
        self.bf    = BruteForce()
        self.lock  = threading.Lock()
        self._next = 1
        self.dims  = 0

    def insert(self, title: str, text: str, emb: list[float]) -> int:
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self._next, title, text, emb)
            self._next += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: list[float], k: int, max_dist: float = 0.7) -> list[tuple[float, DocItem]]:
        with self.lock:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)
            return [
                (d, self.store[id])
                for d, id in raw
                if id in self.store and d <= max_dist
            ]

    def remove(self, id: int) -> bool:
        with self.lock:
            if id not in self.store:
                return False
            del self.store[id]
            self.hnsw.remove(id)
            self.bf.remove(id)
            return True

    def all(self) -> list[DocItem]:
        with self.lock:
            return list(self.store.values())

    def size(self) -> int:
        with self.lock:
            return len(self.store)

    def get_dims(self) -> int:
        return self.dims

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end   = min(i + chunk_words, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
        i += step
    return chunks

# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.host        = host
        self.port        = port
        self.embed_model = "nomic-embed-text"
        self.gen_model   = "llama3.2"
        self._base       = f"http://{host}:{port}"

    def _post(self, path: str, data: dict, timeout: int = 30) -> Optional[dict]:
        try:
            body = json.dumps(data).encode()
            req  = urllib.request.Request(
                self._base + path,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None

    def _get(self, path: str, timeout: int = 2) -> Optional[dict]:
        try:
            req = urllib.request.Request(self._base + path)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None

    def is_available(self) -> bool:
        return self._get("/api/tags") is not None

    def embed(self, text: str) -> list[float]:
        resp = self._post("/api/embeddings",
                          {"model": self.embed_model, "prompt": text},
                          timeout=30)
        if resp and "embedding" in resp:
            return resp["embedding"]
        return []

    def generate(self, prompt: str) -> str:
        resp = self._post("/api/generate",
                          {"model": self.gen_model, "prompt": prompt, "stream": False},
                          timeout=180)
        if resp and "response" in resp:
            return resp["response"]
        return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  DEMO DATA  (16D categorical vectors)
# =====================================================================

def load_demo(db: VectorDB):
    dist = get_dist_fn("cosine")
    # Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    demo_items = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
    ]
    for meta, cat, emb in demo_items:
        db.insert(meta, cat, emb, dist)

# =====================================================================
#  JSON HELPERS
# =====================================================================

def js(s: str) -> str:
    return json.dumps(s)

def j_vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.4f}" for x in v) + "]"

def parse_vec(s: str) -> list[float]:
    try:
        return [float(x) for x in s.split(",") if x.strip()]
    except Exception:
        return []

# =====================================================================
#  HTTP REQUEST HANDLER
# =====================================================================

class Handler(BaseHTTPRequestHandler):
    db: VectorDB
    doc_db: DocumentDB
    ollama: OllamaClient

    def log_message(self, format, *args):
        pass  # suppress default access log

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, code: int, body: str, ct: str = "application/json"):
        data = body.encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> str:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length).decode() if length else ""

    # ── ROUTING ──────────────────────────────────────────────────────

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        qs     = parse_qs(parsed.query)

        def qp(key, default=""):
            vals = qs.get(key, [])
            return vals[0] if vals else default

        if path == "/":
            self._serve_html()
        elif path == "/search":
            self._search(qp("v"), qp("k", "5"), qp("metric", "cosine"), qp("algo", "hnsw"))
        elif path == "/items":
            self._items()
        elif path == "/benchmark":
            self._benchmark(qp("v"), qp("k", "5"), qp("metric", "cosine"))
        elif path == "/hnsw-info":
            self._hnsw_info()
        elif path == "/stats":
            self._stats()
        elif path == "/doc/list":
            self._doc_list()
        elif path == "/status":
            self._status()
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_body()
        if path == "/insert":
            self._insert(body)
        elif path == "/doc/insert":
            self._doc_insert(body)
        elif path == "/doc/search":
            self._doc_search(body)
        elif path == "/doc/ask":
            self._doc_ask(body)
        else:
            self._send(404, '{"error":"not found"}')

    def do_DELETE(self):
        path = urlparse(self.path).path
        m = re.match(r"^/delete/(\d+)$", path)
        if m:
            self._delete(int(m.group(1)))
            return
        m = re.match(r"^/doc/delete/(\d+)$", path)
        if m:
            self._doc_delete(int(m.group(1)))
            return
        self._send(404, '{"error":"not found"}')

    # ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

    def _serve_html(self):
        html_path = os.path.join(os.path.dirname(__file__), "index.html")
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                self._send(200, f.read(), "text/html")
        except FileNotFoundError:
            self._send(404, "index.html not found", "text/plain")

    def _search(self, v_str, k_str, metric, algo):
        q = parse_vec(v_str)
        if len(q) != DIMS:
            self._send(400, f'{{"error":"need {DIMS}D vector"}}')
            return
        try:   k = int(k_str)
        except: k = 5
        out = Handler.db.search(q, k, metric, algo)
        results = []
        for h in out.hits:
            results.append(
                f'{{"id":{h.id},"metadata":{js(h.meta)},"category":{js(h.cat)},'
                f'"distance":{h.dist:.6f},"embedding":{j_vec(h.emb)}}}'
            )
        body = (f'{{"results":[{",".join(results)}],'
                f'"latencyUs":{out.us},"algo":{js(out.algo)},"metric":{js(out.metric)}}}')
        self._send(200, body)

    def _insert(self, body: str):
        try:
            data = json.loads(body)
            meta = data.get("metadata", "")
            cat  = data.get("category", "")
            emb  = data.get("embedding", [])
            if not meta or not emb or len(emb) != DIMS:
                raise ValueError
        except Exception:
            self._send(400, '{"error":"invalid body"}')
            return
        id = Handler.db.insert(meta, cat, emb, get_dist_fn("cosine"))
        self._send(200, f'{{"id":{id}}}')

    def _delete(self, id: int):
        ok = Handler.db.remove(id)
        self._send(200, f'{{"ok":{"true" if ok else "false"}}}')

    def _items(self):
        items = Handler.db.all()
        parts = []
        for v in items:
            parts.append(
                f'{{"id":{v.id},"metadata":{js(v.metadata)},'
                f'"category":{js(v.category)},"embedding":{j_vec(v.emb)}}}'
            )
        self._send(200, f'[{",".join(parts)}]')

    def _benchmark(self, v_str, k_str, metric):
        q = parse_vec(v_str)
        if len(q) != DIMS:
            self._send(400, f'{{"error":"need {DIMS}D vector"}}')
            return
        try:   k = int(k_str)
        except: k = 5
        b = Handler.db.benchmark(q, k, metric)
        self._send(200,
            f'{{"bruteforceUs":{b.bf_us},"kdtreeUs":{b.kd_us},'
            f'"hnswUs":{b.hnsw_us},"itemCount":{b.n}}}')

    def _hnsw_info(self):
        gi = Handler.db.hnsw_info()
        self._send(200, json.dumps(gi))

    def _stats(self):
        self._send(200,
            f'{{"count":{Handler.db.size()},"dims":{DIMS},'
            f'"algorithms":["bruteforce","kdtree","hnsw"],'
            f'"metrics":["euclidean","cosine","manhattan"]}}')

    # ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

    def _doc_insert(self, body: str):
        try:
            data  = json.loads(body)
            title = data.get("title", "")
            text  = data.get("text", "")
            if not title or not text:
                raise ValueError
        except Exception:
            self._send(400, '{"error":"need title and text"}')
            return

        chunks = chunk_text(text, 250, 30)
        ids    = []
        for i, chunk in enumerate(chunks):
            emb = Handler.ollama.embed(chunk)
            if not emb:
                self._send(500,
                    '{"error":"Ollama unavailable. Install from https://ollama.com '
                    'then run: ollama pull nomic-embed-text && ollama pull llama3.2"}')
                return
            chunk_title = (f"{title} [{i+1}/{len(chunks)}]"
                           if len(chunks) > 1 else title)
            ids.append(Handler.doc_db.insert(chunk_title, chunk, emb))

        self._send(200,
            f'{{"ids":[{",".join(map(str, ids))}],'
            f'"chunks":{len(chunks)},"dims":{Handler.doc_db.get_dims()}}}')

    def _doc_delete(self, id: int):
        ok = Handler.doc_db.remove(id)
        self._send(200, f'{{"ok":{"true" if ok else "false"}}}')

    def _doc_list(self):
        docs  = Handler.doc_db.all()
        parts = []
        for d in docs:
            preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
            words   = len(d.text.split())
            parts.append(
                f'{{"id":{d.id},"title":{js(d.title)},'
                f'"preview":{js(preview)},"words":{words}}}'
            )
        self._send(200, f'[{",".join(parts)}]')

    def _doc_search(self, body: str):
        try:
            data     = json.loads(body)
            question = data.get("question", "")
            k        = int(data.get("k", 3))
            if not question:
                raise ValueError
        except Exception:
            self._send(400, '{"error":"need question"}')
            return

        q_emb = Handler.ollama.embed(question)
        if not q_emb:
            self._send(500, '{"error":"Ollama unavailable"}')
            return

        hits  = Handler.doc_db.search(q_emb, k)
        parts = []
        for d, item in hits:
            parts.append(
                f'{{"id":{item.id},"title":{js(item.title)},"distance":{d:.4f}}}'
            )
        self._send(200, f'{{"contexts":[{",".join(parts)}]}}')

    def _doc_ask(self, body: str):
        try:
            data     = json.loads(body)
            question = data.get("question", "")
            k        = int(data.get("k", 3))
            if not question:
                raise ValueError
        except Exception:
            self._send(400, '{"error":"need question"}')
            return

        # Step 1: embed the question
        q_emb = Handler.ollama.embed(question)
        if not q_emb:
            self._send(500, '{"error":"Ollama unavailable"}')
            return

        # Step 2: retrieve top-k relevant chunks
        hits = Handler.doc_db.search(q_emb, k)

        # Step 3: build prompt
        ctx = ""
        for i, (_, item) in enumerate(hits):
            ctx += f"[{i+1}] {item.title}:\n{item.text}\n\n"

        prompt = (
            "You are a helpful assistant. Answer the user's question directly. "
            "Use the provided context if it contains relevant information. "
            "If it doesn't, just use your own general knowledge. "
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
            "'the context doesn't mention'. Just answer the question naturally.\n\n"
            f"Context:\n{ctx}"
            f"Question: {question}\n\nAnswer:"
        )

        # Step 4: generate answer
        answer = Handler.ollama.generate(prompt)

        # Step 5: return everything
        contexts = []
        for d, item in hits:
            contexts.append(
                f'{{"id":{item.id},"title":{js(item.title)},'
                f'"text":{js(item.text)},"distance":{d:.4f}}}'
            )
        self._send(200,
            f'{{"answer":{js(answer)},"model":{js(Handler.ollama.gen_model)},'
            f'"contexts":[{",".join(contexts)}],"docCount":{Handler.doc_db.size()}}}')

    def _status(self):
        up = Handler.ollama.is_available()
        self._send(200,
            f'{{"ollamaAvailable":{"true" if up else "false"},'
            f'"embedModel":{js(Handler.ollama.embed_model)},'
            f'"genModel":{js(Handler.ollama.gen_model)},'
            f'"docCount":{Handler.doc_db.size()},'
            f'"docDims":{Handler.doc_db.get_dims()},'
            f'"demoDims":{DIMS},'
            f'"demoCount":{Handler.db.size()}}}')

# =====================================================================
#  MAIN
# =====================================================================

def main():
    db     = VectorDB(DIMS)
    doc_db = DocumentDB()
    ollama = OllamaClient()

    load_demo(db)

    # Inject dependencies into handler class
    Handler.db     = db
    Handler.doc_db = doc_db
    Handler.ollama = ollama

    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    server = HTTPServer(("0.0.0.0", 8080), Handler)
    print("Server running on http://localhost:8080")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")

if __name__ == "__main__":
    main()
