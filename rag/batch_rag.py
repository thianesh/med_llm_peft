import os, glob, math, asyncio, aiohttp, ujson, uuid
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.init import Auth

DATA_DIR = "C:/Users/maths/prof/waveate_verba/rag/test_upload"
MODEL = "hf.co/mradermacher/MedEmbed-large-v0.1-GGUF:Q4_K_M"  # Ollama model tag
OLLAMA_URL = "http://127.0.0.1:11434/api/embed"              # /api/embed supports batch
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_EMBED = 128          # how many chunks to embed per HTTP request
EMBED_CONCURRENCY = 4      # how many embed requests in-flight
WEAV_BATCH_SIZE = 500      # how many objects per Weaviate batch
WEAV_CONCURRENCY = 4       # how many concurrent batch sends
COLLECTION = "medbooks"

def chunk_text(txt, size, overlap):
    # simple greedy chunker
    out, i, n = [], 0, len(txt)
    while i < n:
        j = min(n, i + size)
        out.append(txt[i:j])
        i = j - overlap if j < n else j
    return out

# ---- Embedder (async, batched) ----
async def embed_many(texts):
    """
    texts: list[str], returns list[list[float]] matching order
    """
    sem = asyncio.Semaphore(EMBED_CONCURRENCY)
    async with aiohttp.ClientSession(json_serialize=ujson.dumps) as session:

        async def one_batch(batch):
            # keepalive helps throughput on Ollama
            payload = {"model": MODEL, "input": batch, "options": {"keep_alive": "5m"}}
            async with sem:
                async with session.post(OLLAMA_URL, json=payload, timeout=aiohttp.ClientTimeout(total=600)) as r:
                    if r.status != 200:
                        txt = await r.text()
                        raise RuntimeError(f"Ollama embed failed {r.status}: {txt[:300]}")
                    data = await r.json(loads=ujson.loads)
                    # Ollama returns {"embeddings": [[...], ...]} for batch input
                    embs = data.get("embeddings")
                    if not embs or len(embs) != len(batch):
                        raise RuntimeError("Embedding count mismatch")
                    return embs

        # split into batches
        batches = [texts[i:i+BATCH_EMBED] for i in range(0, len(texts), BATCH_EMBED)]
        results = []
        pbar = tqdm(total=len(batches), desc="Embedding batches")
        # run in waves so order is preserved
        for i in range(0, len(batches), EMBED_CONCURRENCY):
            slice_batches = batches[i:i+EMBED_CONCURRENCY]
            chunk_embs = await asyncio.gather(*[one_batch(b) for b in slice_batches])
            for em in chunk_embs:
                results.extend(em)
            pbar.update(len(slice_batches))
        pbar.close()
        return results

# ---- Weaviate client & collection ----
client = weaviate.connect_to_local()  # or connect_to_custom if remote/auth
coll = client.collections.get(COLLECTION)

# ---- Load & chunk files (this part stays synchronous) ----
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
if not files:
    raise SystemExit(f"No .txt files found under {DATA_DIR}")

# choose subset or all files
selected_files = files[-1:]  # or files[-1:] like before

objects, texts_for_embed = [], []
for path in selected_files:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    base = os.path.basename(path)
    for i, c in enumerate(chunks):
        s = c.strip()
        if not s:
            continue
        # prepare object now; vector later
        objects.append({
            "properties": {"path": base, "chunk_id": i, "text": s},
            "uuid": str(uuid.uuid4()),  # generate ahead so vector aligns to object
        })
        texts_for_embed.append(s)

print(f"Prepared {len(objects)} chunks from {len(selected_files)} file(s)")

# ---- 1) Get embeddings (batched + concurrent) ----
vectors = asyncio.run(embed_many(texts_for_embed))
assert len(vectors) == len(objects)

# ---- 2) Insert with Weaviate batcher (fixed size + concurrent) ----
print("Inserting into Weaviate...")
with coll.batch.fixed_size(batch_size=WEAV_BATCH_SIZE, concurrent_requests=WEAV_CONCURRENCY) as batch:
    pbar = tqdm(total=len(objects), desc="Weaviate insert")
    # stream through in memory-light slices to reduce Python overhead
    for i in range(0, len(objects), WEAV_BATCH_SIZE):
        end = i + WEAV_BATCH_SIZE
        slice_objs = objects[i:end]
        slice_vecs = vectors[i:end]
        # add to batch
        for o, v in zip(slice_objs, slice_vecs):
            batch.add_object(
                properties=o["properties"],
                uuid=o["uuid"],
                vector=v,             # bring-your-own vector
            )
        pbar.update(len(slice_objs))
    pbar.close()

# optional: check batch errors
if coll.batch.failed_objects:
    print(f"Failed objects: {len(coll.batch.failed_objects)} (first few shown)")
    for e in coll.batch.failed_objects[:5]:
        print(e.message)

print(f"Done. Inserted ~{len(objects) - len(coll.batch.failed_objects)} objects.")
client.close()
