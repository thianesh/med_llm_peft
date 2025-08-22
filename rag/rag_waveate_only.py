import os, glob, uuid
from tqdm import tqdm
import weaviate

DATA_DIR = "C:/Users/maths/prof/waveate_verba/rag/test_upload"  # input .txt dir
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
WEAV_BATCH_SIZE = 500      # objects per Weaviate batch
WEAV_CONCURRENCY = 4       # parallel batch requests
COLLECTION = "medbooks"    # already created & vectorizer-configured

def chunk_text(txt: str, size: int, overlap: int):
    out, i, n = [], 0, len(txt)
    while i < n:
        j = min(n, i + size)
        out.append(txt[i:j])
        i = j - overlap if j < n else j
    return out

# ---- Weaviate client & target collection ----
client = weaviate.connect_to_local()   # use connect_to_custom(...) if remote/auth
coll = client.collections.get(COLLECTION)

# ---- Load & chunk files ----
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
if not files:
    raise SystemExit(f"No .txt files found under {DATA_DIR}")

selected_files = files[-1:]  # change as needed (e.g., files for all)
objects = []

for path in selected_files:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    base = os.path.basename(path)

    for i, c in enumerate(chunks):
        s = c.strip()
        if not s:
            continue

        # IMPORTANT: only properties; DO NOT pass vector
        objects.append({
            "properties": {"path": base, "chunk_id": i, "text": s},
            "uuid": str(uuid.uuid4()),
        })

print(f"Prepared {len(objects)} chunks from {len(selected_files)} file(s)")

# ---- Batch insert (server will vectorize from your configured vectorizer) ----
print("Inserting into Weaviate...")
with coll.batch.fixed_size(
    batch_size=WEAV_BATCH_SIZE,
    concurrent_requests=WEAV_CONCURRENCY
) as batch:
    pbar = tqdm(total=len(objects), desc="Weaviate insert")
    for i in range(0, len(objects), WEAV_BATCH_SIZE):
        slice_objs = objects[i:i + WEAV_BATCH_SIZE]
        for o in slice_objs:
            batch.add_object(
                properties=o["properties"],
                uuid=o["uuid"]
                # NO 'vector=' here -> Weaviate vectorizes automatically
            )
        pbar.update(len(slice_objs))
    pbar.close()

# Optional: report failed objects, if any
if coll.batch.failed_objects:
    print(f"Failed objects: {len(coll.batch.failed_objects)} (first few shown)")
    for e in coll.batch.failed_objects[:5]:
        print(e.message)

print(f"Done. Inserted ~{len(objects) - len(coll.batch.failed_objects)} objects.")
client.close()
