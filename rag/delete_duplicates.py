import time
from tqdm import tqdm
import weaviate

COLLECTION = "medbooks"

client = weaviate.connect_to_local()
coll = client.collections.get(COLLECTION)

seen = set()
dupe_ids = []
total = 0

for obj in tqdm(coll.iterator(return_properties=["path", "chunk_id"]), desc="Scan"):
    total += 1
    p = obj.properties or {}
    key = (p.get("path"), p.get("chunk_id"))
    if key in seen:
        dupe_ids.append(obj.uuid)
    else:
        seen.add(key)

print(f"Scanned: {total}, uniques: {len(seen)}, duplicates: {len(dupe_ids)}")

BATCH = 256
for i in tqdm(range(0, len(dupe_ids), BATCH), desc="Delete"):
    for _id in dupe_ids[i:i+BATCH]:
        coll.data.delete_by_id(_id)
    time.sleep(0.05)

client.close()
print("Done.")
