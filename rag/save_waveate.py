import weaviate
import requests, json
from weaviate.classes.config import Configure

client = weaviate.connect_to_local()

client.close()