import chromadb

client = chromadb.PersistentClient('./chroma_cbx')
collection = client.get_or_create_collection('cbx_trauma')
existing = collection.get(include=[])
print('Documents in collection:', len(existing.get('ids', [])))
