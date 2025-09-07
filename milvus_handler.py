import os
from pymilvus import MilvusClient, connections, Collection, DataType

class MilvusHandler:
    """
    Singleton handler for managing a Milvus collection, including schema creation,
    inserting embeddings, and performing similarity searches.

    This class abstracts connection management, collection schema setup, and 
    CRUD-like operations for Milvus vector database.

    Attributes:
        instance (MilvusHandler): Singleton instance of the handler.
        is_initialized (bool): Indicates whether the handler has been initialized.
        ip (str): IP address of the Milvus server.
        port (int): Port number of the Milvus server.
        collection_name (str): Name of the Milvus collection.
        dim (int): Dimension of the embedding vectors.
    """
    #needed for new
    instance = None
    is_initialized = False
    
    def __init__(self, ip=None, port=19530, collection_name="codPenal_collection", dim=3072):

        if not MilvusHandler.is_initialized:
            self.ip = ip or os.getenv("MILVUS_IP", "localhost")
            self.port = port
            self.collection_name = collection_name
            self.dim = dim

            # Connect client
            self.client = MilvusClient(uri=f"http://{self.ip}:{self.port}")
            self.conn = connections.connect(host=self.ip, port=self.port)

            # Ensure schema + collection
            self.schema = self._create_schema()
            self.collection = self._create_or_load_collection()

            MilvusHandler.is_initialized = True

    def __new__(cls):

        # is verify the memory address
        if(cls.instance is None and cls.is_initialized is False):
            cls.instance = super(MilvusHandler, cls).__new__(cls)
   
        return cls.instance
    
    def _create_schema(self):
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
        return schema

    def _create_or_load_collection(self):
        collection = Collection(name=self.collection_name, schema=self.schema)
        index_params = {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
            "metric_type": "COSINE"
        }
        collection.create_index("embedding", index_params)
        collection.load()
        return collection

    def insert(self, data: list):
        """Insert a list of dicts {text: str, embedding: list}"""
        return self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )

    def search(self, embedding: list, top_k=2):
        """Search most similar entries to given embedding"""
        res = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=top_k,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["text"]
        )

        hits = res[0]
        results = []
        for h in hits:
            results.append({
                "id": getattr(h, "id", getattr(h, "pk", -1)),
                "score": getattr(h, "distance", getattr(h, "score", 0.0)),
                "text": h.get("text") if hasattr(h, "get") else None,
            })
        return results



# # INSERT MILVUS

#SET UP FOR PROD
# embeddings = []
# articles = parse_legal_articles(codPenal)

# for ix, article in enumerate(articles) :
#     embeddings.append({ "text" : str(article), "embedding" : get_embedding(str(article))})

# print(len(embeddings))

# insert_data(db_name, embeddings)


#SET UP FOR TEST
# embeddingsTest = []
# articlesTest = [articles[0],articles[1]]

# for ix, article in enumerate(articlesTest):
#     embeddingsTest.append({ "text" : str(article), "embedding" : get_embedding(str(article))})

# print(len(embeddingsTest))

# insert_data(db_name_test, embeddingsTest)



