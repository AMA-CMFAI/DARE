import os
import logging
import torch
import chromadb
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DARE_R_Retriever:
    def __init__(self, db_download_dir="./rpkb_db", collection_name="inference"):
        # Download the DARE model
        self.model_id = "Stephen-SMJ/DARE-R-Retriever"
        logger.info(f"🔄 Loading DARE model from HF: {self.model_id}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_id, trust_remote_code=False)
        self.model.to(device)
        self.model.eval()
        logger.info(f"✅ Model loaded on {device}.")

        # Download the RPKB
        self.dataset_id = "Stephen-SMJ/RPKB"
        db_target_path = os.path.join(db_download_dir, "DARE_db")

        if not os.path.exists(db_target_path):
            logger.info(f"📥 Downloading RPKB database from HF Datasets: {self.dataset_id}...")
            snapshot_download(
                repo_id=self.dataset_id,
                repo_type="dataset",
                local_dir=db_download_dir,
                allow_patterns="DARE_db/*"
            )
            logger.info("✅ Database downloaded successfully.")
        else:
            logger.info(f"📂 Local database found at {db_target_path}, skipping download.")

        self.chroma_client = chromadb.PersistentClient(path=db_target_path)

        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            count = self.collection.count()
            logger.info(f"✅ Collection '{collection_name}' connected. Total functions: {count}")
        except Exception as e:
            logger.error(f"❌ Failed to get collection '{collection_name}': {e}")
            raise e

    def search(self, query: str, top_k: int = 5):
        print("\n" + "=" * 60)
        print(f"🔍 [DARE Search] Query: {query[:100]}...")
        print("=" * 60)

        with torch.no_grad():
            query_embedding = self.model.encode(query, convert_to_tensor=False).tolist()

        # Retrieval
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        if not results['ids'] or len(results['ids'][0]) == 0:
            print("❌ No results found.")
            return None

        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]

        print(f"🏆 Top {top_k} Retrieval Results:")
        for rank, (doc_id, dist, meta, doc) in enumerate(zip(ids, distances, metadatas, documents)):
            print(f"\n   [{rank + 1}] ID: {doc_id} (Distance: {dist:.4f})")
            print(f"      📦 Package: {meta.get('package_name', 'N/A')} :: Function: {meta.get('function_name', 'N/A')}")

            preview = doc.replace('\n', ' ') + "..."
            print(f"      📄 Content Preview: {preview}")

        return results


if __name__ == "__main__":
    retriever = DARE_R_Retriever()

    test_query = (
        "I need to perform an environmental affinity assignment for a collection of coral fossils. "
        "Please write an R script that processes my occurrence data fossilEnv_data.csv to find the "
        "preferred 'bath' (bathymetry) for each genus across different time bins. Set set.seed(123) "
        "for reproducibility. From the final affinity results, identify the environment that appears "
        "most frequently. Specifically, print the name of this environment, and the count of taxa preferring it."
    )

    retriever.search(test_query, top_k=2)