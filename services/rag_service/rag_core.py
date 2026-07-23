"""
RAG retrieval core logic for the standalone rag_service microservice.

Extracted verbatim (paths adjusted to be self-contained under this service's
own directory) from the monolith's rag_retrieval.py. Same responsibility as
before: given text, find similar past labelled examples in the FAISS vector
store built from the HeBERT training corpus. Does not classify or decide
anything itself.
"""
import json
import os

from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_FILE = os.path.join(SERVICE_DIR, "training", "distress_dataset_final.txt")
FAISS_INDEX_DIR = os.path.join(SERVICE_DIR, "training", "faiss_index")
INDEX_METADATA_FILE = os.path.join(FAISS_INDEX_DIR, "source_meta.json")

EMBEDDING_MODEL_ID = os.environ.get("RAG_EMBEDDING_MODEL_ID", "cohere.embed-multilingual-v3")
EMBEDDING_QUERY_MAX_CHARS = 2000
BEDROCK_AWS_REGION = os.environ.get("BEDROCK_AWS_REGION", "us-east-1")
TOP_K = int(os.environ.get("RAG_TOP_K", "3"))

LABEL_TO_TEXT = {"0": "רגיל / ללא סימני מצוקה", "1": "מצוקה מזוהה"}


def _load_documents(path: str = TRAINING_FILE) -> list[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"RAG knowledge base file not found: '{path}'")

    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n").strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                continue

            text, label = parts[0].strip(), parts[1].strip()
            if text == "text" and label == "label":
                continue
            if not text or label not in LABEL_TO_TEXT:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={"label": label, "label_text": LABEL_TO_TEXT[label], "source_line": line_num},
                )
            )

    if not documents:
        raise ValueError(f"No usable rows parsed from '{path}'")

    return documents


class RAGContextRetriever:
    """Single responsibility: retrieve similar past-labelled examples for a given text."""

    def __init__(
        self,
        data_path: str = TRAINING_FILE,
        embedding_model_id: str = EMBEDDING_MODEL_ID,
        aws_region: str = BEDROCK_AWS_REGION,
        top_k: int = TOP_K,
    ):
        self.top_k = top_k
        self._data_path = data_path
        self._embedding_model_id = embedding_model_id
        self.embeddings = BedrockEmbeddings(
            model_id=embedding_model_id, region_name=aws_region, normalize=True
        )
        self.vector_store: FAISS | None = None
        self.retriever = None

    def build_index(self) -> None:
        if self._load_cached_index():
            return

        documents = _load_documents(self._data_path)
        self.vector_store = FAISS.from_documents(
            documents, self.embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        self._save_index()
        print(
            f"[RAG] FAISS index built with {len(documents)} documents from "
            f"'{self._data_path}' and cached to '{FAISS_INDEX_DIR}'."
        )

    def _source_signature(self) -> dict:
        stat = os.stat(self._data_path)
        return {
            "source_path": os.path.abspath(self._data_path),
            "source_mtime": stat.st_mtime,
            "source_size": stat.st_size,
            "embedding_model": self._embedding_model_id,
        }

    def _load_cached_index(self) -> bool:
        if not os.path.exists(INDEX_METADATA_FILE):
            return False

        try:
            with open(INDEX_METADATA_FILE, "r", encoding="utf-8") as f:
                cached_signature = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False

        if cached_signature != self._source_signature():
            print("[RAG] Cached FAISS index is stale (source file/model changed) - rebuilding.")
            return False

        try:
            self.vector_store = FAISS.load_local(
                FAISS_INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"[RAG] Failed to load cached FAISS index ({e}) - rebuilding.")
            return False

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        print(
            f"[RAG] Loaded cached FAISS index from '{FAISS_INDEX_DIR}' "
            f"({self.vector_store.index.ntotal} vectors)."
        )
        return True

    def _save_index(self) -> None:
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        self.vector_store.save_local(FAISS_INDEX_DIR)
        with open(INDEX_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self._source_signature(), f)

    def retrieve(self, text: str, k: int | None = None) -> list[dict]:
        if self.vector_store is None:
            raise RuntimeError("RAG index not built yet - call build_index() first.")

        if not text or not text.strip():
            return []

        query_text = text[:EMBEDDING_QUERY_MAX_CHARS]
        results = self.vector_store.similarity_search_with_score(query_text, k=k or self.top_k)

        return [
            {
                "text": doc.page_content,
                "label": doc.metadata.get("label_text", "לא ידוע"),
                "similarity_score": round(float(score), 4),
            }
            for doc, score in results
        ]

    def build_context(self, text: str, k: int | None = None) -> tuple[str, list[dict]]:
        examples = self.retrieve(text, k=k)

        if not examples:
            return "לא נמצא הקשר רלוונטי במאגר הידע.", []

        lines = [
            f'- "{ex["text"]}" (תויג בעבר כ: {ex["label"]}, ציון דמיון: {ex["similarity_score"]})'
            for ex in examples
        ]
        context = "מקרים דומים שאותרו במאגר הידע:\n" + "\n".join(lines)
        return context, examples
