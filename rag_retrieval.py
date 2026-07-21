import json
import os

# רכיבי LangChain: Document (סכמת המסמכים), BedrockEmbeddings ו-FAISS (Vector DB) הם
# כולם ממשקי LangChain רשמיים (langchain_core / langchain_community / langchain_aws).
# ה-embeddings עצמם נקראים דרך Amazon Bedrock - אותו client/credentials/IAM שכבר
# מוגדרים ב-distress_classification.py (Bedrock), אין ספק תוכנה/API חדש בפרויקט.
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

# מאגר הידע של ה-RAG: אותו קובץ אימון של HeBERT (ml_comprehend.py), text<TAB>label.
# חשוב: אלו לא "מסמכי מדיניות" אלא הודעות היסטוריות שכבר תויגו (0=רגיל, 1=מצוקה) -
# ה-RAG לא מסווג בעצמו, הוא רק מאתר מקרים דומים מהעבר כדי לתת לסוכן ההחלטה
# (Decision Agent) הקשר תומך מעבר לסיווג הבודד שמגיע מ-Bedrock.
TRAINING_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "training", "distress_dataset_final.txt"
)

# מיקום שמירת ה-FAISS index על הדיסק (תוצר בנייה - לא חלק ממאגר המקור, ולכן ב-.gitignore),
# כדי שלא נצטרך להריץ embedding מחדש על כל 4992 השורות בכל startup של safesignal.py.
FAISS_INDEX_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "training", "faiss_index"
)
INDEX_METADATA_FILE = os.path.join(FAISS_INDEX_DIR, "source_meta.json")

# מודל ה-embeddings של Cohere על Amazon Bedrock (מולטילינגואלי - תומך גם עברית וגם
# אנגלית) - נקרא דרך אותו חשבון AWS/Bedrock שכבר משמש את distress_classification.py,
# אותו BEDROCK_AWS_REGION env var לעקביות.
EMBEDDING_MODEL_ID = os.environ.get("RAG_EMBEDDING_MODEL_ID", "cohere.embed-multilingual-v3")

# Cohere's Bedrock embedding model hard-rejects (ValidationException) any single
# input over 2048 characters. Only the search *query* passes through here (the
# indexed knowledge-base documents are already short example sentences) -- this
# is a similarity-search lookup that enriches the Decision Agent's context, not
# the safety-critical distress classification itself (that's HeBERT's job, which
# chunks the full text -- see ml_comprehend.py), so truncating a long scraped
# page down to its opening chars is an acceptable tradeoff here.
EMBEDDING_QUERY_MAX_CHARS = 2000
BEDROCK_AWS_REGION = os.environ.get("BEDROCK_AWS_REGION", "us-east-1")

TOP_K = int(os.environ.get("RAG_TOP_K", "3"))

LABEL_TO_TEXT = {"0": "רגיל / ללא סימני מצוקה", "1": "מצוקה מזוהה"}


def _load_documents(path: str = TRAINING_FILE) -> list[Document]:
    """
    טוען את קובץ ה-training (text<TAB>label) לרשימת Document של LangChain.
    כל שורה היא כבר יחידת טקסט קצרה ועצמאית (פוסט בודד) - אין צורך בפיצול נוסף
    (text splitting), רק בניקוי שורות ריקות/כותרת.
    """
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
                continue  # שורת הכותרת של הקובץ
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
    """
    צומת ה-Context Retrieval בפייפליין (RAG - LangChain + Vector DB).
    אחריות יחידה: להביא הקשר תומך-החלטה מבוסס מקרים דומים מהעבר. הוא לא מסווג
    ולא מחליט בעצמו - זה תפקידם של Bedrock (distress_classification.py) וה-
    Decision Agent שאחריו.
    """

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
        # normalize=True + MAX_INNER_PRODUCT (IndexFlatIP) להלן = בדיוק אותו רעיון כמו
        # faiss.normalize_L2 + IndexFlatIP בסקריפט המקורי: inner product בין וקטורים
        # מנורמלים שקול לדמיון קוסינוס. (distance_strategy=COSINE ב-langchain-community
        # מטעה: הוא בפועל עדיין בונה IndexFlatL2 גולמי על הווקטורים הלא-מנורמלים.)
        self.embeddings = BedrockEmbeddings(
            model_id=embedding_model_id, region_name=aws_region, normalize=True
        )
        self.vector_store: FAISS | None = None
        self.retriever = None

    def build_index(self) -> None:
        """
        שלב ה-Ingestion (חובה): טוען FAISS index שמור מהדיסק אם קיים ותואם למקור
        (_load_cached_index), אחרת בונה מחדש - טעינת מאגר הדוגמאות + embeddings +
        בניית FAISS index, ושומר אותו לדיסק לפעם הבאה. נקרא פעם אחת ב-startup של
        safesignal.py (lifespan) - בדיוק כמו טעינת HeBERT/Ollama בשלבים האחרים.
        """
        if self._load_cached_index():
            return

        documents = _load_documents(self._data_path)
        self.vector_store = FAISS.from_documents(
            documents, self.embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )
        # ממשק ה-Retriever הסטנדרטי של LangChain (BaseRetriever/Runnable) - זה האובייקט
        # שכל שרשרת/סוכן LangChain אחר בהמשך (כולל Decision Agent) יכול לצרוך ישירות
        # דרך retriever.invoke(query), בלי לדעת כלום על FAISS שמאחוריו. retrieve() למטה
        # משתמש בווקטור-store ישירות כי הוא צריך גם ציוני דמיון (ה-Retriever הסטנדרטי
        # מחזיר רק Document-ים, בלי ציון) - שתי הדרכים הן LangChain API רשמי.
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        self._save_index()
        print(
            f"[RAG] FAISS index built with {len(documents)} documents from "
            f"'{self._data_path}' and cached to '{FAISS_INDEX_DIR}'."
        )

    def _source_signature(self) -> dict:
        """
        'טביעת אצבע' של מקור הנתונים + מודל ה-embeddings, לזיהוי מתי הקאש בדיסק
        עדיין תקף מול מתי הוא צריך להיבנות מחדש (הקובץ השתנה / המודל השתנה).
        """
        stat = os.stat(self._data_path)
        return {
            "source_path": os.path.abspath(self._data_path),
            "source_mtime": stat.st_mtime,
            "source_size": stat.st_size,
            "embedding_model": self._embedding_model_id,
        }

    def _load_cached_index(self) -> bool:
        """
        טוען FAISS index שמור מ-FAISS_INDEX_DIR אם קיים ותואם למקור הנוכחי - חוסך
        embedding מחדש של כל המאגר בכל startup. מחזיר False (ולא זורק) בכל מצב
        שבו הקאש חסר/לא תקף/פגום, כדי ש-build_index() תמיד ידע לבנות מחדש בבטחה.
        """
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
        """
        מאתר את k הדוגמאות הדומות ביותר בבסיס הידע. מחזיר dict-ים פשוטים
        (לא אובייקטי LangChain) כדי שהתוצאה תהיה JSON נקי שאפשר להעביר הלאה.
        """
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
        """
        עוטף retrieve() ומחזיר גם מחרוזת context קריאה-לאדם (context) וגם את
        הרשימה הגולמית (retrieved_examples), כדי שהצומת הבא (Decision Agent)
        יוכל להשתמש בכל צורה שנוחה לו.
        """
        examples = self.retrieve(text, k=k)

        if not examples:
            return "לא נמצא הקשר רלוונטי במאגר הידע.", []

        lines = [
            f'- "{ex["text"]}" (תויג בעבר כ: {ex["label"]}, ציון דמיון: {ex["similarity_score"]})'
            for ex in examples
        ]
        context = "מקרים דומים שאותרו במאגר הידע:\n" + "\n".join(lines)
        return context, examples


# ==========================================================
#      אזור בדיקת המודול (Main Simulation)
# ==========================================================
if __name__ == "__main__":
    retriever = RAGContextRetriever()
    retriever.build_index()

    test_posts = [
        "אין לי כוח לקום מהמיטה כבר שבוע, לא יודעת מה לעשות.",
        "מישהו יודע איפה אפשר להזמין פיצה טובה באזור?",
        "כולם מתעלמים ממני בבית ספר כאילו אני לא קיים",
    ]

    print("\n--- מריץ בדיקות אחזור RAG ---")
    for i, post in enumerate(test_posts, 1):
        context, examples = retriever.build_context(post)
        print(f"\n[פוסט בדיקה {i}] {post}")
        print(context)
        print("-" * 50)
