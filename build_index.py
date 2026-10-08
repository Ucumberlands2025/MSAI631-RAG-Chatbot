"""Explicit build/rebuild command for the document embedding index."""
from rag import DOCS_DIR, EMBEDDING_MODEL, INDEX_DIR, build_store


def main() -> None:
    from sentence_transformers import SentenceTransformer
    print(f"Loading {EMBEDDING_MODEL} on CPU...")
    encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    count = build_store(encoder, docs=DOCS_DIR, destination=INDEX_DIR)
    print(f"Indexed {count} passages into {INDEX_DIR}.")


if __name__ == "__main__":
    main()
