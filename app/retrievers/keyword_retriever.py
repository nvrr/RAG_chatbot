
from typing import Any

from sqlalchemy import text
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict

from app.database import engine


class PostgresFullTextRetriever(BaseRetriever):

    engine: Any
    collection_name: str
    k: int = 3
    language: str = "english"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:

        sql = text("""
            SELECT
                e.id,
                e.document,
                e.cmetadata,
                ts_rank_cd(
                    to_tsvector(
                        :language,
                        COALESCE(e.document, '')
                    ),
                    websearch_to_tsquery(:language, :query)
                ) AS score

            FROM langchain_pg_embedding e

            JOIN langchain_pg_collection c
                ON c.uuid = e.collection_id

            WHERE
                c.name = :collection_name
                AND to_tsvector(
                    :language,
                    COALESCE(e.document, '')
                ) @@ websearch_to_tsquery(:language, :query)

            ORDER BY score DESC
            LIMIT :k
        """)

        with self.engine.connect() as connection:
            results = connection.execute(
                sql,
                {
                    "language": self.language,
                    "query": query,
                    "collection_name": self.collection_name,
                    "k": self.k,
                },
            ).mappings().all()

        documents = []

        for row in results:
            metadata = row["cmetadata"] or {}
            metadata["chunk_id"] = str(row["id"])
            metadata["keyword_score"] = float(row["score"])

            documents.append(
                Document(
                    page_content=row["document"],
                    metadata=metadata,
                )
            )

        return documents
