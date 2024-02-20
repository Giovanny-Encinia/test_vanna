import json
from typing import List
import uuid
from abc import abstractmethod
import pandas as pd
import redis
import numpy as np
from redis.commands.search.query import Query
from dotenv import load_dotenv, find_dotenv
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from ..base import SqlCxCopilotBase
import os
from openai import AzureOpenAI
from langchain_community.vectorstores.redis import Redis
from redis.commands.search.field import (
    NumericField,
    TextField,
    VectorField,
)
from typing import Any, List, Optional
from redis.commands.search.query import Query
import numpy as np
import time
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()
default_ef = "embedding_functions.DefaultEmbeddingFunction()"
VECTOR_DIMENSION = 1536
REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = os.environ["REDIS_PORT"]
REDIS_PASSWORD = os.environ["REDIS_PASSWORD"]
REDIS_URL = f"redis://{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}"


def connect_openai():
    _ = load_dotenv(find_dotenv())
    # openai.api_key = os.environ["OPENAI_API_KEY"]
    # openai.api_base = os.environ["OPENAI_API_BASE"]
    # openai.api_type = os.environ["API_TYPE"]
    # openai.api_version = os.environ["API_VERSION"]
    azure_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    return azure_client


def check_index_exists(client, index_name):

    pattern = f"{index_name}:*"
    keys = []
    cursor = 0

    while True:
        print("cheking index exists")
        cursor, partial_keys = client.scan(cursor, match=pattern)

        if partial_keys:
            break

        keys.extend(partial_keys)

        if cursor == 0:
            break

    if partial_keys:
        return True
    else:
        return False


class RedisVectorStore(SqlCxCopilotBase):
    def __init__(
        self,
        config={
            "host": REDIS_HOST,
            "port": REDIS_PORT,
            "password": REDIS_PASSWORD,
            "decode_responses": True,
            "ssl": True,
            "openai_embedding_engine": "CX_EMB_TEST",
        },
    ):
        SqlCxCopilotBase.__init__(self, config=config)
        self.embeddings = None
        self.azure_client = connect_openai()
        self.redis_name_prefix = os.getenv("redis_name_prefix", None)
        self.openai_embedding_engine = config.pop("openai_embedding_engine", None)
        schema_sql = (
            TextField("$.question", as_name="question"),
            TextField("$.sql", as_name="sql"),
            VectorField(
                "$.embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT64",
                    "DIM": VECTOR_DIMENSION,
                    "DISTANCE_METRIC": "IP",
                },
                as_name="embedding",
            ),
        )
        schema_ddl = (
            TextField("$.ddl", as_name="ddl"),
            VectorField(
                "$.embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT64",
                    "DIM": VECTOR_DIMENSION,
                    "DISTANCE_METRIC": "IP",
                },
                as_name="embedding",
            ),
        )
        schema_doc = (
            TextField("$.doc", as_name="doc"),
            VectorField(
                "$.embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT64",
                    "DIM": VECTOR_DIMENSION,
                    "DISTANCE_METRIC": "IP",
                },
                as_name="embedding",
            ),
        )

        self.redis_client = redis.Redis(**config)
        metadata = zip(["sql", "ddl", "doc"], [schema_sql, schema_ddl, schema_doc])

        for subname, schema in metadata:
            redis_name = self.redis_name_prefix + subname
            prefix = "{" + f"{redis_name}" + "}:"
            # it_exists = check_index_exists(self.redis_client, policy_str)

            # if not it_exists:

            definition = IndexDefinition(prefix=[prefix], index_type=IndexType.JSON)
            index_name = f"idx:ID{redis_name}"

            try:
                res = self.redis_client.ft(index_name).create_index(
                    fields=schema, definition=definition
                )
            except Exception as e:
                print(e, "jumping to next")
                pass

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        encoded_queries = (
            self.azure_client.embeddings.create(
                input=[data], model=os.getenv("AZURE_EMBEDDING_ENGINE")
            )
            .data[0]
            .embedding
        )
        return encoded_queries

    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        question_sql_json = {
            "question": question,
            "sql": sql,
        }

        embedding = self.generate_embedding(json.dumps(question_sql_json))
        question_sql_json["embedding"] = embedding
        redis_name = self.redis_name_prefix + "sql"
        prefix = "{" + f"{redis_name}" + "}"
        current_time = time.time()
        redis_key = f"{prefix}:{current_time}"
        r = self.redis_client.json().set(
            redis_key, "$", question_sql_json, f"{redis_name}"
        )

        return r

    def add_ddl(self, ddl: str, **kwargs) -> str:
        ddl_json = {
            "ddl": ddl,
        }

        embedding = self.generate_embedding(json.dumps(ddl_json))
        ddl_json["embedding"] = embedding
        redis_name = self.redis_name_prefix + "ddl"
        prefix = "{" + f"{redis_name}" + "}"
        current_time = time.time()
        redis_key = f"{prefix}:{current_time}"
        r = self.redis_client.json().set(redis_key, "$", ddl_json, f"{redis_name}")

        return r

    def add_documentation(self, documentation: str, **kwargs) -> str:
        doc_json = {
            "doc": documentation,
        }

        embedding = self.generate_embedding(json.dumps(doc_json))
        doc_json["embedding"] = embedding
        redis_name = self.redis_name_prefix + "doc"
        prefix = "{" + f"{redis_name}" + "}"
        current_time = time.time()
        redis_key = f"{prefix}:{current_time}"
        r = self.redis_client.json().set(redis_key, "$", doc_json, f"{redis_name}")

        return r

    def get_training_data(self, **kwargs) -> pd.DataFrame:
        return []

    def remove_training_data(self, id: str, **kwargs) -> bool:
        return []

    def remove_collection(self, collection_name: str) -> bool:
        """
        This function can reset the collection to empty state.

        Args:
            collection_name (str): sql or ddl or documentation

        Returns:
            bool: True if collection is deleted, False otherwise
        """
        return []

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        embedding = self.generate_embedding(question)

        redis_name = self.redis_name_prefix + "sql"
        return_fields = ["question", "sql"]
        results = self.similarity_search_by_vector(
            id_name=redis_name, embedding=embedding, return_fields=return_fields
        )

        return results

    def get_related_ddl(self, question: str, **kwargs) -> list:
        embedding = self.generate_embedding(question)

        redis_name = self.redis_name_prefix + "ddl"
        return_fields = ["ddl"]
        results = self.similarity_search_by_vector(
            id_name=redis_name, embedding=embedding, return_fields=return_fields
        )
        results = [result["ddl"] for result in results]

        return results

    def get_related_label(self, question: str, **kwargs) -> list:
        embedding = self.generate_embedding(question)

        redis_name = self.redis_name_prefix + "label"
        return_fields = ["Label", "column_name"]
        results = self.similarity_search_by_vector(
            id_name=redis_name, embedding=embedding, return_fields=return_fields, k=10
        )
        results = [
            {"column name": result["column_name"], "label": result["Label"]}
            for result in results
        ]

        return results

    def get_related_documentation(self, question: str, **kwargs) -> list:
        embedding = self.generate_embedding(question)

        redis_name = self.redis_name_prefix + "doc"
        return_fields = ["doc"]
        results = self.similarity_search_by_vector(
            id_name=redis_name, embedding=embedding, return_fields=return_fields
        )
        results = [result["doc"] for result in results]

        return results

    def similarity_search_by_vector(
        self,
        embedding,
        k=4,
        return_fields: List[str] = [],
        id_name="",
        distance_threshold=0.21,
    ) -> List[dict]:
        return_fields.append("similarities")
        query = (
            Query(f"(*)=>[KNN {k} @embedding $query_vector AS similarities]")
            .sort_by("similarities")
            .return_fields(*return_fields)
            .dialect(2)
        )

        if isinstance(embedding, list):
            encoded_query = np.array(embedding, dtype=np.float64).tobytes()
        else:
            encoded_query = embedding.tobytes()

        query_params = {"query_vector": encoded_query}
        result_docs = (
            self.redis_client.ft(f"idx:ID{id_name}")
            .search(
                query,
                query_params,
            )
            .docs
        )
        result_list = [
            result.__dict__
            for result in result_docs
            if float(result.__dict__["similarities"]) < 0.9
        ]
        return result_list

    def max_marginal_relevance_search(
        self,
        question: str,
        embedding_client,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[str] = None,
        return_metadata: bool = True,
        distance_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            query (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (RedisFilterExpression, optional): Optional metadata filter.
                Defaults to None.
            return_metadata (bool, optional): Whether to return metadata.
                Defaults to True.
            distance_threshold (Optional[float], optional): Maximum vector distance
                between selected documents and the query vector. Defaults to None.

        Returns:
            List[Document]: A list of Documents selected by maximal marginal relevance.
        """
        # Embed the query
        embedding = (
            embedding_client.embeddings.create(input=[question], model="CX_EMB_TEST")
            .data[0]
            .embedding
        )

        # Fetch the initial documents
        prefetch_docs = self.similarity_search_by_vector(
            embedding,
            k=fetch_k,
            distance_threshold=distance_threshold,
            **kwargs,
        )
        prefetch_ids = [doc.metadata["id"] for doc in prefetch_docs]

        # Get the embeddings for the fetched documents
        prefetch_embeddings = [
            _buffer_to_array(
                cast(
                    bytes,
                    # client.hget(prefetch_id, _schema.content_vector_key),
                ),
                # dtype=_schema.vector_dtype,
            )
            for prefetch_id in prefetch_ids
        ]

        # Select documents using maximal marginal relevance
        selected_indices = maximal_marginal_relevance(
            np.array(query_embedding), prefetch_embeddings, lambda_mult=lambda_mult, k=k
        )
        selected_docs = [prefetch_docs[i] for i in selected_indices]

        return selected_docs
