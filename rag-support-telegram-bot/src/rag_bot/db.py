# PostgreSQL connection pool shared by the vector store and the conversation log
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


def connect(dsn: str, max_connections: int = 10) -> ConnectionPool:
    """Pool of connections that understand the pgvector `vector` type."""
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")  # needs a role allowed to create it
    return ConnectionPool(dsn, min_size=1, max_size=max_connections,
                          kwargs={"row_factory": dict_row}, configure=register_vector, open=True)
