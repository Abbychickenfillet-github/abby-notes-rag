CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS chunks CASCADE;

CREATE TABLE chunks (
  id           SERIAL PRIMARY KEY,
  file_path    TEXT         NOT NULL,
  file_hash    TEXT         NOT NULL,
  chunk_index  INTEGER      NOT NULL,
  heading_path TEXT,
  content      TEXT         NOT NULL,
  token_count  INTEGER,
  embedding    vector(1024) NOT NULL,
  created_at   TIMESTAMP    DEFAULT NOW(),
  UNIQUE (file_path, chunk_index)
);

CREATE INDEX chunks_embedding_idx
  ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX chunks_file_path_idx ON chunks (file_path);

CREATE INDEX chunks_content_fts_idx
  ON chunks USING GIN (to_tsvector('simple', content));
