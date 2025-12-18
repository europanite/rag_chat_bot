-- Simple seed table just to verify connectivity.
\connect app_db
SET search_path TO app, public;

CREATE TABLE IF NOT EXISTS ping (
  id SERIAL PRIMARY KEY,
  message TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

INSERT INTO ping (message)
SELECT 'hello from postgres'
WHERE NOT EXISTS (SELECT 1 FROM ping);
