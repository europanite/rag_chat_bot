-- We are now connected to app_db.
-- Keep all objects under schema "app".
CREATE SCHEMA IF NOT EXISTS app AUTHORIZATION app_user;

-- Make "app" the default schema for this role.
ALTER ROLE app_user SET search_path TO app, public;

-- Optional extensions (safe to call repeatedly).
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Ensure privileges on the schema.
GRANT ALL ON SCHEMA app TO app_user;

-- Use the schema immediately in this session.
SET search_path TO app, public;
