-- Create the app role and database (pure SQL; no env expansion).
-- Keep these names/passwords in sync with your .env.
--   DB_USER=app_user
--   DB_PASS=app_pass
--   DB_NAME=app_db

-- 1) role: DO is fine for roles
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_user') THEN
    CREATE ROLE app_user WITH LOGIN PASSWORD 'app_pass';
  END IF;
END
$$;

-- 2) database: cannot be inside a DO/transaction; use \gexec
SELECT 'CREATE DATABASE app_db OWNER app_user'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'app_db') \gexec

-- 3) connect so that later scripts run against the app DB
\connect app_db
