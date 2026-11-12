-- =====================================================================
-- 02_deploy_streamlit.sql
-- Manual deployment of the dashboard (alternative to `snow streamlit deploy`).
-- Prerequisite: 01_setup_and_load.sql has run and model/train_register.py
-- has written the evidence tables and registered the models.
--
-- IMPORTANT: run every step (01 SQL, the training notebook, and this file)
-- with the same role. The app runs with its owner's rights, so the role that
-- owns the app must also own (or be granted access to) the tables and models.
-- Do not deploy the app as ACCOUNTADMIN.
-- =====================================================================
USE ROLE SYSADMIN;
USE WAREHOUSE DASHBOARD_WH;
USE SCHEMA DEVICE_FINANCING_DB.FINANCING;

-- 1) Upload the app files, keeping the folder layout app/ and app/.streamlit/.
--    Option A: Snowflake CLI / SnowSQL
-- PUT file:///path/to/device-financing-dashboard/app/*.py              @STREAMLIT_STAGE/app/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file:///path/to/device-financing-dashboard/app/environment.yml   @STREAMLIT_STAGE/app/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- PUT file:///path/to/device-financing-dashboard/app/.streamlit/config.toml @STREAMLIT_STAGE/app/.streamlit/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
--    Option B: Snowsight > Catalog > Database Explorer > STREAMLIT_STAGE > "+ Files",
--    with path "app" for the .py files and environment.yml, and "app/.streamlit" for config.toml.

LIST @STREAMLIT_STAGE/app/;   -- expect 8 files

-- 2) Create the Streamlit object.
--    RUNTIME_NAME is required: newer accounts default to the container runtime,
--    which ignores environment.yml. This app is built for the warehouse runtime.
--    Re-run this statement after any change to the staged files; the app keeps
--    its own copy of the files taken at creation time.
CREATE OR REPLACE STREAMLIT DEVICE_FINANCING_DASHBOARD
  FROM '@DEVICE_FINANCING_DB.FINANCING.STREAMLIT_STAGE/app'
  MAIN_FILE = 'streamlit_app.py'
  RUNTIME_NAME = 'SYSTEM$WAREHOUSE_RUNTIME'
  QUERY_WAREHOUSE = DASHBOARD_WH
  TITLE = 'Device financing decisions';

-- 3) Quick checks that the app's dependencies exist
SELECT EVAL_SAMPLE, MODEL, N, LOG_LOSS, AUC FROM MODEL_METRICS ORDER BY EVAL_SAMPLE, LOG_LOSS;
SHOW MODELS IN SCHEMA DEVICE_FINANCING_DB.FINANCING;
LIST @MODEL_STAGE;

-- 4) Pilot access (optional). Representatives use Application review / Offer
--    comparison; managers also use Portfolio overview. Streamlit in Snowflake
--    runs with the owner's rights, so page-level restriction needs app logic;
--    these grants control who can open the app at all.
-- USE ROLE SECURITYADMIN;
-- CREATE ROLE IF NOT EXISTS PILOT_MANAGER;
-- CREATE ROLE IF NOT EXISTS PILOT_REP;
-- GRANT USAGE ON DATABASE DEVICE_FINANCING_DB TO ROLE PILOT_MANAGER;
-- GRANT USAGE ON SCHEMA DEVICE_FINANCING_DB.FINANCING TO ROLE PILOT_MANAGER;
-- GRANT USAGE ON STREAMLIT DEVICE_FINANCING_DB.FINANCING.DEVICE_FINANCING_DASHBOARD TO ROLE PILOT_MANAGER;
-- (repeat the three grants for PILOT_REP)