-- =====================================================================
-- 01_setup_and_load.sql
-- Device-financing dashboard: Snowflake objects + data load
-- Run in a Snowsight worksheet as a role that can create databases
-- (for example SYSADMIN). Edit the names below if your account requires it.
-- =====================================================================
USE ROLE SYSADMIN;

CREATE WAREHOUSE IF NOT EXISTS DASHBOARD_WH
  WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;

CREATE DATABASE IF NOT EXISTS DEVICE_FINANCING_DB;
CREATE SCHEMA IF NOT EXISTS DEVICE_FINANCING_DB.FINANCING;

USE WAREHOUSE DASHBOARD_WH;
USE SCHEMA DEVICE_FINANCING_DB.FINANCING;

-- Stages: raw CSV, model artifacts, Streamlit source files
CREATE STAGE IF NOT EXISTS RAW_STAGE       DIRECTORY = (ENABLE = TRUE);
CREATE STAGE IF NOT EXISTS MODEL_STAGE     DIRECTORY = (ENABLE = TRUE);
CREATE STAGE IF NOT EXISTS STREAMLIT_STAGE DIRECTORY = (ENABLE = TRUE);

-- The source CSV quotes strings and writes missing default_flag as NA.
CREATE OR REPLACE FILE FORMAT CSV_FINANCING
  TYPE = CSV
  SKIP_HEADER = 1
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
  NULL_IF = ('NA', '')
  EMPTY_FIELD_AS_NULL = TRUE;

-- ---------------------------------------------------------------------
-- Upload the file before running the COPY below. Either:
--   (a) Snowsight: Catalog > Database Explorer > DEVICE_FINANCING_DB > FINANCING >
--       Stages > RAW_STAGE > "+ Files", choose financing_data.csv; or
--   (b) Snowflake CLI / SnowSQL:
--       PUT file:///path/to/financing_data.csv @RAW_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
-- ---------------------------------------------------------------------

CREATE OR REPLACE TABLE APPLICATIONS (
  APPLICATION_ID       NUMBER(10,0),   -- source file row number; preserves CSV order
  APPLICATION_DATE     DATE,
  RACE                 STRING,
  HISP                 STRING,
  GENDER               STRING,
  AGE                  NUMBER(3,0),
  INCOME               FLOAT,
  CREDIT_SCORE         NUMBER(4,0),
  EXISTING_CUSTOMER    NUMBER(1,0),
  DEVICE_REQUESTED     STRING,
  DEVICE_FINAL         STRING,
  DEVICE_PRICE         FLOAT,
  DOWN_PAYMENT_BASE    FLOAT,
  DOWN_PAYMENT_FINAL   FLOAT,
  INTERNAL_RISK_SCORE  FLOAT,
  INITIAL_DECISION     STRING,
  FINAL_STATUS         STRING,
  DEFAULT_FLAG         NUMBER(1,0),    -- NULL = not funded, outcome unknown (NOT a non-default)
  PROFIT               FLOAT
);

-- APPLICATION_ID keeps the original row order, so the training script can
-- reproduce the notebook's stratified 75/25 split (random_state = 42).
COPY INTO APPLICATIONS
FROM (
  SELECT METADATA$FILE_ROW_NUMBER,
         $1, $2, $3, $4, $5, $6, $7, $8, $9,
         $10, $11, $12, $13, $14, $15, $16, $17, $18
  FROM @RAW_STAGE/financing_data.csv
)
FILE_FORMAT = (FORMAT_NAME = 'CSV_FINANCING')
ON_ERROR = 'ABORT_STATEMENT';

-- ---------------------------------------------------------------------
-- Reconciliation checks (requirement: reconcile counts with the EDA HTML)
-- Expected: 300,000 rows; 265,060 observed outcomes; 34,940 missing.
-- ---------------------------------------------------------------------
SELECT COUNT(*)                                        AS APPLICATIONS,
       COUNT(DEFAULT_FLAG)                             AS OBSERVED_OUTCOMES,
       COUNT(*) - COUNT(DEFAULT_FLAG)                  AS MISSING_OUTCOMES,
       SUM(IFF(FINAL_STATUS = 'Not_Funded', 1, 0))     AS NOT_FUNDED,
       MIN(APPLICATION_DATE)                           AS FIRST_DATE,
       MAX(APPLICATION_DATE)                           AS LAST_DATE
FROM APPLICATIONS;

-- Missing outcomes must coincide exactly with Not_Funded plans.
SELECT INITIAL_DECISION, FINAL_STATUS,
       COUNT(*) AS N, COUNT(DEFAULT_FLAG) AS N_WITH_OUTCOME
FROM APPLICATIONS
GROUP BY 1, 2
ORDER BY 1, 2;