-- =====================================================================
-- 03_pilot_monitoring.sql
-- Monitoring objects for a controlled pilot (proposal; not needed for the demo).
-- =====================================================================
USE SCHEMA DEVICE_FINANCING_DB.FINANCING;

-- Monthly calibration of the proposed model on funded plans with known outcomes.
CREATE OR REPLACE VIEW V_MONTHLY_CALIBRATION AS
SELECT DATE_TRUNC('MONTH', a.APPLICATION_DATE)             AS MONTH,
       IFF(a.EXISTING_CUSTOMER = 1, 'Existing', 'New')      AS CUSTOMER_STATUS,
       COUNT(*)                                             AS FUNDED_WITH_OUTCOME,
       AVG(a.DEFAULT_FLAG)                                  AS ACTUAL_DEFAULT_RATE,
       AVG(s.PD_HGB)                                        AS PREDICTED_HGB,
       AVG(s.PD_INTERNAL)                                   AS PREDICTED_INTERNAL,
       AVG(a.DEFAULT_FLAG) - AVG(s.PD_HGB)                  AS GAP_HGB
FROM APPLICATIONS a
JOIN APPLICATION_SCORES s ON a.APPLICATION_ID = s.APPLICATION_ID
WHERE a.DEFAULT_FLAG IS NOT NULL
GROUP BY 1, 2;

-- Pilot guardrail: months where the model is off by more than 2 percentage points.
CREATE OR REPLACE VIEW V_CALIBRATION_ALERTS AS
SELECT * FROM V_MONTHLY_CALIBRATION
WHERE ABS(GAP_HGB) > 0.02 AND FUNDED_WITH_OUTCOME >= 500;

-- Log of fast-track decisions during the pilot (filled by the app or the POS integration).
CREATE TABLE IF NOT EXISTS PILOT_DECISION_LOG (
  LOGGED_AT        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
  STORE_ID         STRING,
  REP_ROLE         STRING,
  PILOT_ARM        STRING,      -- 'fast_track' or 'control' (random assignment)
  CURRENT_LANE     STRING,
  PD_HGB           FLOAT,
  ACTION           STRING,
  BASIS            STRING,      -- current rule / model fast-track / review
  MODEL_VERSION    STRING,
  OVERRIDDEN       BOOLEAN,
  OVERRIDE_REASON  STRING
);

-- Optional: refresh a snapshot table monthly with a task.
-- CREATE OR REPLACE TASK REFRESH_CALIBRATION_SNAPSHOT
--   WAREHOUSE = DASHBOARD_WH
--   SCHEDULE = 'USING CRON 0 6 1 * * America/New_York'
-- AS
--   CREATE OR REPLACE TABLE CALIBRATION_SNAPSHOT AS SELECT * FROM V_MONTHLY_CALIBRATION;
-- ALTER TASK REFRESH_CALIBRATION_SNAPSHOT RESUME;