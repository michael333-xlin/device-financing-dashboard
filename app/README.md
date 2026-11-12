# Device Financing Decision Dashboard (Streamlit in Snowflake)

An interactive decision-support prototype for a company's in-store device installment financing program. The dashboard puts three default-risk models (credit-only, internal-only, and HistGradientBoosting) directly into the workflow of the people who make and oversee financing decisions, from sales representatives reviewing a single application to managers evaluating the portfolio-level impact of a policy change.

## Business Context

The company finances the devices its service customers use, in a model similar to a home or auto loan: the company buys the device from the manufacturer, the customer pays an initial down payment, and the remaining balance is collected in monthly installments until the device is paid off.

When a customer applies to finance a device in store, the company's current policy sorts each application into one of three lanes based on a proprietary internal risk score: **Approve**, **Deny**, or **Conditional**. Conditional applicants receive a counteroffer, such as a cheaper device or a larger down payment, and a large share of them walk away during that negotiation, which means lost revenue from customers who would often have repaid.

This project asks a focused question: *can a default model safely fast-track the lower-risk part of the Conditional lane, and what is that worth?* The dashboard is built to answer that question for three audiences at once.

## Data

**Source.** The models are trained on a **simulated dataset** generated to mirror the business of a company offering device installment plans, provided as part of an academic case study. No real customer data is used, and the business practices described here are hypothetical.

**Size and scope.** The dataset contains **300,000 historical applications** with 18 fields each, covering application dates from January 1, 2023 to December 31, 2024. Every record is a 24-month financing plan, and all applications were handled under the company's existing ("status quo") risk policy. The data includes both rejected and approved applications, including approvals that resulted from negotiation. Of the 300,000 applications, 265,060 were funded and 34,940 were not.

**Policy regimes.** The internal risk policy was adjusted twice during the sample period: once in 2023 during a promotional event, and once in 2024 in response to economic shocks and rising defaults, when lending was tightened. The analysis therefore splits the data into three policy regimes, with Regime 3 being the most recent and the one the dashboard's decision thresholds are based on.

**Fields.**

| Column | Type | Description |
|---|---|---|
| `application_date` | Date | Date the application reached its final status |
| `race`, `hisp`, `gender` | Character | Self-reported demographics (including "PreferNot"), collected only for fairness evaluation |
| `age` | Integer | Applicant age in years (18–79) |
| `income` | Integer | Annual income in USD (15,000–200,000) |
| `credit_score` | Integer | FICO credit score (327–850) |
| `existing_customer` | Integer | 1 if the applicant is already a customer of the company, otherwise 0 |
| `device_requested` | Character | Tier (Low, Mid, High) of the device first requested |
| `device_final` | Character | Tier of the device actually sold, which may differ after negotiation |
| `device_price` | Float | Cost of the final device in USD (379–1,219) |
| `down_payment_base` | Float | Down payment first suggested by the company's business rules, as a fraction of device cost |
| `down_payment_final` | Float | Down payment on the final device after any negotiation |
| `internal_risk_score` | Float | The company's proprietary risk score, normalized to zero mean; its formula is not disclosed |
| `initial_decision` | Character | Initial outcome: Approve, Conditional, or Deny |
| `final_status` | Character | Funded_Original, Funded_Revised (successfully negotiated), or Not_Funded |
| `default_flag` | Integer | 1 if the customer stopped paying before the device was paid off |
| `profit` | Float | Estimated profit or loss on the plan in USD (−426.65 to 304.75) |

## Model Training

**Target and training population.** The target is `default_flag`. Because default is only observed for applicants who were actually funded, the models are trained on the 265,060 funded applications.

**Models.** Three models are trained and compared so that the value of a more flexible model can be measured against simple benchmarks:

- **Credit-only**: a cubic spline (5 quantile knots) on credit score, followed by logistic regression.
- **Internal-only**: the same spline + logistic regression on the company's internal risk score, representing the status quo.
- **HistGradientBoosting**: a scikit-learn `HistGradientBoostingClassifier` (learning rate 0.05, up to 200 iterations with early stopping, 15 leaf nodes, minimum 100 samples per leaf, L2 regularization 10). Inputs are credit score, internal risk score, age, income, customer status, requested device tier, and policy regime, with one-hot encoding packaged inside the scikit-learn pipeline so the dashboard applies exactly the same transformation.

The self-reported demographic fields are excluded from model inputs and are used only for the subgroup audit in the dashboard.

**Validation.** Models are evaluated with log loss, since the business decision depends on well-calibrated default probabilities rather than rankings alone; AUC is reported alongside. The funded applications are split 75/25 with stratification on the default flag (`random_state = 42`), and the production models are fit on the 75% training split so the published holdout metrics describe exactly the models the dashboard uses. A separate out-of-time check trains on applications before October 2024 and tests on October–December 2024. Payoff estimates (average gain and loss per plan) are computed from the training split only.

**Pipeline.** `model/train_register.py` runs the full pipeline: train, evaluate, write the results to evidence tables in Snowflake, and register the three models in the Snowflake Model Registry, with a joblib backup saved to a stage.

## Design Approach

**Built around the people who use it.** Each page maps to a specific role and the decision that role actually makes:

| Page | Primary user | Purpose |
|---|---|---|
| Application review | In-store sales representative | Score a single application and see the recommended lane |
| Offer comparison | In-store sales representative | Compare the original offer with a modified offer |
| Portfolio overview | Managers and leadership | Historical KPIs, model evidence, and a value estimate for 1M applications |

**Change as little as possible.** The proposal leaves the existing Approve and Deny thresholds untouched and only applies the model inside the Conditional lane. This keeps the change easy to explain, easy to pilot, and easy to roll back.

**Be honest about what the model can't do.** Applications whose scores fall outside the range the company has historically financed are routed to a supervisor instead of being scored, because predictions there would be extrapolation. The offer comparison page also deliberately avoids claiming that a lower down payment reduces default risk, since the data does not support that causal claim.

**One source of truth for every number.** All figures shown in the dashboard are read from evidence tables written by the training pipeline, so the dashboard, the analysis, and any presentation built on top of it stay consistent.

**Resilient by design.** Model loading falls back from the Snowflake Model Registry to a joblib artifact on a stage, so the app keeps working even if the Registry is unavailable.

## Decision Rules Implemented

1. **Approve and Deny lanes are unchanged.** The current policy is a pair of thresholds on the internal score; in the most recent policy period (Regime 3) they are 0.845 and 1.502, inferred from historical data.
2. **The model is used only within the Conditional lane.** Applicants whose predicted default rate is below the fast-track cap (16% by default) are approved on the original terms; all others continue to negotiation as before.
3. **Some cases go to Review for a supervisor:** invalid inputs, or an internal score or credit score outside the range previously financed in that regime.
4. **Offer comparison makes no causal claim about down payments.** The default rate of a modified offer is assumed equal to the application-stage prediction; any manually entered value is labeled "Entered scenario risk."

## Key Results

All values are read from the Snowflake evidence tables.

| Metric | Value |
|---|---|
| Random holdout log loss (HGB / internal / credit) | 0.272318 / 0.275443 / 0.277364 |
| Out-of-time validation (Oct–Dec 2024) | 0.3344 / 0.3386 / 0.3423 |
| Existing-customer default in final regime: actual / internal predicted / HGB predicted | 11.9% / 8.2% / 11.1% |
| Conditional applicant drop-off after negotiation (Regime 3) | 39.7% |
| Baseline profit for 1M applications | $119.67M |
| Net incremental profit, default settings (cap 16%, acceptance 50%, risk ×1.5) | +$1.13M |
| Net incremental profit, cap 18% | +$3.43M |
| Break-even risk multiplier | approx. ×2.3 |

The HGB log loss differs slightly from the exploratory notebook (0.272329) because of a different scikit-learn version.

## Tech Stack

Snowflake (warehouse, stages, Model Registry), Streamlit in Snowflake, Python, scikit-learn, pandas, Altair, and the Snowflake CLI.

## Project Structure

```
device-financing-dashboard/
├── sql/
│   ├── 01_setup_and_load.sql      Create warehouse, database, stages, and tables; load CSV; reconcile row counts
│   ├── 02_deploy_streamlit.sql    Manual app deployment (without the CLI)
│   └── 03_pilot_monitoring.sql    Proposed calibration monitoring views and decision log table for a pilot
├── model/
│   └── train_register.py          Train → evaluate → write evidence tables → register models
├── app/                           Everything uploaded to Snowflake
│   ├── streamlit_app.py           The three pages
│   ├── data_layer.py              Data access to the Snowflake evidence tables
│   ├── scoring.py                 Model loading: Model Registry → stage artifact
│   ├── finance_rules.py           Decision rules and profit formulas (pure functions, unit-testable)
│   ├── ui.py / config.py          Styling and configuration
│   ├── environment.yml            Streamlit in Snowflake dependencies (check the scikit-learn version)
│   ├── snowflake.yml              Snowflake CLI deployment definition
│   └── .streamlit/config.toml     Theme
└── screenshots/                   Dashboard screenshots
```

## Deploying to Snowflake (about 30 minutes)

**Step 1: Create objects and load data.** Open a worksheet in Snowsight and run the first half of `sql/01_setup_and_load.sql`. Then upload `financing_data.csv` to `RAW_STAGE` via Catalog → Database Explorer → `DEVICE_FINANCING_DB` → `FINANCING` → Stages → `RAW_STAGE` → "+ Files". Run the remaining `COPY INTO` and reconciliation queries; you should see 300,000 / 265,060 / 34,940.

**Step 2: Train and register the models.** Create a new Notebook in Snowsight using `DASHBOARD_WH`, and add `scikit-learn`, `snowflake-ml-python`, `pandas`, and `joblib` under Packages. Paste the contents of `model/train_register.py` into a cell and run it. When it finishes you will have:

- 8 evidence tables, including `MODEL_METRICS` and `APPLICATION_SCORES`
- 3 models in the Model Registry, `DEFAULT_MODEL_HGB`, `DEFAULT_MODEL_INTERNAL`, and `DEFAULT_MODEL_CREDIT`, each at version V1
- A backup artifact at `@MODEL_STAGE/default_models.joblib`

To retrain, change `MODEL_VERSION` to V2 at the top of the file before running it again.

**Step 3: Match the scikit-learn version.** Run `SELECT DISTINCT SKLEARN_VERSION FROM MODEL_INFO;` and put the result on the `scikit-learn=` line of `app/environment.yml`. This is the most common failure point: if the versions don't match, the stage backup model will fail to load.

**Step 4: Deploy the app.** Either:

- With the Snowflake CLI: `cd app && snow streamlit deploy --replace`
- Manually: follow `sql/02_deploy_streamlit.sql` to upload the files in `app/` to `STREAMLIT_STAGE/app/`, then run `CREATE STREAMLIT`

**Step 5: Confirm the model source.** Open the app; the bottom of the sidebar should read `Model · Snowflake Model Registry`. If it reads `Snowflake stage artifact`, the Registry call failed and the app fell back to the stage file. The app still works, but check Registry permissions and confirm that `snowflake-ml-python` is listed in `environment.yml`.
