# Data

`loan_default.csv` contains 148,670 rows and 34 columns. It is the same public data loaded by the original notebook from:

`https://raw.githubusercontent.com/Tantrayoga/loan-default-risk-modeling/refs/heads/main/dataset/Loan_Default.csv`

The repository does not establish the dataset's original creator, precise target definition, collection process, or redistribution license. Verify those details before commercial use or public redistribution. If you decide not to commit the data, delete the CSV; the notebook retains a public-source fallback.

Key modeling fields:

- `Status`: binary response
- `ID`: row identifier; excluded from modeling
- `year`: constant field; excluded from modeling
- `Interest_rate_spread`, `rate_of_interest`, `Upfront_charges`: excluded because outcome-dependent missingness creates target leakage
- `credit_type`: retained in the original notebook for traceability, but flagged as a likely proxy and excluded by the script's strict mode

