# Cox Proportional Hazards Model FeatureCloud App

## Description

This FeatureCloud app implements a federated cox proportional hazards model based on the lifelines library. It implements the WebDisco approach:

https://pubmed.ncbi.nlm.nih.gov/26159465/

This app does not work on a single client execution. At least 2 participants are needed.
This app is frontend based. No config-file needed.

## Input

Input can be specified through the app frontend. There is no config file or input data required upfront.

1. File upload
2. File type (CSV or TSV)
3. Define duration and event columns name
4. Define max number of iterations and convergence criterium
5. Use only the intersection of covariates of all clients (makes computation more rubust, as only features are used that exist on all clients)
6. Use a penalized regression
7. If using penalized regression select which one (Lasso, Ridge, Elastic Net)
8. If using penalized regression select penalizer and l1-ratio

## Output

Output can be downloaded through the app frontend.

## Workflows

This is a standalone app. It is not compatible with other apps.

## Config

No config file needed, everything can be adjusted through the app frontend.

## Privacy

- No patient-level data is exchanged (more infos: https://pubmed.ncbi.nlm.nih.gov/26159465/)
- Exchanges:
  - Mean and Variance of each Feature
  - Event times
  - Number of samples
  - Summary statistics
- No additional privacy-enhancing techniques implemented yet
