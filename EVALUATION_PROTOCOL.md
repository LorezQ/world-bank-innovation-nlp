# Validation and Evaluation Protocol

## Validation sample

The validation sample contains 246 projects drawn using stratified random sampling from the World Bank innovation-classification tiers.

Sampling frame:

- Low: 6,600 projects; 95 sampled
- Moderate: 774 projects; 86 sampled
- High: 200 projects; 65 sampled
- Total portfolio: 7,574 projects
- Total validation sample: 246 projects

Random seed: 20261005.

The stratified design intentionally oversamples Moderate and High projects relative to their prevalence in the portfolio.

## Human reference labels

Projects were reviewed semantically and assigned one of three labels:

- Yes: substantive evidence of innovation
- No: insufficient substantive evidence of innovation
- Borderline: evidence is ambiguous or insufficient for a confident binary judgment

Final labels:

- Yes: 120
- No: 92
- Borderline: 34

The review file was initially AI-preannotated and subsequently reviewed and adjudicated by the human reviewer. These labels are therefore described as human-adjudicated reference labels rather than objective ground truth.

## Primary evaluation

The primary binary evaluation excludes Borderline observations.

For the World Bank keyword baseline:

- Low = predicted negative
- Moderate or High = predicted positive

The same human reference labels and evaluation rules will be used for all alternative classifiers.

## Sampling weights

Portfolio-level performance estimates use inverse sampling-probability weights calculated within each original innovation tier:

- Low: 6600 / 95 = 69.47
- Moderate: 774 / 86 = 9.00
- High: 200 / 65 = 3.08

Unweighted validation-sample results may be reported descriptively but will not be interpreted as portfolio-level performance estimates.

## Metrics

Primary comparison metrics:

- Precision
- Recall
- F1 score

False-positive and false-negative cases will also be examined qualitatively.

## Borderline sensitivity analysis

Two sensitivity specifications will accompany the primary analysis:

1. Borderline treated as No
2. Borderline treated as Yes

These specifications were defined before evaluating the semantic embedding model.

## Uncertainty

Sampling uncertainty for weighted performance metrics will be assessed using stratified bootstrap resampling within the original Low, Moderate, and High sampling strata.

## Baseline results

Primary specification (Borderline excluded):

- Precision: 0.745
- Recall: 0.370
- F1: 0.495

Sensitivity:

- Borderline -> No: Precision 0.630, Recall 0.370, F1 0.466
- Borderline -> Yes: Precision 0.784, Recall 0.289, F1 0.423

These results are design-weighted estimates based on the validation sample and should be interpreted with sampling uncertainty.
