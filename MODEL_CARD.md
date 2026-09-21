# Model card

## Intended use

This model is a portfolio demonstration of a binary customer-churn workflow. It can be used to learn from the code, test the command-line interface, or prototype an internal retention tool.

It is **not** intended to make decisions about real customers. The included model is trained on generated data, not observations from a real business.

## Inputs and output

The model uses account, billing, usage and support features. It returns a probability between 0 and 1 and a label based on the default 0.5 threshold. Customer identifiers are excluded from training.

## Evaluation

The current sample run used a stratified 80/20 train/test split. Model selection and tuning were performed only on the training portion. The saved metrics are in `artifacts/metrics.json`; headline results are reproduced in the README.

## Limitations

- Performance on generated data does not estimate performance on a real customer population.
- The data generator contains deliberately constructed relationships between several features and churn.
- No demographic fairness assessment has been performed.
- A 0.5 threshold is convenient for the demo, but a real threshold should reflect campaign costs and capacity.
- Model quality can decline if customer behaviour or product offerings change.

## Monitoring for a real deployment

With real data, I would monitor input drift, missing-value rates, prediction distribution, calibration, precision and recall after outcomes become available. I would also compare performance across appropriate customer groups and schedule retraining only when supported by those checks.
