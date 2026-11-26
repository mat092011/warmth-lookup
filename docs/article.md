# Two rejected commands: what a warmth-steering experiment actually measured

On a card missing a routing tag, the model could guess a bay or ask for the tag. One model response was `LOOKUP`. Under a different intervention the response was `LOOKUP hidden`. A reader might interpret both as requests, but the interface executed only the first. That distinction accounts for the entire difference observed between the main conditions.

## Method

Sixty synthetic cards used three formats, balanced by correct bay. Each was presented with the selected tag both visible and hidden. Four conditions shared the same cards: neutral baseline, warm prompt, contrastive activation addition (CAA) derived from warm versus neutral responses, and a norm-matched random vector. Each presentation was a new dialogue. On a valid lookup, the interface returned the tag before the model's final answer. The initial gate failed; after an interface amendment, both revised gate checks passed on fresh cards. The main evaluation used frozen inputs.

## What happened

| Condition | Executed lookups, hidden | Executed lookups, visible | Correct, hidden | Malformed final output, hidden |
|---|---:|---:|---:|---:|
| Neutral | 40/60 | 0/60 | 46/60 | 9/60 |
| Warm prompt | 39/60 | 0/60 | 44/60 | 10/60 |
| CAA | 38/60 | 0/60 | 44/60 | 11/60 |
| Random vector | 18/60 | 0/60 | 23/60 | 31/60 |

The primary contrast was CAA minus neutral in **lookup selectivity** (hidden minus visible lookup rate). It was **−3.3 percentage points** (95% stratified paired card-bootstrap interval −8.3 to 0.0). The unit was 60 cards, not 480 independent samples. Every executed hidden-tag lookup was followed by a correct answer. All four conditions answered every visible card correctly.

## What the result does not say

There was no supported increase in executed lookups. On hidden cards 4455 and 4815 the exact-match parser rejected the extra word; the other 118 CAA/baseline pairs agreed on execution. This cannot establish a change in intent or in perceived warmth. The latter was not rated by humans. There was only one model, one random vector and a small set of templates. A useful next test, **not performed here**, would freeze a second parser in advance and compare the effect on literal command compliance versus semantically recognizable requests.

[Full paper](../output/pdf/warmth-lookup.pdf) · [Changed-card records and uncertainty](../v2/full/summary.json) · [Reproduce or audit this result](../README.md)
