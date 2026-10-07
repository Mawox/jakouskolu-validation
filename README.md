# LLM scores of school inspection reports: registered validation against a survey

Code for a Registered Report that tests whether scores a large language model gives to the reports of the Czech School
Inspectorate (ČŠI) agree with what pupils, parents and teachers of the same school say in the Mapa školy survey.
The scores are published on [jakouskolu.cz](https://jakouskolu.cz).

Status: the Stage 1 protocol is being prepared for PCI Registered Reports. No score of the registered configuration
exists yet for a surveyed school, and no association between scores and survey data has been computed. Until the
registered scoring run, the analysis code runs on synthetic scores only.

## What is here

| File | Role |
|---|---|
| `rubric_v2.py`, `data/rubric_v2_anchors.json` | The rubric: construct, four dimensions, scale, evidence rules, the prompt and its JSON schema, with the anchor excerpts the prompt contains. The module refuses to load under any prompt hash other than the locked one. |
| `scoring.py`, `scoring_request.py` | The scoring layer (OpenRouter, synchronous and batch requests, a local cache, verbatim matching of quotes) and the frozen request: model, reasoning effort, number of attempts. |
| `mapa_utils.py` | Survey loaders, the item codebook and the construction of scale scores. |
| `_mapa_linkage_audit.py` | Linkage of survey waves to inspections from dates alone, with the counts the protocol reports. |
| `paper1_analysis.py`, `test_paper1_analysis.py` | The registered primary analysis: linked sample, criterion scales, the score of an analysis from three runs, the test of H1 with Holm adjustment, the gate, and the descriptive pattern of correlations. It refuses a scores file that is neither marked synthetic nor registered by hash. |
| `_reliability_sample.py`, `data/reliability_sample.csv`, `data/development_report_ids.csv` | The seeded draw of the reliability sample and the list of reports scored during development, which are not eligible for it. |

## Running it

```bash
uv run pytest                                             # tests, synthetic scores only
uv run python paper1_analysis.py --synthetic s.parquet    # a synthetic scores file (needs the input files below)
uv run python paper1_analysis.py s.parquet                # the primary analysis on it
```

The tests that need the input files are skipped when the files are absent.

## What is not here

The input files are not in this repository: the index of inspection analyses and the report texts (public documents of
ČŠI, collected for jakouskolu.cz), the school registry, and the Mapa školy survey exports, which belong to the schools
and to Scio. The exact analysis file is available to reviewers and, under a data-use agreement that forbids
re-identification of schools, to researchers: jon.sotola@scio.cz.

`python scoring_request.py` rewrites request fixtures of the website's deployment and needs files that are not part of
this package; the module itself is complete.

## Disclosure

Scio owns both the instrument and the survey, and the author is a co-owner of Scio. The code was written with AI
coding assistants and checked by the author.

## Licence

MIT, see `LICENSE`.
