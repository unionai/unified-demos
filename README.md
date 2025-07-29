# Union AI Demo Workflows

Demo Scripts: https://docs.google.com/document/d/1mXDpGbHVdK-OgVUre56uCfzdwxaXP6Ioka_Zb8qqDeI/edit?tab=t.0

## Setup Env

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create
the virtual environment:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Running the demo

Train a model

```bash
union run --max-parallelism 3 --remote unified_wfs.py unified_demo_wf --inputs-file inputs.yaml
```

Activate model training schedule and deployment pipeline:

```bash
union register unified_register_wfs.py
```

This will:
- Schedule the HPO workflow.
- Activate the triggers for swapping out prod vs. dev models.
