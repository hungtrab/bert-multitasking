# NLP Project Progress

## Current status

The project is ready for teammates to review and edit the report/slide deck.

Main deliverables:

- Report source: `deliverables/report/report.tex`
- Report PDF: `deliverables/report/report.pdf`
- Slide source: `deliverables/slides/presentation.tex`
- Slide PDF: `deliverables/slides/presentation.pdf`

## What has been implemented

- Self-implemented BERT encoder forward pass.
- Multitask model for:
  - SST sentiment classification.
  - Quora paraphrase detection.
  - STS semantic textual similarity.
- Rich relational layer for Para/STS sentence-pair tasks.
- Interleaved multitask training.
- AdamW, LR scheduler, and Muon optimizer experiments.
- Evaluation scripts and aggregate score calculation.
- Gradio checkpoint inference demo.
- Smart Student Support demo using the HUST handbook subset.
- HUST-theme LaTeX slide deck.
- Academic-style LaTeX report.

## Key result to present

Best self-implemented checkpoint:

- Config: `configs/report_best_self_impl_20e_linear_warmup_lr2e5.yaml`
- Checkpoint: `runs/report_best_self_impl_20e_linear_warmup_lr2e5/best.pt`
- SST accuracy: `0.520`
- Paraphrase accuracy: `0.864`
- STS Pearson: `0.785`
- Aggregate score: `0.759`

Sanity baseline:

- HuggingFace-backed baseline aggregate score: `0.761`
- Self-implemented BERT hidden-state max difference vs HuggingFace: `4.29e-6`

This is the main point to emphasize: the team re-coded the BERT encoder and still reached almost the same score as the library-backed baseline.

## Notes for slide/report editing

- Slide deck is in English.
- Report is in English.
- Both contain placeholders for demo screenshots.
- Report contains an Error Analysis placeholder for 2-3 STS failure examples.
- Slides include diagram placeholders for:
  - BERT architecture.
  - Training pipeline.
- Replace placeholders with real screenshots/diagrams before final submission if time allows.

## Demo commands

Checkpoint inference demo:

```bash
PYTHONPATH=src python -m scripts.demo_gradio \
  --config configs/report_best_self_impl_20e_linear_warmup_lr2e5.yaml \
  --checkpoint runs/report_best_self_impl_20e_linear_warmup_lr2e5/best.pt
```

Smart Student Support demo:

```bash
PYTHONPATH=src python -m scripts.demo_student_support \
  --config configs/report_best_self_impl_20e_linear_warmup_lr2e5.yaml \
  --checkpoint runs/report_best_self_impl_20e_linear_warmup_lr2e5/best.pt
```

## Build commands

Build report:

```bash
cd deliverables/report
latexmk -pdf report.tex
```

Build slides:

```bash
cd deliverables/slides
latexmk -pdf presentation.tex
```

