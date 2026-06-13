# Roles and Real Contributions

This file records the work that was actually completed and used in the current project deliverables. It does not claim that every planned item in `docs/TASKS.md` was fully completed.

Report and slides are considered shared group work. Each role below lists the concrete project work that feeds into the report, slides, experiments, or demos.

## Role #1: Data Pipeline

Owner in plan: Vu Thuong Tin

### Actually completed

- Implemented dataset abstractions for the three benchmark tasks:
  - SST-5 single-sentence sentiment classification.
  - Quora Question Pairs paraphrase detection.
  - STS-B semantic textual similarity.
- Implemented batch containers for single-sentence and sentence-pair tasks.
- Implemented CSV loading, text normalization, tokenization, padding, and collate functions.
- Implemented multitask dataloader construction for train/dev/test splits.
- Added support for optional test files when available.
- Prepared a small English HUST handbook subset for the Smart Student Support demo.

### Files connected to this role

- `src/multitask_bert/data/datasets.py`
- `src/multitask_bert/data/loaders.py`
- `src/multitask_bert/data/__init__.py`
- `cea-2-1-1-sv-can-biet-full-version.410094.19628.pdf`
- HUST handbook subset data used by `scripts/demo_student_support.py`

### Brought into report/slides

- Dataset description: SST, Quora, STS-B.
- Explanation of task formats and metrics.
- Smart Student Support demo data source.
- Error Analysis placeholder for STS examples.

### Remaining manual work

- Fill the report Error Analysis section with 2-3 real STS wrong-prediction examples.

## Role #2: Self-Implemented BERT Encoder

Owner in plan: Chu Anh Duc

### Actually completed

- Implemented a custom BERT encoder instead of using `transformers.AutoModel` in the main forward path.
- Implemented:
  - token embeddings,
  - position embeddings,
  - segment embeddings,
  - multi-head self-attention,
  - feed-forward layers,
  - residual connections,
  - layer normalization,
  - dropout,
  - pooler output.
- Added a HuggingFace backend only as a reference/sanity-check backend.
- Added loading of HuggingFace-compatible pretrained weights into the custom encoder.
- Verified numerical parity against HuggingFace:
  - max `last_hidden_state` difference: `4.29e-6`,
  - max `pooler_output` difference: `9.54e-7`.

### Files connected to this role

- `src/multitask_bert/models/bert_encoder.py`
- `src/multitask_bert/models/__init__.py`
- `tests/test_bert_encoder.py`
- `configs/report_best_hf_10e.yaml`

### Brought into report/slides

- Main selling point: the BERT encoder was re-coded by the team.
- BERT architecture section.
- HuggingFace parity check.
- Claim that self-implemented score `0.759` nearly matches HF baseline `0.761`.

### Remaining manual work

- Replace the BERT architecture diagram placeholder in the slides with an actual diagram.

## Role #3: Training and Optimization

Owner in plan: Tran Quang Hung

### Actually completed

- Implemented shared training infrastructure.
- Implemented task-specific training losses:
  - SST cross-entropy,
  - Quora binary cross-entropy,
  - STS mean squared error.
- Implemented round-robin and interleaved multitask training loops.
- Implemented checkpoint saving based on best aggregate score.
- Implemented training history logging.
- Implemented AdamW optimizer setup with no-decay parameter groups.
- Implemented learning-rate scheduling:
  - constant,
  - linear warmup/decay,
  - cosine warmup/decay,
  - constant with warmup.
- Implemented Muon + auxiliary Adam optimizer experiment path.
- Ran/report-prepared optimizer and scheduler ablations:
  - AdamW constant LR,
  - AdamW linear warmup/decay,
  - AdamW cosine warmup/decay,
  - Muon high-LR run,
  - Muon fine-tuning-safe run.

### Files connected to this role

- `src/multitask_bert/training/trainer.py`
- `src/multitask_bert/training/round_robin.py`
- `src/multitask_bert/training/interleaved.py`
- `src/multitask_bert/training/optim.py`
- `src/multitask_bert/training/steps.py`
- `scripts/train.py`
- `queue_next_experiments.sh`
- `queue_scheduler_experiments.sh`
- `queue_replicate_hf_self.sh`
- `runs_compare/`
- `configs/report_best_self_impl_20e_linear_warmup_lr2e5.yaml`
- Muon and scheduler config files under `configs/`

### Brought into report/slides

- Training pipeline.
- Interleaved multitask learning.
- Best training recipe:
  - AdamW,
  - learning rate `2e-5`,
  - weight decay `0.01`,
  - 20 epochs,
  - linear warmup/decay.
- AdamW vs Muon analysis.
- Final conclusion: AdamW is clearly better; Muon damages pretrained weights in this setup.

### Remaining manual work

- Replace the training pipeline diagram placeholder in the slides with an actual diagram.

## Role #4: Task Heads and Relational Layer

Owner in plan: Nguyen Xuan Khai

### Actually completed

- Implemented task-specific model heads:
  - sentiment head for SST,
  - paraphrase head for Quora,
  - STS cosine/regression head.
- Implemented rich relational architecture for sentence-pair tasks.
- Implemented shared Para/STS relational layer using sentence-pair interaction features.
- Integrated the heads and relational layer into `MultitaskBERT`.
- Added prediction methods used by training, evaluation, and demos:
  - `predict_sentiment`,
  - `predict_paraphrase`,
  - `predict_similarity`.

### Files connected to this role

- `src/multitask_bert/models/heads.py`
- `src/multitask_bert/models/relational.py`
- `src/multitask_bert/models/multitask.py`
- `tests/test_relational.py`

### Brought into report/slides

- Multitask architecture.
- Task-specific heads.
- Rich relational layer for Para/STS.
- Explanation of sentence-pair interaction:
  - pooled vectors,
  - element-wise interaction,
  - shared Para/STS features.

### Remaining manual work

- If there is time, add a small visual diagram of the rich relational layer to the slide deck.

## Role #5: Evaluation, Scripts, Demos, and Deliverables

Owner in plan: Do Dang Vu

### Actually completed

- Implemented evaluation metrics:
  - SST accuracy,
  - SST F1,
  - paraphrase accuracy,
  - STS Pearson correlation,
  - aggregate score.
- Implemented evaluator for dev/test loaders.
- Implemented train/evaluate/predict script entry points.
- Implemented Gradio checkpoint inference demo:
  - SST mode,
  - paraphrase mode,
  - STS mode.
- Implemented Smart Student Support demo:
  - sentiment/urgency interpretation,
  - duplicate FAQ check,
  - handbook entry ranking.
- Created final report deliverables:
  - `deliverables/report/report.tex`,
  - `deliverables/report/report.pdf`.
- Created final slide deliverables using the HUST beamer theme:
  - `deliverables/slides/presentation.tex`,
  - `deliverables/slides/presentation.pdf`.
- Added `progress.md` to help teammates continue editing.

### Files connected to this role

- `src/multitask_bert/evaluation/metrics.py`
- `src/multitask_bert/evaluation/evaluator.py`
- `scripts/evaluate.py`
- `scripts/predict.py`
- `scripts/demo_gradio.py`
- `scripts/demo_student_support.py`
- `Makefile`
- `deliverables/report/`
- `deliverables/slides/`
- `progress.md`

### Brought into report/slides

- Result tables.
- Aggregate score formula.
- AdamW/Muon comparison figure.
- Demo sections.
- Demo command slides.
- Placeholder screenshots for both demos.

### Remaining manual work

- Run both demos visually and replace screenshot placeholders.
- Rebuild final PDF after placeholders are replaced.

## Shared Group Deliverables

These are not assigned to only one role:

- Final report writing.
- Final slide editing.
- Demo screenshots.
- Presentation script.
- Final PDF build.

All five members should review the final report and slides, but each person should mainly explain the part connected to their role.

