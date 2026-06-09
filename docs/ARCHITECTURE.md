# Architecture

```
                 ┌─────────────────────────────────┐
                 │   YAML config (configs/*.yaml)  │
                 └────────────────┬────────────────┘
                                  │ load_config()
              ┌───────────────────▼────────────────────┐
              │  scripts/train.py — orchestrator       │
              └─────────┬───────────┬──────────┬───────┘
                        │           │          │
                build_loaders   MultitaskBERT  Trainer (round_robin | interleaved)
                        │           │          │
                ┌───────▼───┐  ┌────▼────┐  ┌──▼──────────────┐
                │  data/    │  │ models/ │  │ training/       │
                │ datasets  │  │ encoder │  │ trainer (base)  │
                │ loaders   │  │ heads   │  │ round_robin     │
                └───────────┘  │ relat.  │  │ interleaved     │
                               │ multi   │  │ optim           │
                               └─────────┘  └──┬──────────────┘
                                               │ uses
                                       ┌───────▼────────┐
                                       │ losses/        │
                                       │ smart          │
                                       └────────────────┘
                                               ▲
                                       ┌───────┴────────┐
                                       │ evaluation/    │
                                       │ metrics        │
                                       │ evaluator      │
                                       └────────────────┘
```

## Forward paths

### Plain (default)

```
SST  : ids -> BertEncoder -> [CLS] -> SentimentHead -> 5 logits
Para : ids1, ids2 -> [CLS]_u, [CLS]_v -> ParaphraseHead([h_u; h_v; |h_u - h_v|]) -> 1 logit
STS  : ids1, ids2 -> [CLS]_u, [CLS]_v -> STSCosineHead = (cos + 1)/2 * 5
```

### Rich relational (`use_relational_layer: true`)

```
Para+STS share the layer:

  shared = LeakyReLU(W * (h_u ⊙ h_v) / (||h_u|| · ||h_v||))

  Para  -> Linear -> 1 logit
  STS   -> Linear -> sigmoid * 5
```

Notice: in the relational mode we **cannot** use cosine for STS anymore
(it would collapse the relational layer). We use a sigmoid-bounded scalar
output instead, which retains gradient signal across the full `[0, 5]`
range.

## Encoder Implementation

`models/bert_encoder.py` now contains a self-implemented BERT encoder:

```text
token ids
  -> word + position + token-type embeddings
  -> multi-head self-attention blocks
  -> feed-forward blocks
  -> pooled [CLS] representation
```

The code does not call `transformers.AutoModel` for the forward pass. It only
uses the pretrained `bert-base-uncased` config/tokenizer/weight-file format so
the local implementation can start from standard BERT weights.

The public contract remains:

```python
EncoderOutput(cls, sequence)
```

This keeps the rest of the multitask model independent from the encoder
implementation.

## SMART (regularisation)

Implementation in `losses/smart.py`. The key contract is:

```python
loss = task_loss + smart(model, encoder, input_ids, attention_mask,
                         forward_fn, clean_output, divergence)
```

`forward_fn` is a closure that runs the model from word embeddings (so we
can perturb them by `δ`). It's task-specific — see `MultitaskTrainer.sst_loss`
for an example. The implementation toggles `model.eval()` inside the
adversarial pass to silence dropout (a known landmine from v1).

## What is *not* in v2 (yet)

* Multi-GPU / DDP.
* TensorBoard / Weights & Biases — easy to add in `MultitaskTrainer`.
* Quantisation / TorchScript export.

These are left as homework for the keen.
