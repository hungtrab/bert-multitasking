# Phân chia công việc — Multitask BERT v2 (5 người)

> Mục tiêu: 1 codebase sạch, mỗi người có ranh giới rõ ràng, ai sửa file nào cũng không đụng người khác.

## Sở hữu module

| # | Người | Module | Files chính | Acceptance |
|---|---|---|---|---|
| 1 | **Vũ Thường Tín** | Data pipeline | `src/multitask_bert/data/{datasets,loaders}.py`, `tests/test_data*.py` | DataLoader cho 3 task chạy được; pad/collate đúng dtype; có test load CSV thật từ v1 |
| 2 | **Chu Anh Đức** | Encoder + tokenizer | `src/multitask_bert/models/bert_encoder.py`, swap-in option cho custom BERT của v1 | `BertEncoder(token_ids, mask)` trả về `(B, hidden)`; có script benchmark forward speed |
| 3 | **Trần Quang Hưng** | Training loops + SMART | `src/multitask_bert/training/{trainer,round_robin,interleaved,optim}.py`, `src/multitask_bert/losses/smart.py` | round-robin và interleaved chạy 1 epoch trên CPU không lỗi; SMART không hủy training (loss không nan) |
| 4 | **Nguyễn Xuân Khải** | Heads + relational layer | `src/multitask_bert/models/{heads,relational,multitask}.py`, `tests/test_relational.py` | `MultitaskBERT` chuyển toggle `use_relational_layer` được; cosine recoverable test pass |
| 5 | **Đỗ Đăng Vũ** | Eval + scripts + CI + report | `src/multitask_bert/evaluation/{metrics,evaluator}.py`, `scripts/{train,evaluate,predict}.py`, `Makefile`, GitHub Actions, slide+report | `make eval` in JSON metrics; CSV output đúng format chấm |

## Quy tắc làm việc

* **1 PR / 1 module**. Reviewer là người sở hữu module liên quan gần nhất (vd PR cho `training/` thì người Eval review).
* **Test trước khi merge**: mỗi PR phải `make test` xanh.
* **Config-driven**: không hardcode hyperparams trong code. Mọi thứ đi qua YAML.
* **Không động vào `weights/` của v1**. Refactor logic only.

## Mốc thời gian gợi ý (4 tuần)

| Tuần | Việc |
|---|---|
| 1 | Mỗi người scaffold module của mình; `make test` chạy được; data pipeline đọc được CSV từ v1 |
| 2 | Round-robin baseline reproducible (không cần đạt số trong paper, chỉ cần loss giảm); Eval báo cáo đúng metric |
| 3 | Interleaved + Rich Relational + SMART hoàn chỉnh; ablation script |
| 4 | Viết báo cáo, slide; kiểm tra reproducibility (3 seed → trung bình) |

## Báo cáo (NLP_Report v2 — 8 trang)

| Mục | Người |
|---|---|
| 1. Introduction & motivation | cả nhóm |
| 2. Architecture (encoder, heads, relational layer) | Khải |
| 3. Multitask training (round-robin vs interleaved) | Hưng |
| 4. SMART regularisation | Hưng |
| 5. Data pipeline & preprocessing | Tín |
| 6. Experiments & results | Vũ |
| 7. Ablations | Đức |
| 8. Conclusion | cả nhóm |
