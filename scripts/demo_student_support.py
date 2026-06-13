"""Smart Student Support Assistant demo.

This is a product-style demo built on the same single multitask checkpoint.
It maps the three benchmark tasks to a concrete use case:

* SST: detect student sentiment / urgency.
* Para: detect duplicate FAQ questions.
* STS: rank semantically related documents or support posts.

Usage:
    PYTHONPATH=src python -m scripts.demo_student_support \
      --config configs/report_best_self_impl_20e_linear_warmup_lr2e5.yaml \
      --checkpoint runs/report_best_self_impl_20e_linear_warmup_lr2e5/best.pt

Optional CSV upload columns:
    title, question, answer, content

Only ``title`` plus one of ``question`` / ``content`` / ``answer`` is needed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from scripts.demo_gradio import DemoModel


DEFAULT_KB = [
    {
        "title": "Final report submission",
        "question": "How do I submit the final project report?",
        "answer": "Upload the final report PDF to the LMS submission page before the deadline.",
        "content": "Students should submit the final project report as a PDF file on the LMS. "
        "The submission page is under Course Project > Final Report.",
    },
    {
        "title": "BERT fine-tuning learning rate",
        "question": "Why does my BERT validation score drop during fine-tuning?",
        "answer": "Try a smaller learning rate, warmup, linear decay, and keep the best checkpoint.",
        "content": "For BERT fine-tuning, validation performance can drop when the learning rate is too "
        "high or training runs too long. Linear warmup and decay are common choices.",
    },
    {
        "title": "Presentation requirement",
        "question": "What should we include in the final presentation?",
        "answer": "Explain the problem, datasets, model architecture, training strategy, results, and demo.",
        "content": "The final presentation should cover the NLP tasks, the multitask BERT architecture, "
        "the training pipeline, experiment results, limitations, and a short live demo.",
    },
    {
        "title": "Team contribution split",
        "question": "How should a five-member NLP project team divide the work?",
        "answer": "Split into data, encoder, task heads, training, and evaluation/reporting.",
        "content": "A clean five-person split is data preprocessing, self-implemented BERT encoder, "
        "multitask heads, training and optimization, and evaluation or report writing.",
    },
    {
        "title": "Dataset format",
        "question": "What format do SST, Quora, and STS datasets use?",
        "answer": "SST uses one sentence and sentiment label. Quora and STS use sentence pairs.",
        "content": "SST is a single-sentence sentiment classification task. Quora paraphrase and STS "
        "are sentence-pair tasks for duplicate detection and semantic similarity scoring.",
    },
]


@dataclass
class KBDocument:
    title: str
    question: str
    answer: str
    content: str

    @property
    def search_text(self) -> str:
        return " ".join(x for x in [self.title, self.question, self.content] if x).strip()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    return parser.parse_args(argv)


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _row_to_doc(row: dict[str, Any], idx: int) -> KBDocument:
    title = _clean_cell(row.get("title")) or f"Document {idx + 1}"
    question = _clean_cell(row.get("question"))
    answer = _clean_cell(row.get("answer"))
    content = _clean_cell(row.get("content")) or _clean_cell(row.get("text"))

    if not question:
        question = title
    if not content:
        content = answer or question
    if not answer:
        answer = content

    return KBDocument(title=title, question=question, answer=answer, content=content)


def load_kb(csv_file) -> list[KBDocument]:
    if csv_file is None:
        return [_row_to_doc(row, idx) for idx, row in enumerate(DEFAULT_KB)]

    path = csv_file.name if hasattr(csv_file, "name") else str(csv_file)
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    docs = [_row_to_doc(row, idx) for idx, row in enumerate(df.to_dict(orient="records"))]
    return [doc for doc in docs if doc.search_text]


class StudentSupportDemo:
    def __init__(self, model: DemoModel):
        self.model = model

    @torch.no_grad()
    def _duplicate_prob(self, query: str, candidate: str) -> float:
        ids1, mask1 = self.model._encode(query)
        ids2, mask2 = self.model._encode(candidate)
        logit = self.model.model.predict_paraphrase(ids1, mask1, ids2, mask2)
        return float(logit.sigmoid().item())

    @torch.no_grad()
    def _similarity_score(self, query: str, candidate: str) -> float:
        ids1, mask1 = self.model._encode(query)
        ids2, mask2 = self.model._encode(candidate)
        raw_score = float(self.model.model.predict_similarity(ids1, mask1, ids2, mask2).item())
        score_0_5 = raw_score * float(self.model.cfg.losses.get("sts_label_scale", 5.0))
        return max(0.0, min(5.0, score_0_5))

    def analyze(self, query: str, csv_file=None, top_k: int = 3):
        query = query.strip()
        if not query:
            empty = pd.DataFrame(columns=["title", "score", "question", "answer"])
            return "Please enter a student question.", {}, "No query.", empty, empty, ""

        docs = load_kb(csv_file)
        if not docs:
            empty = pd.DataFrame(columns=["title", "score", "question", "answer"])
            return "No knowledge base rows found.", {}, "No documents.", empty, empty, ""

        sentiment_label, sentiment_probs = self.model.sentiment(query)
        priority = self._priority_from_sentiment(sentiment_label)

        duplicate_rows = []
        related_rows = []
        for doc in docs:
            duplicate_rows.append(
                {
                    "title": doc.title,
                    "duplicate_probability": self._duplicate_prob(query, doc.question),
                    "question": doc.question,
                    "answer": doc.answer,
                }
            )
            related_rows.append(
                {
                    "title": doc.title,
                    "similarity_0_5": self._similarity_score(query, doc.search_text),
                    "question": doc.question,
                    "answer": doc.answer,
                }
            )

        duplicate_df = (
            pd.DataFrame(duplicate_rows)
            .sort_values("duplicate_probability", ascending=False)
            .head(int(top_k))
            .reset_index(drop=True)
        )
        related_df = (
            pd.DataFrame(related_rows)
            .sort_values("similarity_0_5", ascending=False)
            .head(int(top_k))
            .reset_index(drop=True)
        )

        recommendation = self._recommendation(query, priority, duplicate_df, related_df)
        return sentiment_label, sentiment_probs, priority, duplicate_df, related_df, recommendation

    @staticmethod
    def _priority_from_sentiment(sentiment_label: str) -> str:
        lowered = sentiment_label.lower()
        if "very negative" in lowered or lowered.startswith("1 - negative"):
            return "High priority: student may be confused or frustrated."
        if "negative" in lowered:
            return "Medium priority: negative sentiment detected."
        return "Normal priority."

    @staticmethod
    def _recommendation(query: str, priority: str, duplicate_df: pd.DataFrame, related_df: pd.DataFrame) -> str:
        best_dup = duplicate_df.iloc[0] if len(duplicate_df) else None
        best_rel = related_df.iloc[0] if len(related_df) else None

        lines = [
            "## Suggested support action",
            f"**Student query:** {query}",
            f"**Priority:** {priority}",
        ]
        if best_dup is not None and float(best_dup["duplicate_probability"]) >= 0.5:
            lines.extend(
                [
                    "",
                    "**Likely duplicate FAQ found.**",
                    f"- Matched item: {best_dup['title']}",
                    f"- Existing answer: {best_dup['answer']}",
                ]
            )
        elif best_rel is not None:
            lines.extend(
                [
                    "",
                    "**No strong duplicate detected, but related material was found.**",
                    f"- Suggested material: {best_rel['title']}",
                    f"- Suggested answer: {best_rel['answer']}",
                ]
            )
        return "\n".join(lines)


def build_app(demo: StudentSupportDemo):
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Gradio is not installed. Run: pip install gradio") from exc

    with gr.Blocks(title="Smart Student Support Assistant") as app:
        gr.Markdown(
            """
            # Smart Student Support Assistant

            Product-style demo using one multitask BERT checkpoint:

            - **SST** detects student sentiment / urgency.
            - **Para** checks whether the question duplicates an existing FAQ.
            - **STS** ranks related course documents or support posts.

            You can use the built-in sample knowledge base, or upload a CSV with columns
            `title`, `question`, `answer`, `content`.
            """
        )

        with gr.Row():
            query = gr.Textbox(
                label="Student question",
                lines=4,
                value="I am really confused about BERT fine-tuning. My validation score keeps dropping, what should I do?",
            )
            with gr.Column():
                kb_file = gr.File(label="Optional knowledge base CSV", file_types=[".csv"])
                top_k = gr.Slider(label="Top K", minimum=1, maximum=5, step=1, value=3)
                analyze_button = gr.Button("Analyze student query", variant="primary")

        with gr.Row():
            sentiment = gr.Textbox(label="SST sentiment")
            priority = gr.Textbox(label="Support priority")

        sentiment_probs = gr.Label(label="Sentiment probabilities")
        duplicate_table = gr.Dataframe(label="Para: likely duplicate FAQs")
        related_table = gr.Dataframe(label="STS: related documents")
        recommendation = gr.Markdown(label="Suggested action")

        analyze_button.click(
            demo.analyze,
            inputs=[query, kb_file, top_k],
            outputs=[
                sentiment,
                sentiment_probs,
                priority,
                duplicate_table,
                related_table,
                recommendation,
            ],
        )

    return app


def main(argv=None) -> int:
    args = parse_args(argv)
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    model = DemoModel(args.config, args.checkpoint, args.device)
    demo = StudentSupportDemo(model)
    app = build_app(demo)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
