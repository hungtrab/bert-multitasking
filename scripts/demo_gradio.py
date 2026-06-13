"""Small Gradio demo for the multitask BERT checkpoint.

Usage:
    PYTHONPATH=src python -m scripts.demo_gradio \
      --config configs/report_best_self_impl_20e_linear_warmup_lr2e5.yaml \
      --checkpoint runs/report_best_self_impl_20e_linear_warmup_lr2e5/best.pt

Install Gradio if needed:
    pip install gradio
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from multitask_bert.models import MultitaskBERT, build_tokenizer
from multitask_bert.utils import load_config


SENTIMENT_LABELS = {
    0: "very negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "very positive",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args(argv)


class DemoModel:
    def __init__(self, config_path: str, checkpoint_path: str, device: str | None = None):
        self.cfg = load_config(config_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = build_tokenizer(self.cfg.model.encoder)

        self.model = MultitaskBERT(self.cfg).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.max_seq_len = int(self.cfg.data.max_seq_len)

    def _encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    @torch.no_grad()
    def sentiment(self, sentence: str):
        sentence = sentence.strip()
        if not sentence:
            return "Please enter a sentence.", {}

        ids, mask = self._encode(sentence)
        logits = self.model.predict_sentiment(ids, mask)
        probs = logits.softmax(dim=-1).squeeze(0).detach().cpu()
        pred = int(probs.argmax().item())
        label_scores = {SENTIMENT_LABELS[i]: float(probs[i]) for i in range(len(probs))}
        return f"{pred} - {SENTIMENT_LABELS[pred]}", label_scores

    @torch.no_grad()
    def paraphrase(self, sentence_1: str, sentence_2: str):
        sentence_1 = sentence_1.strip()
        sentence_2 = sentence_2.strip()
        if not sentence_1 or not sentence_2:
            return "Please enter both sentences.", 0.0

        ids1, mask1 = self._encode(sentence_1)
        ids2, mask2 = self._encode(sentence_2)
        logit = self.model.predict_paraphrase(ids1, mask1, ids2, mask2)
        prob = float(logit.sigmoid().item())
        label = "duplicate / same meaning" if prob >= 0.5 else "not duplicate"
        return label, prob

    @torch.no_grad()
    def similarity(self, sentence_1: str, sentence_2: str):
        sentence_1 = sentence_1.strip()
        sentence_2 = sentence_2.strip()
        if not sentence_1 or not sentence_2:
            return "Please enter both sentences.", 0.0

        ids1, mask1 = self._encode(sentence_1)
        ids2, mask2 = self._encode(sentence_2)
        raw_score = float(self.model.predict_similarity(ids1, mask1, ids2, mask2).item())
        score_0_5 = raw_score * float(self.cfg.losses.get("sts_label_scale", 5.0))
        score_0_5 = max(0.0, min(5.0, score_0_5))
        return f"{score_0_5:.2f} / 5", score_0_5


def build_app(demo_model: DemoModel):
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Gradio is not installed. Run: pip install gradio") from exc

    with gr.Blocks(title="Multitask BERT NLP Demo") as app:
        gr.Markdown(
            """
            # Multitask BERT NLP Demo

            One checkpoint, three inference modes:
            **SST sentiment classification**, **paraphrase detection**, and
            **semantic textual similarity**.
            """
        )

        with gr.Tab("SST: Sentiment"):
            sst_text = gr.Textbox(
                label="Sentence",
                value="I am really confused and frustrated about this assignment.",
            )
            sst_button = gr.Button("Analyze sentiment")
            sst_label = gr.Textbox(label="Predicted class")
            sst_probs = gr.Label(label="Class probabilities")
            sst_button.click(demo_model.sentiment, inputs=sst_text, outputs=[sst_label, sst_probs])

        with gr.Tab("Para: Duplicate question"):
            para_1 = gr.Textbox(label="Sentence 1", value="How do I submit the final report?")
            para_2 = gr.Textbox(label="Sentence 2", value="Where should I upload my project report?")
            para_button = gr.Button("Check paraphrase")
            para_label = gr.Textbox(label="Prediction")
            para_prob = gr.Number(label="Duplicate probability")
            para_button.click(
                demo_model.paraphrase,
                inputs=[para_1, para_2],
                outputs=[para_label, para_prob],
            )

        with gr.Tab("STS: Similarity"):
            sts_1 = gr.Textbox(label="Sentence 1", value="I do not understand how to fine-tune BERT.")
            sts_2 = gr.Textbox(
                label="Sentence 2",
                value="I am confused about training a pretrained transformer.",
            )
            sts_button = gr.Button("Score similarity")
            sts_label = gr.Textbox(label="Similarity score")
            sts_score = gr.Number(label="Numeric score")
            sts_button.click(
                demo_model.similarity,
                inputs=[sts_1, sts_2],
                outputs=[sts_label, sts_score],
            )

    return app


def main(argv=None) -> int:
    args = parse_args(argv)
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    demo_model = DemoModel(args.config, args.checkpoint, args.device)
    app = build_app(demo_model)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
