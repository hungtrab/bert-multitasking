import torch

from multitask_bert.models.bert_encoder import MiniBertConfig, MiniBertModel


def _tiny_config() -> MiniBertConfig:
    return MiniBertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )


def test_minibert_forward_shapes():
    model = MiniBertModel(_tiny_config())
    token_ids = torch.randint(0, 32, (3, 7))
    attention_mask = torch.ones(3, 7, dtype=torch.long)

    out = model(token_ids, attention_mask)

    assert out.cls.shape == (3, 16)
    assert out.sequence.shape == (3, 7, 16)


def test_minibert_forward_from_embeddings_shapes():
    model = MiniBertModel(_tiny_config())
    token_ids = torch.randint(0, 32, (2, 5))
    attention_mask = torch.ones(2, 5, dtype=torch.long)
    embeddings = model.embed_tokens(token_ids)

    out = model.forward_from_embeddings(embeddings, attention_mask)

    assert out.cls.shape == (2, 16)
    assert out.sequence.shape == (2, 5, 16)

