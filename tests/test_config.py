from pathlib import Path

from multitask_bert.utils import load_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_default_loads():
    cfg = load_config(CONFIG_DIR / "default.yaml")
    assert cfg.model.encoder == "bert-base-uncased"
    assert cfg.training.batch_size == 32
    assert cfg.losses.smart.enabled is False


def test_inheritance():
    cfg = load_config(CONFIG_DIR / "round_robin.yaml")
    # inherited from default
    assert cfg.model.encoder == "bert-base-uncased"
    # overridden in round_robin
    assert cfg.experiment_name == "round_robin"


def test_smart_enables_smart():
    cfg = load_config(CONFIG_DIR / "smart.yaml")
    assert cfg.losses.smart.enabled is True
    assert cfg.model.use_relational_layer is True   # inherited from rich_relational
    assert cfg.experiment_name == "smart"


def test_path_interpolation():
    cfg = load_config(CONFIG_DIR / "round_robin.yaml")
    assert cfg.paths.output_dir == "runs/round_robin"
