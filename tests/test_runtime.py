from bgfm.runtime import apply_globals, apply_mapping


def test_apply_globals_is_case_insensitive_for_existing_names():
    ns = {"SEED": 1, "OUT_DIR": "x"}
    apply_globals(ns, {"seed": 42, "out_dir": "y", "unknown": 7})
    assert ns["SEED"] == 42
    assert ns["OUT_DIR"] == "y"
    assert "unknown" not in ns


def test_apply_mapping_preserves_existing_key_case():
    cfg = {"SEED": 1}
    apply_mapping(cfg, {"seed": 42})
    assert cfg["SEED"] == 42
