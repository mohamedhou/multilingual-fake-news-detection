from src.train_baseline import normalize_label


def test_fake_labels():
    assert normalize_label("fake") == "fake"
    assert normalize_label("1") == "fake"


def test_real_labels():
    assert normalize_label("real") == "real"
    assert normalize_label("0") == "real"
