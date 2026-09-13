from mlinstruct.train.utils.early_stopper import EarlyStopper


def test_improving_sequence_never_stops():
    stopper = EarlyStopper(patience=2, min_delta=0.01)
    for vloss in [1.0, 0.8, 0.6, 0.4, 0.2]:
        assert stopper.early_stop(vloss) is False


def test_flat_plateau_stops_after_patience_epochs():
    stopper = EarlyStopper(patience=3, min_delta=0.01)
    assert stopper.early_stop(1.0) is False

    assert stopper.early_stop(1.0) is False
    assert stopper.early_stop(1.0) is False
    assert stopper.early_stop(1.0) is True


def test_improvement_resets_counter():
    stopper = EarlyStopper(patience=2, min_delta=0.01)
    assert stopper.early_stop(1.0) is False
    assert stopper.early_stop(1.0) is False
    assert stopper.counter == 1

    assert stopper.early_stop(0.5) is False
    assert stopper.counter == 0


def test_min_delta_boundary_case():
    stopper = EarlyStopper(patience=1, min_delta=0.1)
    stopper.early_stop(1.0)

    assert stopper.early_stop(0.95) is True
    assert stopper.min_vloss == 1.0
