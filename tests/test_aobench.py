from procom.oracle import aobench


def test_every_layer_has_a_percentage():
    for layer in range(aobench.N_LAYERS):
        percent = aobench.layer_to_percent(layer)
        assert int(aobench.N_LAYERS * (percent / 100)) == layer


def test_trained_window_gets_the_paper_percentages():
    # layer_combinations of ceselder/qwen3-8b-ao-v3-best
    assert [aobench.layer_to_percent(layer) for layer in aobench.TRAINED_WINDOW] == [
        59,
        62,
        64,
        67,
        70,
    ]


def test_schedule_starts_at_the_trained_window_and_moves_outwards():
    starts = aobench.window_schedule()
    assert starts[:5] == [21, 20, 22, 19, 23]
    assert sorted(starts) == list(range(13, 32))
    assert aobench.window_layers(starts[0]) == list(aobench.TRAINED_WINDOW)
    assert max(aobench.window_layers(max(starts))) == aobench.N_LAYERS - 1


def test_schedule_with_a_stride_keeps_the_trained_window():
    assert sorted(aobench.window_schedule(stride=2)) == [13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    assert sorted(aobench.window_schedule(first_start=14, stride=3)) == [15, 18, 21, 24, 27, 30]


def test_eta_uses_per_eval_durations_and_a_fallback():
    durations = {"a": [100.0, 200.0], "b": [50.0]}
    remaining = [(21, "a"), (21, "b"), (21, "c")]
    assert aobench.estimate_remaining_seconds(remaining, durations) == 150 + 50 + 350 / 3
    assert aobench.estimate_remaining_seconds(remaining, {}) is None
