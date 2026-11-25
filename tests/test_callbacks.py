from pathlib import Path

from train.callbacks import MetricCheckpointEarlyStopping


class FakeTrainer(object):
    def __init__(self, tmp):
        self.wdir = Path(tmp)
        self.last = self.wdir / "last.pt"
        self.epoch, self.metrics, self.stop = 0, {}, False

    def step(self, cb, epoch, map50):
        self.epoch = epoch - 1
        self.metrics = {"metrics/mAP50(B)": map50}
        self.last.write_text("weights@{}".format(epoch))
        cb.on_model_save(self)


def test_saves_best_and_stops_after_patience(tmp_path):
    cb = MetricCheckpointEarlyStopping(patience=3, min_delta=0.01, target=0.65)
    tr = FakeTrainer(tmp_path)
    seq = [0.40, 0.55, 0.66, 0.665, 0.664, 0.669, 0.60]  # tiny gains < min_delta after epoch 3
    for e, v in enumerate(seq, 1):
        tr.step(cb, e, v)
        if tr.stop:
            break
    assert cb.target_epoch == 3
    assert cb.best_epoch == 6 and abs(cb.best - 0.669) < 1e-9       # best still saved...
    assert (tmp_path / "best_map50.pt").read_text() == "weights@6"
    assert tr.stop and cb.stopped_epoch == 6                          # ...but no *meaningful* gain since epoch 3


def test_no_stop_while_improving(tmp_path):
    cb = MetricCheckpointEarlyStopping(patience=2, min_delta=0.0)
    tr = FakeTrainer(tmp_path)
    for e in range(1, 11):
        tr.step(cb, e, e / 10.0)
    assert not tr.stop and cb.best_epoch == 10
