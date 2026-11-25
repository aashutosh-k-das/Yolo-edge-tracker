"""Training callbacks: early stopping + best-checkpoint saving on the metric YOU care about.

Why not just Ultralytics' built-ins? Its best.pt and `patience` both track "fitness", which in
recent versions is mAP50-95. Our project target is mAP@50 > 0.65, so this callback:
  * saves  weights/best_map50.pt  whenever val mAP50 improves
  * stops training when mAP50 has not improved by > min_delta for `patience` epochs
  * announces the epoch where the target is first reached
  * restores its state on --resume (reads results.csv)
  * writes callback_summary.json for the report

Hooked on `on_model_save` (fires once per epoch right after last.pt is written, with that
epoch's val metrics). `on_fit_epoch_end` is avoided because Ultralytics also fires it during
the final best.pt evaluation, which would be miscounted as an extra epoch.
"""
import csv
import json
import logging
import shutil

try:
    from ultralytics.utils import LOGGER
except ImportError:  # lets the logic be unit-tested without ultralytics/torch
    LOGGER = logging.getLogger(__name__)


class MetricCheckpointEarlyStopping(object):
    def __init__(self, monitor="metrics/mAP50(B)", patience=30, min_delta=0.002, target=None,
                 filename="best_map50.pt"):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.target = target
        self.filename = filename
        self.best = -1.0            # best value seen (checkpoint saved)
        self.best_epoch = 0
        self.ref = -1.0             # value that last reset the patience counter
        self.ref_epoch = 0
        self.target_epoch = None
        self.stopped_epoch = None
        self.history = []

    def attach(self, model):
        model.add_callback("on_train_start", self.on_train_start)
        model.add_callback("on_model_save", self.on_model_save)
        model.add_callback("on_train_end", self.on_train_end)
        return self

    def _path(self, trainer):
        return trainer.wdir / self.filename

    # ----------------------------------------------------------------- hooks
    def on_train_start(self, trainer):
        """On --resume, rebuild best/patience state from the existing results.csv."""
        if trainer.start_epoch == 0 or not trainer.csv.exists():
            return
        with open(trainer.csv) as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items()}
                if self.monitor in row and row[self.monitor]:
                    self._update(int(float(row["epoch"])), float(row[self.monitor]), trainer=None)
        LOGGER.info("[callback] resumed: best {}={:.4f} @ epoch {}".format(
            self.monitor, self.best, self.best_epoch))

    def on_model_save(self, trainer):
        value = trainer.metrics.get(self.monitor)
        if value is None:
            return
        self._update(trainer.epoch + 1, float(value), trainer)

    def on_train_end(self, trainer):
        path = self._path(trainer)
        if path.exists():
            from ultralytics.utils.torch_utils import strip_optimizer
            strip_optimizer(path)  # drop optimizer state -> ~1/3 the file size
        summary = {
            "monitor": self.monitor, "best": round(self.best, 5), "best_epoch": self.best_epoch,
            "target": self.target, "target_reached_epoch": self.target_epoch,
            "early_stopped_epoch": self.stopped_epoch, "checkpoint": str(path),
        }
        with open(trainer.save_dir / "callback_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        LOGGER.info("[callback] summary: {}".format(summary))

    # ----------------------------------------------------------------- logic
    def _update(self, epoch, value, trainer):
        self.history.append((epoch, value))
        msg = []
        if value > self.best:
            self.best, self.best_epoch = value, epoch
            if trainer is not None:
                shutil.copy2(trainer.last, self._path(trainer))
                msg.append("new best -> {}".format(self.filename))
        if value > self.ref + self.min_delta:
            self.ref, self.ref_epoch = value, epoch
        if self.target is not None and value >= self.target and self.target_epoch is None:
            self.target_epoch = epoch
            msg.append("TARGET {} REACHED".format(self.target))

        wait = epoch - self.ref_epoch
        if trainer is not None:
            LOGGER.info("[callback] epoch {} {}={:.4f} (best {:.4f} @ {}, no-improve {}/{}) {}".format(
                epoch, self.monitor, value, self.best, self.best_epoch, wait, self.patience,
                " | ".join(msg)))
            if self.patience and wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop = True
                LOGGER.info("[callback] early stopping: no {} improvement > {} for {} epochs".format(
                    self.monitor, self.min_delta, self.patience))
