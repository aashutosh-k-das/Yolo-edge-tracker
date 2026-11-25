"""ByteTrack multi-object tracker (Zhang et al., ECCV 2022), standalone implementation.

Key idea: associate *high*-confidence detections first, then try to rescue unmatched tracks
with *low*-confidence detections (occluded / blurred objects) instead of throwing them away.

Input to `BYTETracker.update`: detections as an (N, 6) array [x1, y1, x2, y2, score, cls]
in original-image pixels (+ optionally the frame, for appearance-based re-identification). IMPORTANT: run the detector with a LOW confidence threshold
(~0.1) so the second association stage has low-score boxes to work with.

Occlusion handling:
  * short occlusion  - a lost track keeps being Kalman-predicted for `track_buffer` frames and is
                       re-activated with its OLD id when an IoU match appears (standard ByteTrack)
  * re-identification (reid=True, an extension to ByteTrack) - a 4th stage matches still-unmatched
    high-score detections to lost tracks by centre distance to the predicted OR last-seen position
    (hovering drones, erratic / fast small objects where IoU is 0), same class, similar size, and
    - if a frame is given - similar HSV colour histogram.

Pure numpy/scipy, Python 3.6+ compatible (Jetson Nano / JetPack 4.6).
"""
import numpy as np

from .kalman_filter import KalmanFilter
from .appearance import cosine_distance, hsv_hist
from .matching import fuse_score, iou_distance, linear_assignment


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack(object):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.score = float(score)
        self.cls = int(cls)
        self.cls_votes = {self.cls: self.score}  # score-weighted class votes over the track's life
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.state = TrackState.New
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0
        self.last_obs = self.tlbr        # last *observed* box (not predicted) - used by re-ID
        self.curr_feat = None            # appearance of the current detection
        self.feat = None                 # smoothed appearance of the track (EMA)
        self.reid_count = 0

    def _absorb(self, det):
        """Take score / class / last-observed box / appearance from a matched detection."""
        self.score = det.score
        # majority class: a detector that flickers car/truck (or drone/aircraft) keeps one stable label
        self.cls_votes[det.cls] = self.cls_votes.get(det.cls, 0.0) + det.score
        self.cls = max(self.cls_votes, key=self.cls_votes.get)
        self.last_obs = det.tlbr
        if det.curr_feat is not None:
            f = det.curr_feat if self.feat is None else 0.9 * self.feat + 0.1 * det.curr_feat
            self.feat = f / (np.linalg.norm(f) + 1e-12)

    # ----------------------------------------------------------------- Kalman
    @staticmethod
    def multi_predict(stracks):
        if not stracks:
            return
        mean = np.asarray([st.mean.copy() for st in stracks])
        cov = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                mean[i][7] = 0  # freeze height velocity for lost tracks
        mean, cov = STrack.shared_kalman.multi_predict(mean, cov)
        for i, st in enumerate(stracks):
            st.mean, st.covariance = mean[i], cov[i]

    def activate(self, kalman_filter, frame_id, track_id):
        self.kalman_filter = kalman_filter
        self.track_id = track_id
        self.mean, self.covariance = kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.feat = self.curr_feat
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # A new track is only confirmed once it is matched again (except on the first frame).
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self._absorb(new_track)

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self._absorb(new_track)

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @property
    def end_frame(self):
        return self.frame_id

    # -------------------------------------------------------------- geometry
    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret

    def __repr__(self):
        return "STrack(id={}, cls={}, frames={}-{})".format(
            self.track_id, self.cls, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, track_thresh=0.5, low_thresh=0.1, match_thresh=0.8,
                 track_buffer=30, frame_rate=30, new_track_thresh=None,
                 reid=False, reid_max_dist=2.0, reid_app_thresh=0.35, reid_size_ratio=3.0):
        """
        track_thresh:     detections above this are "high" confidence (stage-1 association).
        low_thresh:       detections between low_thresh and track_thresh go to stage 2.
        match_thresh:     max (1 - IoU*score) cost accepted in stage 1.
        track_buffer:     frames a lost track is kept alive (at 30 FPS; scaled by frame_rate).
        new_track_thresh: min score to start a new track (default track_thresh + 0.1).
        reid:             enable the re-identification stage for lost tracks.
        reid_max_dist:    max centre distance, in units of the track's box diagonal.
        reid_app_thresh:  max appearance (cosine) distance when both features exist.
        reid_size_ratio:  max area ratio between the lost track and the detection.
        """
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.det_thresh = new_track_thresh if new_track_thresh is not None else track_thresh + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.reid = reid
        self.reid_max_dist = reid_max_dist
        self.reid_app_thresh = reid_app_thresh
        self.reid_size_ratio = reid_size_ratio
        self.kalman_filter = KalmanFilter()
        self.reset()

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.frame_id = 0
        self._id_count = 0
        self.reid_events = 0

    def _next_id(self):
        self._id_count += 1
        return self._id_count

    @staticmethod
    def _to_stracks(dets):
        return [STrack(np.r_[d[:2], d[2:4] - d[:2]], d[4], d[5]) for d in dets]

    def update(self, dets, frame=None):
        """dets: (N, 6) [x1, y1, x2, y2, score, cls]; frame: BGR image (optional, enables
        appearance features for re-ID). Returns list of active, confirmed STracks."""
        self.frame_id += 1
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 6)
        activated, refind, lost, removed = [], [], [], []

        scores = dets[:, 4]
        dets_high = self._to_stracks(dets[scores >= self.track_thresh])
        dets_low = self._to_stracks(dets[(scores > self.low_thresh) & (scores < self.track_thresh)])
        if self.reid and frame is not None:
            for d in dets_high:
                d.curr_feat = hsv_hist(frame, d.tlbr)

        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked = [t for t in self.tracked_stracks if t.is_activated]

        # ---- Stage 1: confirmed + lost tracks  <->  high-score detections
        pool = _joint(tracked, self.lost_stracks)
        STrack.multi_predict(pool)
        dists = fuse_score(iou_distance(pool, dets_high), dets_high)
        matches, u_track, u_det = linear_assignment(dists, self.match_thresh)
        for it, idet in matches:
            track, det = pool[it], dets_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)

        # ---- Stage 2: remaining *tracked* tracks  <->  low-score detections
        r_tracked = [pool[i] for i in u_track if pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked, dets_low)
        matches, u_track2, _ = linear_assignment(dists, 0.5)
        for it, idet in matches:
            track, det = r_tracked[it], dets_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)
        for it in u_track2:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        # ---- Stage 3: unconfirmed (1-frame-old) tracks  <->  leftover high detections
        leftovers = [dets_high[i] for i in u_det]
        dists = fuse_score(iou_distance(unconfirmed, leftovers), leftovers)
        matches, u_unconf, u_det = linear_assignment(dists, 0.7)
        for it, idet in matches:
            unconfirmed[it].update(leftovers[idet], self.frame_id)
            activated.append(unconfirmed[it])
        for it in u_unconf:
            unconfirmed[it].mark_removed()
            removed.append(unconfirmed[it])

        # ---- Stage 4 (extension): re-identify lost tracks with leftover high detections
        remaining = [leftovers[i] for i in u_det]
        if self.reid and remaining:
            candidates = [t for t in self.lost_stracks if t.state == TrackState.Lost] + lost
            matches, u_rem = self._reid_match(candidates, remaining)
            for it, idet in matches:
                candidates[it].re_activate(remaining[idet], self.frame_id)
                candidates[it].reid_count += 1
                refind.append(candidates[it])
            self.reid_events += len(matches)
            lost = [t for t in lost if t.state == TrackState.Lost]
            remaining = [remaining[i] for i in u_rem]

        # ---- Start new tracks
        for track in remaining:
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id, self._next_id())
            activated.append(track)

        # ---- Expire tracks that have been lost too long
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        # ---- Bookkeeping
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = _joint(self.tracked_stracks, activated)
        self.tracked_stracks = _joint(self.tracked_stracks, refind)
        self.lost_stracks = _sub(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = _sub(self.lost_stracks, removed)
        self.tracked_stracks, self.lost_stracks = _remove_duplicates(
            self.tracked_stracks, self.lost_stracks)

        return [t for t in self.tracked_stracks if t.is_activated]

    def _reid_match(self, tracks, dets):
        """Gated Hungarian matching of lost tracks to detections.

        motion cost     = min(dist to predicted centre, dist to last-seen centre) / (reid_max_dist * diag)
        appearance cost = cosine distance of HSV histograms (only if both exist)
        Gates: same class, motion cost <= 1, area ratio <= reid_size_ratio, appearance <= thresh.
        """
        if not tracks or not dets:
            return np.empty((0, 2), dtype=int), list(range(len(dets)))
        cost = np.full((len(tracks), len(dets)), np.inf)
        for i, t in enumerate(tracks):
            pred, last = t.tlbr, t.last_obs
            diag = np.hypot(last[2] - last[0], last[3] - last[1])
            area_t = max((last[2] - last[0]) * (last[3] - last[1]), 1e-6)
            centres = [(pred[:2] + pred[2:]) / 2, (last[:2] + last[2:]) / 2]
            for j, d in enumerate(dets):
                if d.cls != t.cls:
                    continue
                b = d.tlbr
                ratio = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6) / area_t
                if not 1.0 / self.reid_size_ratio <= ratio <= self.reid_size_ratio:
                    continue
                c = (b[:2] + b[2:]) / 2
                motion = min(np.hypot(*(c - p)) for p in centres) / (self.reid_max_dist * diag + 1e-6)
                if motion > 1.0:
                    continue
                if t.feat is not None and d.curr_feat is not None:
                    app = cosine_distance(t.feat, d.curr_feat)
                    if app > self.reid_app_thresh:
                        continue
                    cost[i, j] = 0.5 * motion + 0.5 * app / self.reid_app_thresh
                else:
                    cost[i, j] = motion
        matches, _, u_det = linear_assignment(np.where(np.isfinite(cost), cost, 1e5), 1.0)
        return matches, u_det

    def lost_tracks(self):
        """Currently lost (occluded) but still recoverable tracks - for visualisation."""
        return [t for t in self.lost_stracks if t.state == TrackState.Lost]


def tracks_to_array(tracks):
    """List[STrack] -> (N, 7) array [x1, y1, x2, y2, track_id, score, cls]."""
    if not tracks:
        return np.zeros((0, 7), dtype=np.float32)
    return np.array([np.r_[t.tlbr, t.track_id, t.score, t.cls] for t in tracks], dtype=np.float32)


def _joint(a, b):
    seen = set(t.track_id for t in a)
    out = list(a)
    for t in b:
        if t.track_id not in seen:
            seen.add(t.track_id)
            out.append(t)
    return out


def _sub(a, b):
    ids = set(t.track_id for t in b)
    return [t for t in a if t.track_id not in ids]


def _remove_duplicates(a, b):
    """If a tracked and a lost track overlap heavily, keep the one that has lived longer."""
    dist = iou_distance(a, b)
    pairs = np.where(dist < 0.15)
    dup_a, dup_b = set(), set()
    for p, q in zip(*pairs):
        age_p = a[p].frame_id - a[p].start_frame
        age_q = b[q].frame_id - b[q].start_frame
        if age_p > age_q:
            dup_b.add(q)
        else:
            dup_a.add(p)
    return ([t for i, t in enumerate(a) if i not in dup_a],
            [t for i, t in enumerate(b) if i not in dup_b])
