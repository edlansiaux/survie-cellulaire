"""Hungarian-algorithm cell tracker."""
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.spatial.distance import cdist
from skimage import measure


class SimpleTracker:
    def __init__(self, max_distance=50, min_track_length=3):
        self.max_distance    = max_distance
        self.min_track_length = min_track_length

    def _centroids(self, lf):
        props = measure.regionprops(lf.astype(int))
        if not props:
            return np.zeros((0, 2)), np.array([])
        return np.array([p.centroid for p in props]), np.array([p.label for p in props])

    def track_sequence(self, labels_stack):
        T, tracks, tid, fa = labels_stack.shape[0], {}, 1, {}
        c0, ids0 = self._centroids(labels_stack[0])
        fa[0] = {}
        for cid in ids0:
            tracks[tid] = [(0, cid)]; fa[0][cid] = tid; tid += 1
        for t in range(1, T):
            cc, idc = self._centroids(labels_stack[t])
            cp, idp = self._centroids(labels_stack[t - 1])
            fa[t] = {}
            if len(cc) == 0:
                continue
            if len(cp) == 0:
                for cid in idc:
                    tracks[tid] = [(t, cid)]; fa[t][cid] = tid; tid += 1
                continue
            D = cdist(cp, cc); D[D > self.max_distance] = np.inf
            ri, ci_ = optimize.linear_sum_assignment(D)
            matched = set()
            for r, c in zip(ri, ci_):
                if np.isfinite(D[r, c]):
                    tkid = fa[t - 1][idp[r]]
                    tracks[tkid].append((t, idc[c])); fa[t][idc[c]] = tkid; matched.add(c)
            for i, cid in enumerate(idc):
                if i not in matched:
                    tracks[tid] = [(t, cid)]; fa[t][cid] = tid; tid += 1
        return {k: v for k, v in tracks.items() if len(v) >= self.min_track_length}, fa

    def extract_track_statistics(self, tracks, labels_stack):
        rows = []
        for tk, track in tracks.items():
            frames = [f for f, _ in track]
            areas  = [p.area for f, cid in track
                      for p in measure.regionprops(labels_stack[f].astype(int))
                      if p.label == cid]
            rows.append({"track_id": tk, "length": len(track),
                         "start_frame": min(frames), "end_frame": max(frames),
                         "mean_area": np.mean(areas) if areas else 0})
        return pd.DataFrame(rows)
