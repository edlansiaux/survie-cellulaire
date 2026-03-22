"""Omnipose-based cell segmentation with Otsu fallback."""
import numpy as np
from skimage import measure, morphology
from skimage.filters import threshold_otsu


class OmniposeSegmenter:
    def __init__(self, diameter=75, flow_threshold=0.6, use_gpu=False):
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.model = None
        try:
            from cellpose_omni import models
            self.model = models.CellposeModel(model_type="cyto2_omni", gpu=use_gpu)
        except Exception:
            pass

    def segment_sequence(self, seq, max_frames=None):
        T = min(seq.shape[0], max_frames) if max_frames else seq.shape[0]
        labels = np.zeros((T, *seq.shape[1:]), dtype=np.uint16)
        for t in range(T):
            frame = (seq[t] * 255).astype(np.uint8) if seq[t].max() <= 1 else seq[t].astype(np.uint8)
            try:
                if self.model:
                    labels[t] = self.model.eval(frame, diameter=self.diameter,
                                                flow_threshold=self.flow_threshold)[0]
                else:
                    raise RuntimeError("no model")
            except Exception:
                labels[t] = measure.label(
                    morphology.binary_dilation(frame > threshold_otsu(frame), morphology.disk(2))
                )
        return labels
