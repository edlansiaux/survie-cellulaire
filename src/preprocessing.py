"""Classical and N2V preprocessing for fluorescence microscopy images."""
import numpy as np
from scipy import ndimage
from skimage import morphology


def pnormalize(im, p1=1, p2=99.8):
    lo, hi = np.percentile(im, (p1, p2))
    return np.clip((im.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)


class ClassicalPreprocessor:
    """Rolling-ball background subtraction → median filter → percentile normalisation."""
    def __init__(self, rolling_ball_radius=50, median_size=3):
        self.radius = rolling_ball_radius
        self.med    = median_size

    def process_sequence(self, seq):
        seq = np.atleast_3d(seq) if seq.ndim == 2 else seq
        out = np.zeros(seq.shape, dtype=np.float32)
        selem = morphology.disk(self.radius)
        for t in range(seq.shape[0]):
            f = seq[t].astype(np.float32)
            f = np.clip(f - morphology.opening(f, selem), 0, None)
            f = ndimage.median_filter(f, size=self.med)
            out[t] = pnormalize(f)
        return out


class N2VPreprocessor:
    """N2V-style denoising (Gaussian approximation; use csbdeep for full N2V)."""
    def __init__(self, rolling_ball_radius=50):
        self.classical = ClassicalPreprocessor(rolling_ball_radius)

    def process_sequence(self, seq):
        from scipy.ndimage import gaussian_filter
        base = self.classical.process_sequence(seq)
        return np.stack(
            [pnormalize(0.7 * f + 0.3 * gaussian_filter(f, sigma=1.5)) for f in base]
        )
