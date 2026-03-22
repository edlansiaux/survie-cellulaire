"""Quantitative analysis: segmentation metrics + survival statistics."""
import numpy as np
import pandas as pd
from skimage import measure


class CellAnalysis:
    @staticmethod
    def analyze_segmentation_frame(lf):
        if lf.max() == 0:
            return {"n_cells": 0, "mean_area": 0}
        areas = [p.area for p in measure.regionprops(lf.astype(int))]
        return {"n_cells": len(areas), "mean_area": float(np.mean(areas))}

    @staticmethod
    def analyze_cell_survival(tracks_df):
        if len(tracks_df) == 0:
            return {"total_tracked": 0, "mean_lifetime": 0, "persistence_ratio": 0}
        lf = tracks_df["length"].values
        return {"total_tracked": len(tracks_df), "mean_lifetime": float(np.mean(lf)),
                "persistence_ratio": float(np.mean(lf >= 5))}

    @staticmethod
    def generate_report(labels_stack, tracks_df):
        a = CellAnalysis()
        seg_df = pd.DataFrame([
            dict(a.analyze_segmentation_frame(labels_stack[t]), frame=t)
            for t in range(labels_stack.shape[0])
        ])
        surv = a.analyze_cell_survival(tracks_df)
        return {
            "segmentation": {
                "dataframe": seg_df,
                "summary": {
                    "mean_cells_per_frame": seg_df["n_cells"].mean(),
                    "std_cells_per_frame":  seg_df["n_cells"].std(),
                    "overall_mean_area":    seg_df["mean_area"].mean(),
                    "overall_std_area":     seg_df["mean_area"].std(),
                }
            },
            "tracking": {"dataframe": tracks_df, "summary": surv}
        }
