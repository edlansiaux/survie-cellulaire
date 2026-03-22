# Cell Imaging Analysis Pipeline

Automated preprocessing → segmentation → tracking → analysis for fluorescence microscopy.

## Features
- **Preprocessing**: Classical (rolling-ball + median) or N2V-style denoising
- **Segmentation**: Omnipose `cyto2_omni` (Otsu fallback if unavailable)
- **Tracking**: Hungarian algorithm centroid linking
- **Analysis**: Cell count, area, survival metrics

## Quick start
```bash
pip install -r requirements.txt
python -c "from src import ClassicalPreprocessor, OmniposeSegmenter; print('ok')"
```

## References
- Omnipose: https://github.com/kevinjohncutler/omnipose
- CSBDeep / N2V: https://github.com/csbdeep/csbdeep
