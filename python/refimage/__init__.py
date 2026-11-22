"""Images with metadata, allocation-aware processing pipelines and FITS output.

``refimage.Image`` holds pixels (uint8, uint16 -- optionally tagged 10/12/14-bit -- or
float32) together with a timestamp, an exposure and FITS-style metadata cards;
``numpy.asarray(image)`` views its pixels without copying.

``refimage.Pipeline`` is a reusable recipe (debayer, luma, scaling, type conversion,
crop/ROI, flips and rotations, resize, homography warp and ``map_coordinates``).
``Pipeline.compile`` allocates every buffer once and returns a ``Runner`` whose ``run``
reuses them -- and, with ``out=``, the output image -- for every frame.
"""

from ._native import (
    ExposureResult,
    Image,
    ImageSpec,
    OptimumExposure,
    Pipeline,
    RefImageError,
    Runner,
    __version__,
)

__all__ = [
    "ExposureResult",
    "Image",
    "ImageSpec",
    "OptimumExposure",
    "Pipeline",
    "RefImageError",
    "Runner",
    "__version__",
]
