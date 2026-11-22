"""Type stubs and documentation for ``refimage._native``, the compiled core of ``refimage``.

Conventions used throughout:

- **Pixel types** are named ``"u8"``, ``"u16"``, ``"f32"`` (the storage types) and
  ``"u10"``, ``"u12"``, ``"u14"`` (``uint16`` storage tagged with the sensor's meaningful
  bit depth). Where a storage type is expected, a numpy dtype (``np.uint8``,
  ``"float32"``, ...) is accepted too. ``float32`` images are normalized: pixel
  operations saturate them into ``[0, 1]``.
- **Color spaces** are ``"gray"``, ``"rgb"``, a Bayer pattern (``"rggb"``, ``"bggr"``,
  ``"gbrg"``, ``"grbg"``, optionally prefixed ``"bayer_"``), or a ``(channels, name)``
  tuple for anything else.
- **Shapes** follow numpy: ``(height, width)`` for one channel, ``(height, width,
  channels)`` otherwise, row-major with channels interleaved. Sizes that the Rust API
  takes as ``width, height`` (``crop``, ``resize_to_fit``, ``ImageSpec``) keep that order;
  ``output_shape`` arguments are numpy-style ``(rows, cols)``.
- **Exposures** are float seconds or a ``datetime.timedelta``; timestamps are
  timezone-aware ``datetime`` objects, stored in UTC.
- Every pixel-processing call releases the GIL.
- Invalid images, pipelines, metadata or FITS options raise :class:`RefImageError`.
"""

from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Sequence, final

import numpy as np
import numpy.typing as npt

__version__: str
"""Version of the compiled extension (the ``refimage-py`` crate version)."""

ColorSpace = str | tuple[int, str]
PixelTypeName = Literal["u8", "u10", "u12", "u14", "u16", "f32"]
BoundaryModeName = Literal["constant", "nearest", "edge", "mirror", "symmetric", "reflect", "wrap"]
SplineOrder = Literal[0, 1, 3]
ExposureLike = float | timedelta
MetaValue = int | float | str | datetime | timedelta
PixelArray = npt.NDArray[np.uint8] | npt.NDArray[np.uint16] | npt.NDArray[np.float32]
FitsCompressionName = Literal["gzip", "rice", "hcompress"]

class RefImageError(ValueError):
    """Raised for invalid images, pipelines, metadata and FITS output.

    A subclass of :class:`ValueError`. I/O failures while writing FITS files raise the
    usual :class:`OSError` subclasses instead, and a buffer conflict on
    ``Runner.run(..., out=...)`` raises :class:`BufferError`.
    """

@final
class Image:
    """An image: pixels plus metadata.

    The pixels are ``uint8``, ``uint16`` (optionally tagged as 10-, 12- or 14-bit sensor
    data) or ``float32``, stored row-major with channels interleaved. The metadata is a
    UTC timestamp, an exposure, an optional frame counter, and any number of named,
    FITS-style cards (see :meth:`set_key`).

    ``numpy.asarray(img)`` (or :meth:`numpy`) is a zero-copy, **writable** view of the
    pixels: writing to the array changes the image, and a :class:`Runner` writing into
    the image (``runner.run(frame, out=img)``) is visible through existing views.

    Images compare equal (``==``) when pixels and metadata are identical, and support
    :func:`copy.copy`, :func:`copy.deepcopy` and :mod:`pickle`.

    Examples
    --------
    >>> raw = np.zeros((1080, 1920), np.uint16)
    >>> img = refimage.Image(raw, "rggb", exposure=0.02, bit_depth=12)
    >>> img.set_key("CAMERA", "ZWO ASI2600", comment="camera model")
    >>> img.shape, img.pixel_type
    ([1080, 1920], 'u12')
    """

    def __new__(
        cls,
        data: PixelArray,
        color_space: ColorSpace | None = None,
        *,
        timestamp: datetime | None = None,
        exposure: ExposureLike | None = None,
        bit_depth: int | None = None,
        frame_id: int | None = None,
    ) -> Image:
        """Create an image by copying a numpy array.

        Parameters
        ----------
        data
            ``(height, width)`` or ``(height, width, channels)`` array of dtype
            ``uint8``, ``uint16`` or ``float32``. Any memory layout is accepted; the
            pixels are copied into row-major order.
        color_space
            The color space. Defaults to ``"gray"`` for one channel and ``"rgb"`` for
            three; any other channel count needs an explicit ``(channels, name)`` tuple.
            Raw sensor data is a Bayer pattern, e.g. ``"rggb"``.
        timestamp
            Acquisition time, a timezone-aware ``datetime``. Defaults to now.
        exposure
            Exposure in seconds or as a ``timedelta``. Defaults to ``0`` (unknown).
        bit_depth
            Tag ``uint16`` data as ``10``, ``12`` or ``14`` bit, so pipeline operations
            scale and saturate against the sensor's true range.
        frame_id
            Acquisition frame counter.

        Raises
        ------
        RefImageError
            For an unsupported dtype or shape, a color space whose channel count does
            not match the array, or ``bit_depth`` on non-``uint16`` data.
        """

    @staticmethod
    def from_bytes(data: bytes) -> Image:
        """Decode an image produced by :meth:`to_bytes`.

        Also accepts the bytes Rust produces with ``bincode::serialize`` (bincode 1.x) of
        a ``GenericImageRef`` or ``GenericImageOwned``, so frames sent by a Rust
        acquisition program can be decoded directly.

        Raises
        ------
        RefImageError
            If ``data`` is not a serialized image.
        """

    def to_bytes(self) -> bytes:
        """Serialize pixels and metadata to bytes.

        The encoding is bincode 1.x of ``GenericImageOwned``, identical to what Rust's
        ``bincode::serialize`` writes; read it back with :meth:`from_bytes`.
        """

    def copy(self) -> Image:
        """Return a deep copy (pixels and metadata)."""

    def __copy__(self) -> Image:
        """Support :func:`copy.copy`; a deep copy, like :meth:`copy`."""

    def __deepcopy__(self, memo: dict[int, Any], /) -> Image:
        """Support :func:`copy.deepcopy`."""

    def numpy(self) -> npt.NDArray[Any]:
        """Return a zero-copy, writable numpy view of the pixels.

        Equivalent to ``numpy.asarray(img)``. The view is shaped ``(height, width)`` or
        ``(height, width, channels)`` and has the image's :attr:`dtype`. While any view
        is alive, operations that would reallocate the pixels (a :class:`Runner` writing
        a differently-shaped result into this image) raise :class:`BufferError`.
        """

    def __buffer__(self, flags: int, /) -> memoryview:
        """Export the pixels through the buffer protocol (C-contiguous, writable)."""

    @property
    def width(self) -> int:
        """Width in pixels."""

    @property
    def height(self) -> int:
        """Height in pixels."""

    @property
    def channels(self) -> int:
        """Interleaved channels per pixel (1 for gray and Bayer, 3 for RGB)."""

    @property
    def shape(self) -> list[int]:
        """The numpy shape: ``[height, width]`` or ``[height, width, channels]``."""

    @property
    def dtype(self) -> np.dtype[Any]:
        """The numpy dtype of the pixels (``uint8``, ``uint16`` or ``float32``)."""

    @property
    def pixel_type(self) -> PixelTypeName:
        """The pixel type, including any bit-depth tag (e.g. ``"u12"``)."""

    @property
    def bit_depth(self) -> int:
        """Meaningful bits per sample: 8, 10, 12, 14, 16, or 32 for ``float32``."""

    @property
    def color_space(self) -> ColorSpace:
        """``"gray"``, ``"rgb"``, a Bayer pattern such as ``"rggb"``, or a ``(channels, name)`` tuple."""

    @property
    def nbytes(self) -> int:
        """Size of the pixel data in bytes."""

    @property
    def timestamp(self) -> datetime:
        """Acquisition time as a UTC ``datetime``. Settable with any aware ``datetime``."""

    @timestamp.setter
    def timestamp(self, value: datetime) -> None: ...
    @property
    def exposure(self) -> float:
        """Exposure in seconds (``0.0`` when unknown). Settable with seconds or a ``timedelta``."""

    @exposure.setter
    def exposure(self, value: ExposureLike) -> None: ...
    @property
    def frame_id(self) -> int | None:
        """Acquisition frame counter, or ``None`` if unset. Settable with an unsigned 32-bit int."""

    @frame_id.setter
    def frame_id(self, value: int) -> None: ...
    @property
    def metadata(self) -> dict[str, MetaValue]:
        """The metadata cards as an ordered ``{KEY: value}`` dict.

        A copy: modifying the dict does not change the image (use :meth:`set_key` and
        :meth:`remove_key`). Keys are upper-case. ``timedelta`` and ``datetime`` values
        are stored as several cards (the value plus ``KEY_S`` / ``KEY_NS`` seconds and
        nanoseconds), all of which appear here. The timestamp, exposure and frame id are
        properties, not cards.
        """

    def get_key(self, key: str, default: Any = None) -> MetaValue | Any:
        """Return the value of metadata card ``key`` (case-insensitive), or ``default``."""

    def get_comment(self, key: str) -> str | None:
        """Return the comment of metadata card ``key``, or ``None``."""

    def set_key(self, key: str, value: MetaValue, comment: str | None = None) -> None:
        """Insert or replace metadata card ``key``.

        Parameters
        ----------
        key
            Card name, at most 80 characters, case-insensitive (stored upper-case).
            ``TIMESTAMP``, ``EXPOSURE`` and ``FRAMEID`` are reserved; use the
            :attr:`timestamp`, :attr:`exposure` and :attr:`frame_id` properties.
        value
            An ``int``, ``float``, ``str``, aware ``datetime`` or ``timedelta``.
        comment
            Optional comment, written to FITS headers alongside the value.

        Raises
        ------
        RefImageError
            For a reserved, empty or over-long key, or an over-long value or comment.
        TypeError
            For any other value type.
        """

    def remove_key(self, key: str) -> MetaValue:
        """Remove metadata card ``key`` and return its value.

        Raises
        ------
        KeyError
            If there is no such card.
        """

    def apply(self, pipeline: Pipeline) -> Image:
        """Run ``pipeline`` once and return the result as a new image.

        Same as ``pipeline.apply(self)``. The result carries a copy of this image's
        metadata. For a stream of frames, use :meth:`Pipeline.compile` instead.
        """

    def warp(
        self,
        inverse_map: npt.ArrayLike,
        output_shape: tuple[int, int] | None = None,
        order: SplineOrder = 1,
        mode: BoundaryModeName = "constant",
        cval: float = 0.0,
        prefilter: bool = True,
    ) -> Image:
        """Resample through a homography, like ``skimage.transform.warp`` with a matrix.

        Parameters
        ----------
        inverse_map
            3x3 matrix taking ``(column, row, 1)`` of an *output* pixel to ``(column,
            row, w)`` in the input, which is divided by ``w`` (scikit-image's
            convention). Any affine map, rotation, shear, rescale or perspective
            transform can be expressed this way.
        output_shape
            ``(rows, cols)`` of the result. Defaults to this image's shape.
        order
            Spline order: ``0`` nearest, ``1`` bilinear, ``3`` bicubic B-spline.
        mode
            How the input is extended past its edge: ``"constant"`` (fill with
            ``cval``), ``"nearest"``/``"edge"``, ``"mirror"``/``"symmetric"``,
            ``"reflect"`` or ``"wrap"`` (scipy's definitions).
        cval
            Fill value for ``"constant"``, in raw stored units (``255`` is white on a
            ``uint8`` image, ``1.0`` on a ``float32`` one).
        prefilter
            Apply scipy's B-spline prefilter (only affects ``order=3``). Turn it off to
            smooth instead of interpolate.

        Returns
        -------
        Image
            A new image of ``output_shape``, same pixel type and color space, carrying
            this image's metadata. Results are computed in ``float64``, then rounded
            (integer types) and saturated into the pixel type.

        Raises
        ------
        RefImageError
            For a matrix that is not 3x3 or holds non-finite values, an order other
            than 0/1/3, an unknown mode, or a Bayer image (debayer first).

        Notes
        -----
        For ``order=3`` the result matches ``scipy.ndimage`` (scikit-image's own cubic
        kernel for transform objects is not a B-spline and is not reproduced).
        """

    def map_coordinates(
        self,
        coordinates: npt.ArrayLike,
        order: SplineOrder = 1,
        mode: BoundaryModeName = "constant",
        cval: float = 0.0,
        prefilter: bool = True,
    ) -> Image:
        """Sample at arbitrary positions, like ``scipy.ndimage.map_coordinates``.

        Parameters
        ----------
        coordinates
            Array of shape ``(2, rows, cols)``: the input row (``coordinates[0]``) and
            column (``coordinates[1]``) of every output pixel, in pixel units with the
            centre of the first pixel at ``0.0``.
        order, mode, cval, prefilter
            As for :meth:`warp`.

        Returns
        -------
        Image
            A new ``(rows, cols)`` image (all channels sampled at the same positions)
            carrying this image's metadata.
        """

    def write_fits(
        self,
        path: str | PathLike[str],
        compression: FitsCompressionName | None = None,
        *,
        overwrite: bool = False,
        level: int | None = None,
        tile_rows: int | None = None,
        scale: int = 0,
        smooth: bool = False,
        quantize: float | None = None,
    ) -> Path:
        """Write the image and its metadata to a FITS file.

        Parameters
        ----------
        path
            Destination file.
        compression
            ``None`` for a plain image HDU, or a tile-compressed HDU: ``"gzip"``
            (lossless), ``"rice"`` or ``"hcompress"`` (lossless for integer pixels;
            ``float32`` is quantized).
        overwrite
            Replace an existing file instead of raising :class:`FileExistsError`.
        level
            ``"gzip"`` effort, 0 (store) to 9 (smallest); default 6.
        tile_rows
            Image rows per compressed tile (default: one row for gzip/rice, the whole
            plane for hcompress).
        scale, smooth
            ``"hcompress"`` options; ``scale=0`` is lossless.
        quantize
            Quantization level for compressed ``float32`` data (cfitsio's
            ``QUANTIZE_LEVEL``, default 4.0; larger keeps more precision).

        Returns
        -------
        pathlib.Path
            The path written.

        Notes
        -----
        Metadata cards become header cards (``HIERARCH`` for long keys, ``CONTINUE`` for
        long strings); multi-channel images are stored as planes; ``uint16`` uses
        ``BZERO = 32768``.
        """

    def to_fits_bytes(
        self,
        compression: FitsCompressionName | None = None,
        *,
        level: int | None = None,
        tile_rows: int | None = None,
        scale: int = 0,
        smooth: bool = False,
        quantize: float | None = None,
    ) -> bytes:
        """Return the FITS file :meth:`write_fits` would write, as bytes."""

@final
class ImageSpec:
    """The shape and pixel type a :class:`Pipeline` is compiled for.

    Compiling against an ``ImageSpec`` builds a :class:`Runner` without needing a
    sample frame. :attr:`Runner.input_spec` and :attr:`Runner.output_spec` are
    ``ImageSpec`` objects. Specs compare equal when every field matches.
    """

    def __new__(
        cls,
        width: int,
        height: int,
        color_space: ColorSpace | None = None,
        pixel_type: PixelTypeName | Any | None = None,
    ) -> ImageSpec:
        """Describe a ``width`` x ``height`` frame; ``color_space`` defaults to
        ``"gray"`` and ``pixel_type`` to ``"u16"``."""

    @property
    def width(self) -> int:
        """Width in pixels."""

    @property
    def height(self) -> int:
        """Height in pixels."""

    @property
    def channels(self) -> int:
        """Interleaved channels per pixel."""

    @property
    def color_space(self) -> ColorSpace:
        """The color space, as :attr:`Image.color_space` reports it."""

    @property
    def pixel_type(self) -> PixelTypeName:
        """``"u8"``, ``"u10"``, ``"u12"``, ``"u14"``, ``"u16"`` or ``"f32"``."""

    @property
    def shape(self) -> list[int]:
        """The numpy shape of a matching frame."""

    @property
    def dtype(self) -> np.dtype[Any]:
        """The numpy dtype of a matching frame."""

@final
class Pipeline:
    """A reusable, serializable processing recipe.

    Build one by chaining steps. Every method returns a **new** pipeline, so a base
    recipe can be extended without being modified. Then either:

    - :meth:`apply` it once to an image, or
    - :meth:`compile` it against a frame shape to get a :class:`Runner`, which
      allocates every buffer once and reuses them for every frame.

    Steps are validated when the pipeline is compiled or applied, against the actual
    shape flowing through the chain (e.g. ``to_luma`` before ``debayer`` on Bayer data
    is an error).

    Pipelines compare equal (``==``) when their steps match, serialize to the same JSON
    the Rust crate uses (:meth:`to_json`), and pickle.

    Examples
    --------
    >>> p = (refimage.Pipeline()
    ...      .debayer("linear")
    ...      .warp(inverse, (1080, 1920), order=3, mode="nearest")
    ...      .to_luma()
    ...      .convert("u8"))
    >>> runner = p.compile(first_frame, color_space="rggb")
    >>> out = runner.run(first_frame)
    >>> for frame in frames:
    ...     runner.run(frame, out=out)   # no pixel allocation per frame
    """

    def __new__(cls) -> Pipeline:
        """Create an empty (identity) pipeline."""

    def debayer(self, method: Literal["none", "nearest", "linear", "cubic"] = "linear") -> Pipeline:
        """Demosaic a single-channel Bayer image to RGB.

        ``"none"`` places each sample in its own channel without interpolating;
        ``"nearest"``, ``"linear"`` and ``"cubic"`` interpolate the missing samples.
        """

    def to_luma(self, weights: Sequence[float] | None = None) -> Pipeline:
        """Convert to gray luma.

        Uses Rec.601 weights (0.299, 0.587, 0.114) on RGB, or one weight per channel
        from ``weights``. A gray image passes through unchanged; Bayer data is an error
        (debayer first).
        """

    def scale(self, gain: float, offset: float = 0.0) -> Pipeline:
        """Apply ``y = x * gain + offset`` to the raw stored value.

        The result is saturated into the pixel type (``[0, 1]`` for ``float32``).
        """

    def scale_rational(self, num: int, den: int) -> Pipeline:
        """Multiply by ``num / den``.

        On integer images this is ``round(x * num / den)`` in exact integer
        arithmetic, e.g. ``scale_rational(65535, 4095)`` expands 12-bit data to the
        full 16-bit range with no rounding error. ``den`` must be non-zero.
        """

    def scale_by(self, factor: float) -> Pipeline:
        """Multiply every pixel by ``factor``, saturating into the pixel type."""

    def convert(self, pixel_type: Literal["u8", "u16", "f32"] | Any) -> Pipeline:
        """Rescale into another storage type (``"u8"``, ``"u16"``, ``"f32"`` or a numpy dtype).

        Values are rescaled between full-scale ranges: a 12-bit-tagged image converts
        to ``u8`` against 4095, and ``f32`` uses ``[0, 1]``.
        """

    def crop(self, x: int, y: int, width: int, height: int) -> Pipeline:
        """Keep the ``width`` x ``height`` rectangle whose top-left corner is ``(x, y)``.

        The rectangle must lie inside the image. On Bayer data the pattern is re-phased
        for an odd origin, so a crop may come before ``debayer``.
        """

    def roi(self, x: int, y: int, width: int, height: int) -> Pipeline:
        """Like :meth:`crop`, but the rectangle may extend past the edge; missing pixels are zero.

        Only the origin ``(x, y)`` must be inside the image.
        """

    def flip_horizontal(self) -> Pipeline:
        """Mirror left to right (Bayer patterns are re-phased)."""

    def flip_vertical(self) -> Pipeline:
        """Mirror top to bottom (Bayer patterns are re-phased)."""

    def rotate_90(self) -> Pipeline:
        """Rotate 90 degrees clockwise; width and height swap. Not valid on Bayer data."""

    def rotate_180(self) -> Pipeline:
        """Rotate 180 degrees (Bayer patterns are re-phased)."""

    def rotate_270(self) -> Pipeline:
        """Rotate 90 degrees counter-clockwise; width and height swap. Not valid on Bayer data."""

    def resize_to_fit(
        self,
        max_width: int,
        max_height: int,
        filter: Literal["bilinear", "bicubic", "lanczos3"] = "bilinear",
    ) -> Pipeline:
        """Resample to the largest size that fits ``max_width`` x ``max_height`` at the same aspect ratio.

        Enlarges images smaller than the box. ``"bilinear"`` is fastest,
        ``"bicubic"`` (Catmull-Rom) sharper, ``"lanczos3"`` highest fidelity with the
        most ringing. Not valid on Bayer data. For an exact (aspect-changing) size, use
        :meth:`warp` with a scaling matrix.
        """

    def warp(
        self,
        inverse_map: npt.ArrayLike,
        output_shape: tuple[int, int],
        order: SplineOrder = 1,
        mode: BoundaryModeName = "constant",
        cval: float = 0.0,
        prefilter: bool = True,
    ) -> Pipeline:
        """Resample through a 3x3 homography into an image of ``output_shape`` ``(rows, cols)``.

        Arguments as for :meth:`Image.warp`, except that ``output_shape`` is required
        (the step must know its size before it sees an image). The cubic prefilter's
        working memory is allocated when the pipeline is compiled, not per frame.

        Raises
        ------
        RefImageError
            Immediately for a malformed matrix, order or mode; at compile time for a
            non-finite matrix or Bayer input.
        """

    def map_coordinates(
        self,
        coordinates: npt.ArrayLike,
        order: SplineOrder = 1,
        mode: BoundaryModeName = "constant",
        cval: float = 0.0,
        prefilter: bool = True,
    ) -> Pipeline:
        """Sample at explicit positions; ``coordinates`` has shape ``(2, rows, cols)``.

        Arguments as for :meth:`Image.map_coordinates`. The coordinates are copied into
        the pipeline (and its JSON), so large coordinate maps make large pipelines.
        """

    def optimize(self) -> Pipeline:
        """Return an equivalent pipeline with redundant steps removed.

        Removes no-ops, folds runs of flips and 90-degree rotations into at most two
        steps, merges nested crops, and drops repeated luma or same-type conversions.
        The output is unchanged for every input the original accepts.
        """

    @property
    def ops(self) -> list[Any]:
        """The steps as JSON-style objects, e.g. ``[{"Debayer": "Linear"}, "ToLuma"]``."""

    def to_json(self) -> str:
        """Serialize the recipe to JSON (the same representation the Rust crate uses)."""

    @staticmethod
    def from_json(json: str) -> Pipeline:
        """Load a recipe written by :meth:`to_json` (or by Rust's ``serde_json``)."""

    def __len__(self) -> int:
        """The number of steps."""

    def apply(self, image: Image | PixelArray, *, color_space: ColorSpace | None = None) -> Image:
        """Run the pipeline once and return a new image.

        Parameters
        ----------
        image
            An :class:`Image` (its metadata is copied to the result) or a numpy array.
        color_space
            Color space of a numpy ``image``, as for :class:`Image`.

        Raises
        ------
        RefImageError
            If the chain is invalid for this input.
        """

    def compile(
        self,
        like: Image | ImageSpec | PixelArray,
        *,
        color_space: ColorSpace | None = None,
        strategy: Literal["tiled", "sequential"] = "tiled",
        tile_rows: int = 0,
        tile_cols: int = 0,
        parallel: bool = True,
    ) -> Runner:
        """Validate the chain for a frame format and allocate every buffer it needs.

        Parameters
        ----------
        like
            A sample :class:`Image` or numpy array, or an :class:`ImageSpec`; only its
            shape, color space and pixel type are used.
        color_space
            Color space of a numpy ``like``, as for :class:`Image`.
        strategy
            ``"tiled"`` cuts each frame into cache-sized bands and runs the whole chain
            per band; ``"sequential"`` runs every step over the whole frame.
        tile_rows
            Rows per band (``0`` picks a size automatically).
        tile_cols
            Columns per tile within a band (``0`` keeps bands full-width).
        parallel
            Process bands on multiple threads. With ``strategy="tiled",
            parallel=False`` a compiled runner allocates no memory per frame.

        Returns
        -------
        Runner

        Raises
        ------
        RefImageError
            If the chain is invalid for this format.
        """

@final
class Runner:
    """A compiled :class:`Pipeline` holding its working buffers.

    Create one with :meth:`Pipeline.compile` and call :meth:`run` (or the runner itself)
    for every frame. If a frame's shape differs from :attr:`input_spec`, the runner
    recompiles itself automatically (which allocates).
    """

    def run(self, frame: Image | PixelArray, out: Image | None = None) -> Image:
        """Process one frame.

        Parameters
        ----------
        frame
            An :class:`Image`, or a numpy array of the compiled shape and dtype. A
            C-contiguous array is read in place without copying; other layouts are
            copied first. An array takes the compiled color space and bit-depth tag.
        out
            Image to write the result into. If it already has the output shape and
            pixel type, its pixel buffer is reused (no allocation) and existing numpy
            views of it see the new frame. Otherwise it is resized, which raises
            :class:`BufferError` while numpy views of it exist. ``out`` may be
            ``frame`` itself.

        Returns
        -------
        Image
            ``out`` if given, otherwise a new image. When ``frame`` is an
            :class:`Image`, the result gets a copy of its metadata; a result from a
            numpy frame gets the current time and zero exposure (an existing ``out``
            keeps its own metadata).
        """

    def __call__(self, frame: Image | PixelArray, out: Image | None = None) -> Image:
        """Same as :meth:`run`."""

    @property
    def input_spec(self) -> ImageSpec:
        """The frame format the runner is currently compiled for."""

    @property
    def output_spec(self) -> ImageSpec:
        """The format :meth:`run` produces."""

    @property
    def is_tiled(self) -> bool:
        """Whether frames are processed in tiles (``False`` when tiling cannot help)."""

    @property
    def tiled_pass_count(self) -> int:
        """Independently tiled passes: 0 when untiled, more than 1 when a geometric step
        (rotation, crop, resize, warp) splits the chain and both sides are tiled."""

    @property
    def scratch_bytes(self) -> int:
        """Working memory held by the runner, in bytes."""

    def recompile(self, like: Image | ImageSpec | PixelArray, *, color_space: ColorSpace | None = None) -> None:
        """Rebuild the buffers for a new frame format (arguments as for :meth:`Pipeline.compile`)."""

@final
class ExposureResult:
    """The outcome of :meth:`OptimumExposure.calculate`."""

    @property
    def exposure(self) -> float:
        """Recommended exposure for the next frame, in seconds."""

    @property
    def bin(self) -> int:
        """Recommended binning (at least 1)."""

    @property
    def measured(self) -> float:
        """The measured percentile value, as a fraction of full scale."""

    @property
    def within_target(self) -> bool:
        """The measurement was already within tolerance; exposure and bin are unchanged."""

    @property
    def clamped(self) -> bool:
        """The recommended exposure hit the ``min_exposure`` or ``max_exposure`` bound."""

@final
class OptimumExposure:
    """Automatic exposure: find the exposure that brings a chosen pixel percentile to a target level.

    The ``percentile_pix`` pixel value is measured (ignoring the ``pixel_exclusion``
    brightest pixels, e.g. hot pixels and cosmic rays) and the exposure is scaled
    linearly so that value lands at ``pixel_tgt`` of full scale. If it is already within
    ``pixel_uncertainty`` the exposure is kept. With ``max_bin > 1``, binning is traded
    against exposure to stay below ``max_exposure``.

    Examples
    --------
    >>> auto = refimage.OptimumExposure(max_exposure=5.0)
    >>> res = auto.calculate(img)          # uses img.exposure
    >>> camera.set_exposure(res.exposure)
    """

    def __new__(
        cls,
        *,
        percentile_pix: float = 0.995,
        pixel_tgt: float = 40000 / 65536,
        pixel_uncertainty: float = 5000 / 65536,
        pixel_exclusion: int = 100,
        min_exposure: ExposureLike | None = None,
        max_exposure: ExposureLike | None = None,
        max_bin: int = 1,
    ) -> OptimumExposure:
        """Configure the calculator.

        Parameters
        ----------
        percentile_pix
            Which pixel to target, as a fraction (``0.995`` is the 99.5th percentile).
        pixel_tgt
            Target value for that pixel, as a fraction of full scale.
        pixel_uncertainty
            Tolerance around ``pixel_tgt``, as a fraction of full scale.
        pixel_exclusion
            Number of brightest pixels to ignore.
        min_exposure, max_exposure
            Exposure bounds in seconds or as ``timedelta``; default 1 ms and 10 s.
        max_bin
            Largest binning factor the result may recommend.

        Raises
        ------
        RefImageError
            For fractions outside ``[0, 1]`` or ``min_exposure > max_exposure``.
        """

    def calculate(
        self,
        image: Image | PixelArray,
        exposure: ExposureLike | None = None,
        bin: int = 1,
        *,
        in_place: bool = False,
    ) -> ExposureResult:
        """Evaluate a frame and recommend the next exposure and binning.

        Parameters
        ----------
        image
            An :class:`Image` or a numpy array.
        exposure
            Exposure the frame was taken with; defaults to ``image.exposure`` and is
            required for a numpy array.
        bin
            Binning the frame was taken with.
        in_place
            Skip copying the pixels. The percentile search partially **reorders the
            pixels** of an :class:`Image`, so use this only on a frame you are about
            to discard.

        Raises
        ------
        RefImageError
            For a zero exposure, an empty image, or a ``pixel_exclusion`` that leaves
            no pixels.
        """
