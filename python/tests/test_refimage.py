"""refimage's Python bindings: Image, metadata, serialization, pipelines, warp, FITS."""

import copy
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import refimage
from refimage import Image, ImageSpec, OptimumExposure, Pipeline, RefImageError

MODES = ["constant", "nearest", "mirror", "reflect", "wrap"]
TS = datetime(2026, 11, 22, 12, 30, 15, 250000, tzinfo=timezone.utc)


def rotation(h, w, deg=12.0, shift=(3.0, -2.0)):
    """skimage-convention inverse map: rotate about the centre, then shift."""
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    return np.array(
        [
            [c, -s, cx - c * cx + s * cy + shift[0]],
            [s, c, cy - s * cx - c * cy + shift[1]],
            [0.0, 0.0, 1.0],
        ]
    )


# --- Image ---------------------------------------------------------------------------


def test_image_holds_pixels_and_metadata():
    data = np.arange(12, dtype=np.uint16).reshape(3, 4)
    img = Image(data, timestamp=TS, exposure=0.25, bit_depth=12, frame_id=7)
    assert (img.width, img.height, img.channels) == (4, 3, 1)
    assert img.shape == [3, 4] and img.dtype == np.uint16
    assert img.pixel_type == "u12" and img.bit_depth == 12
    assert img.color_space == "gray"
    assert img.timestamp == TS and img.exposure == 0.25 and img.frame_id == 7
    np.testing.assert_array_equal(np.asarray(img), data)

    img.exposure = timedelta(milliseconds=40)
    assert img.exposure == pytest.approx(0.04)
    rgb = Image(np.zeros((2, 5, 3), np.float32))
    assert rgb.color_space == "rgb" and rgb.shape == [2, 5, 3]
    custom = Image(np.zeros((2, 2, 4), np.uint8), (4, "rgba"))
    assert custom.color_space == (4, "rgba")
    assert Image(np.zeros((4, 4), np.uint16), "bayer_grbg").color_space == "grbg"


def test_image_rejects_bad_input():
    with pytest.raises(RefImageError, match="dtype"):
        Image(np.zeros((2, 2), np.int32))
    with pytest.raises(RefImageError, match="infer"):
        Image(np.zeros((2, 2, 4), np.uint8))
    with pytest.raises(RefImageError, match="channel"):
        Image(np.zeros((2, 2), np.uint8), "rgb")
    with pytest.raises(RefImageError, match="color space"):
        Image(np.zeros((2, 2), np.uint8), "cmyk")


def test_numpy_view_is_zero_copy_and_writable():
    img = Image(np.zeros((3, 4, 3), np.uint8))
    view = np.asarray(img)
    assert view.shape == (3, 4, 3) and view.flags.writeable and view.flags.c_contiguous
    view[1, 2] = (10, 20, 30)
    np.testing.assert_array_equal(img.numpy()[1, 2], (10, 20, 30))
    # The same memory backs every view.
    assert np.shares_memory(view, img.numpy())
    assert memoryview(img).nbytes == img.nbytes == 36


def test_non_contiguous_input_is_copied_in_row_major_order():
    base = np.arange(48, dtype=np.uint16).reshape(6, 8)
    img = Image(base[::2, 1::3])
    np.testing.assert_array_equal(np.asarray(img), base[::2, 1::3])


def test_metadata_cards():
    img = Image(np.zeros((2, 2), np.uint8))
    img.set_key("camera", "ZWO", comment="model")
    img.set_key("GAIN", 120)
    img.set_key("temp", -10.5)
    img.set_key("OBSTIME", TS)
    img.set_key("cadence", timedelta(seconds=1.5))
    assert img.get_key("CAMERA") == "ZWO" and img.get_comment("camera") == "model"
    assert img.get_key("gain") == 120 and img.get_key("TEMP") == -10.5
    assert img.get_key("OBSTIME") == TS
    assert img.get_key("CADENCE") == timedelta(seconds=1.5)
    assert img.get_key("missing", "dflt") == "dflt"
    img.set_key("GAIN", 200)  # replace
    assert img.metadata["GAIN"] == 200
    assert list(img.metadata)[:2] == ["CAMERA", "GAIN"]
    assert img.remove_key("gain") == 200 and img.get_key("GAIN") is None
    with pytest.raises(KeyError):
        img.remove_key("GAIN")
    with pytest.raises(RefImageError):
        img.set_key("EXPOSURE", 1.0)  # reserved: use the property
    with pytest.raises(TypeError):
        img.set_key("BAD", [1, 2])


def test_bytes_and_pickle_round_trip():
    img = Image(np.arange(30, dtype=np.float32).reshape(5, 6) / 30, timestamp=TS, exposure=1.5)
    img.set_key("OBJECT", "M42")
    for back in (
        Image.from_bytes(img.to_bytes()),
        pickle.loads(pickle.dumps(img)),
        copy.deepcopy(img),
        img.copy(),
    ):
        assert back == img
        assert back.get_key("OBJECT") == "M42" and back.timestamp == TS
        np.testing.assert_array_equal(np.asarray(back), np.asarray(img))
    with pytest.raises(RefImageError, match="serialized"):
        Image.from_bytes(b"nope")


# --- Pipeline / Runner ---------------------------------------------------------------


def bayer(h=64, w=96, seed=0):
    return np.random.default_rng(seed).integers(0, 4096, (h, w), dtype=np.uint16)


def test_pipeline_builders_are_immutable_and_serialize():
    base = Pipeline().debayer("cubic")
    longer = base.to_luma().convert(np.uint8).warp(np.eye(3), (10, 12), order=3, mode="edge")
    assert len(base) == 1 and len(longer) == 4
    assert Pipeline.from_json(longer.to_json()) == longer
    assert pickle.loads(pickle.dumps(longer)) == longer
    assert longer.ops[0] == {"Debayer": "Cubic"}
    assert longer.ops[3]["Warp"]["interp"]["mode"] == "Nearest"
    folded = Pipeline().rotate_90().rotate_90().flip_vertical().optimize()
    assert folded.ops == ["FlipHorizontal"]


def test_apply_matches_runner_and_keeps_metadata():
    img = Image(bayer(), "rggb", timestamp=TS, exposure=0.1, bit_depth=12)
    img.set_key("CAMERA", "x")
    p = Pipeline().debayer("linear").resize_to_fit(48, 48, "lanczos3").to_luma().convert("u8")
    once = p.apply(img)
    assert once.shape == [32, 48] and once.dtype == np.uint8
    assert once.get_key("CAMERA") == "x" and once.timestamp == TS
    for strategy, parallel in [("sequential", True), ("tiled", False), ("tiled", True)]:
        runner = p.compile(img, strategy=strategy, tile_rows=8, parallel=parallel)
        assert runner.output_spec.shape == [32, 48]
        np.testing.assert_array_equal(np.asarray(runner.run(img)), np.asarray(once))
    assert img.apply(p) == once


def test_run_into_out_reuses_its_buffer():
    p = Pipeline().debayer("linear").to_luma()
    frames = [bayer(seed=i) for i in range(4)]
    runner = p.compile(frames[0], color_space="rggb", parallel=False)
    assert runner.input_spec == ImageSpec(96, 64, "rggb", "u16")
    out = runner.run(frames[0])
    view = np.asarray(out)
    ptr = view.ctypes.data
    for f in frames[1:]:
        ret = runner.run(f, out=out)  # numpy frame: read in place, no copy
        assert ret is out
        assert np.asarray(out).ctypes.data == ptr  # same pixel buffer
        expected = p.apply(Image(f, "rggb"))
        np.testing.assert_array_equal(view, np.asarray(expected))  # live view sees it


def test_out_reshape_is_refused_while_viewed():
    runner = Pipeline().crop(0, 0, 4, 4).compile(np.zeros((8, 8), np.uint8))
    out = Image(np.zeros((2, 2), np.uint8))
    view = np.asarray(out)
    with pytest.raises(BufferError):
        runner.run(np.zeros((8, 8), np.uint8), out=out)
    del view
    runner.run(np.ones((8, 8), np.uint8), out=out)
    assert out.shape == [4, 4] and np.asarray(out).sum() == 16


def test_run_in_place_on_the_input_image():
    data = np.arange(20, dtype=np.uint8).reshape(4, 5)
    img = Image(data.copy())
    view = np.asarray(img)
    Pipeline().flip_horizontal().compile(img).run(img, out=img)
    np.testing.assert_array_equal(view, data[:, ::-1])


def test_runner_grows_on_a_new_shape_and_accepts_strided_frames():
    runner = Pipeline().scale_by(2.0).compile(ImageSpec(4, 4, "gray", "u8"))
    big = np.full((6, 9), 3, np.uint8)
    out = runner.run(big[:, ::1])
    assert out.shape == [6, 9] and runner.input_spec.width == 9
    strided = np.arange(36, dtype=np.uint8).reshape(6, 6)[::2, ::2]
    np.testing.assert_array_equal(np.asarray(runner.run(strided)), strided * 2)


def test_pipeline_errors_raise_refimage_error():
    with pytest.raises(RefImageError, match="Bayer"):
        Pipeline().to_luma().compile(np.zeros((8, 8), np.uint16), color_space="rggb")
    with pytest.raises(RefImageError, match="order"):
        Pipeline().warp(np.eye(3), (4, 4), order=2)
    with pytest.raises(RefImageError, match="boundary mode"):
        Pipeline().warp(np.eye(3), (4, 4), mode="bogus")
    with pytest.raises(RefImageError, match="3x3"):
        Pipeline().warp(np.eye(2), (4, 4))
    with pytest.raises(RefImageError, match=r"\(2, rows, cols\)"):
        Pipeline().map_coordinates(np.zeros((3, 2, 2)))
    with pytest.raises(RefImageError, match="strategy"):
        Pipeline().compile(np.zeros((4, 4), np.uint8), strategy="bogus")


def test_runs_release_the_gil():
    p = Pipeline().debayer("cubic").to_luma()
    frames = [bayer(256, 256, seed=i) for i in range(8)]

    def work(f):
        r = p.compile(f, color_space="rggb")
        return np.asarray(r.run(f)).sum()

    with ThreadPoolExecutor(4) as ex:
        sums = list(ex.map(work, frames))
    assert sums == [work(f) for f in frames]


# --- Warp / map_coordinates against SciPy and scikit-image ----------------------------

scipy_ndimage = pytest.importorskip("scipy.ndimage")
skimage_transform = pytest.importorskip("skimage.transform")


@pytest.mark.parametrize("order", [0, 1, 3])
@pytest.mark.parametrize("mode", MODES)
def test_map_coordinates_matches_scipy_on_float32(order, mode):
    rng = np.random.default_rng(order)
    image = rng.random((17, 23)).astype(np.float32)
    coords = rng.uniform(-6, 28, size=(2, 5, 7))
    ours = Image(image).map_coordinates(coords, order=order, mode=mode, cval=0.3)
    theirs = scipy_ndimage.map_coordinates(image.astype(np.float64), coords, order=order, mode=mode, cval=0.3)
    assert ours.shape == [5, 7]
    # float32 images saturate into [0, 1]; values are stored at float32 precision.
    np.testing.assert_allclose(np.asarray(ours), np.clip(theirs, 0, 1), rtol=0, atol=1e-6)


@pytest.mark.parametrize("order", [1, 3])
def test_map_coordinates_on_uint16_rounds_and_saturates(order):
    rng = np.random.default_rng(5)
    image = rng.integers(0, 65536, (20, 20), dtype=np.uint16)
    coords = rng.uniform(-3, 23, size=(2, 9, 11))
    ours = np.asarray(Image(image).map_coordinates(coords, order=order, mode="reflect"))
    theirs = scipy_ndimage.map_coordinates(image.astype(np.float64), coords, order=order, mode="reflect")
    want = np.clip(np.round(theirs), 0, 65535)
    assert ours.dtype == np.uint16
    np.testing.assert_allclose(ours, want, rtol=0, atol=1)  # ties may round either way


@pytest.mark.parametrize("order", [0, 1])
def test_warp_matches_skimage(order):
    image = np.random.default_rng(7).random((40, 50)).astype(np.float32)
    inverse = rotation(40, 50)
    ours = Image(image).warp(inverse, order=order, mode="edge")
    theirs = skimage_transform.warp(image.astype(np.float64), inverse, order=order, mode="edge", preserve_range=True)
    np.testing.assert_allclose(np.asarray(ours), theirs, rtol=0, atol=1e-6)


@pytest.mark.parametrize("order", [0, 1])
def test_skimage_symmetric_mode_is_scipy_reflect(order):
    # scikit-image's numpy.pad-style names: `symmetric` is scipy's `reflect`.
    image = np.random.default_rng(11).random((30, 40)).astype(np.float32)
    inverse = rotation(30, 40, deg=35.0)
    inverse[:2, 2] += 25.0  # push samples well past the edges
    ours = np.asarray(Image(image).warp(inverse, order=order, mode="symmetric"))
    theirs = skimage_transform.warp(
        image.astype(np.float64), inverse, order=order, mode="symmetric", preserve_range=True
    )
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(ours, np.asarray(Image(image).warp(inverse, order=order, mode="reflect")))


def test_cubic_warp_is_scipys_b_spline_per_channel():
    # scikit-image's own order-3 kernel is not a B-spline, so SciPy is the reference.
    rgb = np.random.default_rng(3).random((40, 50, 3)).astype(np.float32) * 0.5 + 0.25
    inverse = rotation(40, 50, deg=-23.0)
    matrix = inverse[:2, :2][::-1, ::-1]
    offset = inverse[:2, 2][::-1]
    ours = np.asarray(Image(rgb).warp(inverse, output_shape=(45, 38), order=3, mode="nearest"))
    assert ours.shape == (45, 38, 3)
    for c in range(3):
        theirs = scipy_ndimage.affine_transform(
            rgb[..., c].astype(np.float64), matrix, offset=offset, output_shape=(45, 38), order=3, mode="nearest"
        )
        np.testing.assert_allclose(ours[..., c], np.clip(theirs, 0, 1), rtol=0, atol=1e-6)


def test_warp_in_a_compiled_pipeline_matches_one_shot():
    raw = bayer(80, 96, seed=4)
    inverse = rotation(80, 96, deg=30.0)
    p = (
        Pipeline()
        .debayer("linear")
        .warp(inverse, (70, 90), order=3, mode="reflect")
        .to_luma()
        .convert("f32")
    )
    once = p.apply(raw, color_space="rggb")
    runner = p.compile(raw, color_space="rggb", tile_rows=8)
    assert runner.tiled_pass_count == 2
    np.testing.assert_array_equal(np.asarray(runner.run(raw)), np.asarray(once))


# --- FITS and exposure ---------------------------------------------------------------


@pytest.mark.parametrize("compression", [None, "gzip", "rice", "hcompress"])
def test_fits_round_trips_through_astropy(tmp_path, compression):
    fits = pytest.importorskip("astropy.io.fits")
    data = np.random.default_rng(1).integers(0, 4096, (32, 40), dtype=np.uint16)
    img = Image(data, timestamp=TS, exposure=0.5)
    img.set_key("OBJECT", "M42", comment="target")
    path = img.write_fits(tmp_path / "f.fits", compression)
    with fits.open(path) as hdul:
        hdu = hdul[1] if compression else hdul[0]
        np.testing.assert_array_equal(hdu.data, data)
        assert hdu.header["OBJECT"] == "M42"
    assert img.to_fits_bytes(compression) == path.read_bytes()
    with pytest.raises(FileExistsError):
        img.write_fits(path, compression)
    img.write_fits(path, compression, overwrite=True)


def test_optimum_exposure_leaves_pixels_alone_unless_in_place():
    rng = np.random.default_rng(2)
    data = rng.integers(0, 20000, (64, 64), dtype=np.uint16)
    img = Image(data, exposure=0.1)
    calc = OptimumExposure(pixel_exclusion=10, max_exposure=5.0)
    res = calc.calculate(img)
    np.testing.assert_array_equal(np.asarray(img), data)
    assert 0.1 < res.exposure <= 5.0 and res.bin == 1 and not res.within_target
    assert calc.calculate(data, exposure=0.1).exposure == res.exposure
    assert calc.calculate(img, in_place=True).exposure == res.exposure
    assert sorted(np.asarray(img).ravel().tolist()) == sorted(data.ravel().tolist())
    with pytest.raises(RefImageError):
        calc.calculate(data)  # a bare array needs an exposure


PUBLIC_CLASSES = [Image, ImageSpec, Pipeline, refimage.Runner, OptimumExposure, refimage.ExposureResult]


@pytest.mark.parametrize("cls", PUBLIC_CLASSES, ids=lambda c: c.__name__)
def test_every_public_member_is_documented(cls):
    """`help()` and IDEs need a docstring on every class, method and property."""
    assert (cls.__doc__ or "").strip(), cls.__name__
    undocumented = [
        name
        for name, member in vars(cls).items()
        if not name.startswith("_") and not (getattr(member, "__doc__", None) or "").strip()
    ]
    assert undocumented == []


def test_version():
    assert refimage.__version__
