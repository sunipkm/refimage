"""refimage-pyo3: refimage images and pipelines through another extension module's Rust
(the test extension in tests/ext), as the refimage package's own objects, without copies.

Build the extension first (``maturin develop -m crates/refimage-pyo3/tests/ext/Cargo.toml``,
or ``maturin build`` and install the wheel) beside refimage.
"""

from __future__ import annotations

import copy
import os
import gc
import pickle
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import refimage
from refimage import Image, Pipeline, RefImageError

ext = pytest.importorskip("refimage_pyo3_test")

KINDS = ["u8", "u12", "rgbf32"]


def ptr(img: Image) -> int:
    """Address of the image's pixel buffer."""
    return np.asarray(img).ctypes.data


# --- the package check -------------------------------------------------------------------


def test_check():
    assert ext.check() == refimage.__version__


def test_protocol_and_version_rule():
    assert refimage._native._interop_protocol() == 1


# --- Rust -> Python, by value: the buffer moves, nothing is copied -------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_rust_image_is_a_refimage_image_on_rusts_buffer(kind):
    img, address = ext.capture(64, 48, kind, 7)
    assert type(img) is Image  # the package's own class, not a look-alike
    assert ptr(img) == address  # the Vec Rust allocated, adopted as is
    want_shape = [48, 64, 3] if kind == "rgbf32" else [48, 64]
    assert img.shape == want_shape
    assert img.pixel_type == {"u8": "u8", "u12": "u12", "rgbf32": "f32"}[kind]
    assert img.color_space == {"u8": "gray", "u12": "rggb", "rgbf32": "rgb"}[kind]


def test_metadata_crosses_exactly():
    img, _ = ext.capture(16, 8, "u12", 7)
    assert img.exposure == 0.25
    assert img.frame_id == 7
    assert img.timestamp == datetime(2023, 11, 14, 22, 13, 20, 123000, tzinfo=timezone.utc)
    assert img.get_key("CAMERA") == "ext"
    assert img.get_comment("CAMERA") == "made in Rust"
    assert img.get_key("GAIN") == 120
    assert img.bit_depth == 12


def test_pixel_values_cross_exactly():
    img, _ = ext.capture(10, 4, "u12", 0)
    want = (np.arange(40, dtype=np.uint64) * 7 % 4096).astype(np.uint16).reshape(4, 10)
    np.testing.assert_array_equal(np.asarray(img), want)


@pytest.mark.parametrize("kind", KINDS)
def test_adopted_image_supports_the_whole_api(kind, tmp_path):
    img, address = ext.capture(32, 24, kind, 3)
    # Copies are deep and owned; equality compares contents.
    dup = img.copy()
    assert dup == img and ptr(dup) != address
    assert copy.deepcopy(img) == img
    assert Image.from_bytes(img.to_bytes()) == img
    assert pickle.loads(pickle.dumps(img)) == img
    # Metadata edits on the adopted image.
    img.set_key("OBJECT", "M42", "target")
    img.exposure = 0.5
    assert img.get_key("OBJECT") == "M42" and img.exposure == 0.5
    # Processing reads the adopted buffer.
    p = Pipeline().flip_horizontal()
    np.testing.assert_array_equal(np.asarray(p.apply(img)), np.asarray(img)[:, ::-1])
    if img.color_space != "rggb":  # warps need demosaiced data
        np.testing.assert_array_equal(
            np.asarray(img.warp(np.eye(3), order=0)), np.asarray(img)
        )
    # FITS output.
    assert img.to_fits_bytes()[:9] == b"SIMPLE  ="
    path = img.write_fits(tmp_path / "f.fits")
    assert path.exists()


def test_numpy_views_write_through_to_rust():
    img, address = ext.capture(16, 8, "u8", 0)
    view = np.asarray(img)
    view[0, 0] = 200
    assert ext.data_ptr(img) == address  # Rust sees the same memory ...
    assert ext.total(img) == float(view.astype(np.float64).sum())  # ... with the edit


def test_runner_writes_into_an_adopted_buffer_in_place():
    img, address = ext.capture(32, 24, "u8", 0)
    src = Image(np.full((24, 32), 9, np.uint8))
    runner = Pipeline().scale(2.0).compile(src, strategy="tiled", parallel=False)
    out = runner.run(src, out=img)
    assert out is img
    assert ptr(img) == address  # same shape: written into the adopted buffer
    assert (np.asarray(img) == 18).all()


def test_runner_reallocates_an_adopted_image_only_without_views():
    img, address = ext.capture(32, 24, "u8", 0)
    runner = Pipeline().crop(0, 0, 8, 8).compile(img)
    view = np.asarray(img)
    with pytest.raises(BufferError):
        runner.run(img, out=img)
    del view
    runner.run(img, out=img)
    assert img.shape == [8, 8]
    assert ptr(img) != address  # now a buffer of refimage's own


def test_exposure_in_place_on_an_adopted_image():
    img, _ = ext.capture(64, 64, "u12", 0)
    res = refimage.OptimumExposure(pixel_exclusion=10).calculate(img, in_place=True)
    assert res.exposure > 0


def test_adopted_memory_outlives_the_image_through_views():
    img, _ = ext.capture(16, 8, "u12", 0)
    view = np.asarray(img)
    want = view.copy()
    del img
    gc.collect()
    np.testing.assert_array_equal(view, want)


def rss_bytes() -> int:
    """Resident memory of this process (Linux)."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


@pytest.mark.skipif(not os.path.exists("/proc/self/statm"), reason="needs /proc")
def test_frames_free_their_buffers():
    # 150 frames of 8 MiB: a leak would add ~1.2 GiB. Each adopted buffer is released by
    # the extension's own allocator when its image goes.
    for i in range(5):
        ext.capture(2048, 2048, "u12", i)
    gc.collect()
    before = rss_bytes()
    for i in range(150):
        img, _ = ext.capture(2048, 2048, "u12", i)
        assert img.frame_id == i
        del img
    gc.collect()
    assert rss_bytes() - before < 64 << 20


# --- Python -> Rust ------------------------------------------------------------------------


def test_rust_reads_and_writes_a_python_image_in_place():
    data = np.arange(20, dtype=np.uint16).reshape(4, 5)
    img = Image(data, "rggb", bit_depth=12)
    assert ext.data_ptr(img) == ptr(img)  # lent, not copied
    ext.invert(img)
    np.testing.assert_array_equal(np.asarray(img), 4095 - data)


def test_owned_extraction_copies_once():
    img, address = ext.capture(16, 8, "u12", 5)
    back = ext.echo(img)
    assert back == img
    assert ptr(back) != address  # the Rust side took its own copy
    again = ext.echo_by_ref(img)
    assert again == img


def test_set_metadata_from_rust():
    img = Image(np.zeros((4, 4), np.uint8))
    ext.stamp(img, 99)
    assert img.frame_id == 99
    assert img.get_key("STAMPED") == 1


# --- Pipelines -----------------------------------------------------------------------------


def test_pipelines_cross_both_ways():
    p = Pipeline().debayer("linear").to_luma().convert("u8")
    q = ext.flipped(p)
    assert type(q) is Pipeline
    assert len(q) == len(p) + 1
    assert q.ops[:-1] == p.ops and q.ops[-1] == "FlipHorizontal"


def test_rust_applies_a_python_pipeline_to_a_lent_image():
    raw = (np.arange(48 * 64) % 4096).astype(np.uint16).reshape(48, 64)
    img = Image(raw, "rggb", bit_depth=12, exposure=timedelta(milliseconds=5))
    p = Pipeline().debayer("linear").to_luma().convert("u8")
    want = p.apply(img)
    assert ext.process(p, img) == want
    ran = ext.run(p, img)
    np.testing.assert_array_equal(np.asarray(ran), np.asarray(want))
    assert ran.exposure == img.exposure


def test_warp_pipelines_keep_their_coefficients():
    m = np.array([[1.0, 0.1, -2.5], [0.05, 0.9, 1.25], [0.0, 0.0, 1.0]])
    p = Pipeline().warp(m, (10, 12), order=3, mode="reflect")
    q = ext.flipped(p)
    assert q.ops[0] == p.ops[0]


# --- Errors --------------------------------------------------------------------------------


def test_errors_are_refimages_own():
    with pytest.raises(RefImageError, match="another extension"):
        ext.fail()


def test_non_images_are_rejected():
    with pytest.raises(TypeError, match="refimage.Image"):
        ext.invert(np.zeros((4, 4), np.uint8))
    with pytest.raises(TypeError, match="refimage.Image"):
        ext.total("not an image")
    with pytest.raises(TypeError, match="refimage.Pipeline"):
        ext.flipped([])


def test_malformed_parts_are_rejected():
    n = refimage._native
    parts = n._image_to_parts(Image(np.zeros((4, 5), np.uint16)))
    with pytest.raises(ValueError, match="uint16"):
        n._image_from_parts({"header": parts["header"], "pixels": np.zeros((4, 5), np.uint8)})
    with pytest.raises(ValueError, match="shaped"):
        n._image_from_parts({"header": parts["header"], "pixels": np.zeros((5, 4), np.uint16)})
    with pytest.raises(ValueError, match="missing"):
        n._image_from_parts({"header": parts["header"]})
    with pytest.raises(ValueError):
        n._image_from_parts({"header": b"junk", "pixels": parts["pixels"]})


def test_unsuitable_arrays_are_copied_not_adopted():
    n = refimage._native
    parts = n._image_to_parts(Image(np.zeros((4, 5), np.uint16)))
    strided = np.arange(40, dtype=np.uint16).reshape(4, 10)[:, ::2]
    img = n._image_from_parts({"header": parts["header"], "pixels": strided})
    np.testing.assert_array_equal(np.asarray(img), strided)
    readonly = np.arange(20, dtype=np.uint16).reshape(4, 5)
    readonly.flags.writeable = False
    img = n._image_from_parts({"header": parts["header"], "pixels": readonly})
    assert ptr(img) != readonly.ctypes.data
    np.testing.assert_array_equal(np.asarray(img), readonly)
