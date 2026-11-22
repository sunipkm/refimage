/* End-to-end test of refimage.h, built and run by tests/c_api.rs. */

#include "refimage.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

/* A NULL the compiler cannot see, for checking that _Nonnull parameters are still
 * guarded at run time (a literal NULL there is a compile-time warning with Clang). */
static void *volatile opaque_null = NULL;
#define NULLPTR(T) ((T)opaque_null)

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,     \
                    #cond);                                                      \
            failures++;                                                          \
        }                                                                        \
    } while (0)

/* A call that must succeed. */
#define OK(call)                                                                 \
    do {                                                                         \
        RefStatus s_ = (call);                                                   \
        if (s_ != REF_OK) {                                                      \
            fprintf(stderr, "%s:%d: %s -> %s: %s\n", __FILE__, __LINE__, #call,  \
                    ref_status_name(s_), ref_last_error());                      \
            failures++;                                                          \
        }                                                                        \
    } while (0)

/* A call that must fail with `want`, leaving a message. */
#define FAILS(want, call)                                                        \
    do {                                                                         \
        RefStatus s_ = (call);                                                   \
        if (s_ != (want)) {                                                      \
            fprintf(stderr, "%s:%d: %s -> %s, expected %s\n", __FILE__,          \
                    __LINE__, #call, ref_status_name(s_), ref_status_name(want)); \
            failures++;                                                          \
        } else if (ref_last_error()[0] == '\0') {                                \
            fprintf(stderr, "%s:%d: %s set no error message\n", __FILE__,        \
                    __LINE__, #call);                                            \
            failures++;                                                          \
        }                                                                        \
    } while (0)

static RefFrame frame_of(const void *data, size_t w, size_t h, RefPixelType pt, RefColor color) {
    RefFrame f;
    memset(&f, 0, sizeof f);
    f.data = data;
    f.width = w;
    f.height = h;
    f.pixel_type = pt;
    f.color = color;
    return f;
}

/* Same shape and pixel bytes (metadata ignored). */
static bool same_pixels(const RefImage *a, const RefImage *b) {
    const void *pa = NULL, *pb = NULL;
    size_t na = 0, nb = 0;
    if (ref_image_data_const(a, &pa, &na) != REF_OK || ref_image_data_const(b, &pb, &nb) != REF_OK)
        return false;
    return na == nb && memcmp(pa, pb, na) == 0;
}

static void test_library(void) {
    CHECK(ref_abi_version() == REF_ABI_VERSION);
    CHECK(ref_version()[0] >= '0' && ref_version()[0] <= '9'); /* "X.Y.Z..." */
    CHECK(strcmp(ref_status_name(REF_ERR_IO), "REF_ERR_IO") == 0);
    CHECK(strcmp(ref_status_name(12345), "unknown status") == 0);
}

static void test_image_basics(void) {
    uint16_t px[4 * 3];
    for (int i = 0; i < 12; i++) px[i] = (uint16_t)(i * 100);
    RefFrame f = frame_of(px, 4, 3, REF_PIXEL_U12, REF_COLOR_BAYER_RGGB);

    RefImage *img = NULL;
    OK(ref_image_new(&f, &img));
    CHECK(img != NULL);

    RefImageInfo info;
    OK(ref_image_info(img, &info));
    CHECK(info.width == 4 && info.height == 3 && info.channels == 1);
    CHECK(info.pixel_type == REF_PIXEL_U12 && info.bit_depth == 12);
    CHECK(info.color == REF_COLOR_BAYER_RGGB);
    CHECK(info.row_bytes == 8 && info.nbytes == 24);

    /* The pixels were copied. */
    const void *data = NULL;
    size_t n = 0;
    OK(ref_image_data_const(img, &data, &n));
    CHECK(n == sizeof px && memcmp(data, px, n) == 0);
    px[0] = 999;
    CHECK(((const uint16_t *)data)[0] == 0);

    /* Writable view. */
    void *wdata = NULL;
    OK(ref_image_data(img, &wdata, NULL));
    ((uint16_t *)wdata)[1] = 4095;

    char name[16];
    size_t len = 0;
    OK(ref_image_color_name(img, name, sizeof name, &len));
    CHECK(strcmp(name, "rggb") == 0 && len == 4);
    OK(ref_image_color_name(img, NULL, 0, &len));
    CHECK(len == 4);
    FAILS(REF_ERR_BUFFER_TOO_SMALL, ref_image_color_name(img, name, 4, &len));

    /* Re-tagging 16-bit storage. */
    OK(ref_image_set_pixel_type(img, REF_PIXEL_U16));
    OK(ref_image_info(img, &info));
    CHECK(info.pixel_type == REF_PIXEL_U16);
    FAILS(REF_ERR_INVALID_ARG, ref_image_set_pixel_type(img, REF_PIXEL_U8));

    /* Standard metadata. */
    OK(ref_image_set_exposure(img, 0.25));
    double exp = 0;
    OK(ref_image_exposure(img, &exp));
    CHECK(exp == 0.25);
    FAILS(REF_ERR_INVALID_ARG, ref_image_set_exposure(img, -1.0));
    int64_t ts = 1700000000123456789LL;
    OK(ref_image_set_timestamp(img, ts));
    int64_t ts2 = 0;
    OK(ref_image_timestamp(img, &ts2));
    CHECK(ts2 == ts);
    uint32_t id = 0;
    FAILS(REF_ERR_NOT_FOUND, ref_image_frame_id(img, &id));
    OK(ref_image_set_frame_id(img, 42));
    OK(ref_image_frame_id(img, &id));
    CHECK(id == 42);

    /* A zero-filled image from a shape, and a custom color space. */
    RefImage *z = NULL;
    RefFrame zf = frame_of(NULL, 5, 2, REF_PIXEL_F32, REF_COLOR_CUSTOM);
    zf.channels = 2;
    zf.color_name = "uv";
    OK(ref_image_new(&zf, &z));
    OK(ref_image_info(z, &info));
    CHECK(info.channels == 2 && info.color == REF_COLOR_CUSTOM && info.nbytes == 5 * 2 * 2 * 4);
    OK(ref_image_color_name(z, name, sizeof name, NULL));
    CHECK(strcmp(name, "uv") == 0);
    OK(ref_image_data_const(z, &data, &n));
    for (size_t i = 0; i < n / 4; i++) CHECK(((const float *)data)[i] == 0.0f);
    ref_image_free(z);

    /* Copy, equality. */
    RefImage *c = NULL;
    OK(ref_image_clone(img, &c));
    bool eq = false;
    OK(ref_image_equal(img, c, &eq));
    CHECK(eq);
    OK(ref_image_set_frame_id(c, 7));
    OK(ref_image_equal(img, c, &eq));
    CHECK(!eq);
    OK(ref_image_copy_from(c, img));
    OK(ref_image_equal(img, c, &eq));
    CHECK(eq);
    ref_image_free(c);

    ref_image_free(img);
    ref_image_free(NULL);
}

static void test_padded_frames(void) {
    /* 3x2 u8 RGB rows padded to 16 bytes. */
    uint8_t buf[2 * 16];
    memset(buf, 0xEE, sizeof buf);
    for (int y = 0; y < 2; y++)
        for (int x = 0; x < 9; x++) buf[y * 16 + x] = (uint8_t)(y * 9 + x);
    RefFrame f = frame_of(buf, 3, 2, REF_PIXEL_U8, REF_COLOR_RGB);
    f.row_stride = 16;
    RefImage *img = NULL;
    OK(ref_image_new(&f, &img));
    const uint8_t *data = NULL;
    size_t n = 0;
    OK(ref_image_data_const(img, (const void **)&data, &n));
    CHECK(n == 18);
    for (int i = 0; i < 18; i++) CHECK(data[i] == i);
    ref_image_free(img);

    f.row_stride = 4; /* shorter than a row */
    FAILS(REF_ERR_INVALID_ARG, ref_image_new(&f, &img));
}

static void test_metadata(void) {
    uint8_t px[4] = {1, 2, 3, 4};
    RefFrame f = frame_of(px, 2, 2, REF_PIXEL_U8, REF_COLOR_GRAY);
    RefImage *img = NULL;
    OK(ref_image_new(&f, &img));

    OK(ref_image_set_int(img, "gain", 120, "camera gain"));
    OK(ref_image_set_real(img, "temp", -10.5, NULL));
    OK(ref_image_set_string(img, "camera", "ZWO", "model"));
    OK(ref_image_set_timestamp_key(img, "dateobs", 1000000000LL, NULL));
    OK(ref_image_set_duration(img, "readout", 0.125, NULL));
    OK(ref_image_set_int(img, "gain", 200, NULL)); /* replace */

    size_t count = 0;
    OK(ref_image_key_count(img, &count));
    CHECK(count >= 5);

    int64_t i = 0;
    OK(ref_image_get_int(img, "GAIN", &i)); /* case-insensitive */
    CHECK(i == 200);
    double d = 0;
    OK(ref_image_get_real(img, "temp", &d));
    CHECK(d == -10.5);
    OK(ref_image_get_real(img, "gain", &d)); /* ints convert */
    CHECK(d == 200.0);
    FAILS(REF_ERR_TYPE, ref_image_get_int(img, "temp", &i));
    char s[32];
    size_t len = 0;
    OK(ref_image_get_string(img, "camera", s, sizeof s, &len));
    CHECK(strcmp(s, "ZWO") == 0 && len == 3);
    OK(ref_image_get_comment(img, "camera", s, sizeof s, NULL));
    CHECK(strcmp(s, "model") == 0);
    OK(ref_image_get_comment(img, "temp", s, sizeof s, NULL));
    CHECK(s[0] == '\0');
    int64_t ns = 0;
    OK(ref_image_get_timestamp(img, "dateobs", &ns));
    CHECK(ns == 1000000000LL);
    OK(ref_image_get_duration(img, "readout", &d));
    CHECK(d == 0.125);

    RefValueKind kind = 99;
    OK(ref_image_value_kind(img, "camera", &kind));
    CHECK(kind == REF_VALUE_STRING);
    OK(ref_image_value_kind(img, "readout", &kind));
    CHECK(kind == REF_VALUE_DURATION);

    /* Iterate the keys. */
    int saw_gain = 0;
    for (size_t k = 0; k < count; k++) {
        char key[96];
        OK(ref_image_key_at(img, k, key, sizeof key, NULL));
        if (strcmp(key, "GAIN") == 0) saw_gain = 1;
    }
    CHECK(saw_gain);
    FAILS(REF_ERR_NOT_FOUND, ref_image_key_at(img, count, s, sizeof s, NULL));

    OK(ref_image_remove_key(img, "temp"));
    FAILS(REF_ERR_NOT_FOUND, ref_image_remove_key(img, "temp"));
    FAILS(REF_ERR_NOT_FOUND, ref_image_get_real(img, "temp", &d));
    FAILS(REF_ERR_UTF8, ref_image_set_int(img, "\xff\xfe", 1, NULL));
    FAILS(REF_ERR_NULL, ref_image_set_int(img, NULLPTR(const char *), 1, NULL));

    /* Round trip through bincode keeps everything. */
    RefBuffer bytes = {0};
    OK(ref_image_to_bytes(img, &bytes));
    CHECK(bytes.data != NULL && bytes.len > 4);
    RefImage *back = NULL;
    OK(ref_image_from_bytes(bytes.data, bytes.len, &back));
    bool eq = false;
    OK(ref_image_equal(img, back, &eq));
    CHECK(eq);
    FAILS(REF_ERR_DECODE, ref_image_from_bytes(bytes.data, 3, &back));
    ref_buffer_free(&bytes);
    CHECK(bytes.data == NULL && bytes.len == 0);
    ref_buffer_free(&bytes); /* idempotent */
    ref_image_free(back);
    ref_image_free(img);
}

static void test_fits(const char *dir) {
    uint16_t px[16 * 8];
    for (int i = 0; i < 16 * 8; i++) px[i] = (uint16_t)(i * 37);
    RefFrame f = frame_of(px, 16, 8, REF_PIXEL_U16, REF_COLOR_GRAY);
    RefImage *img = NULL;
    OK(ref_image_new(&f, &img));
    OK(ref_image_set_string(img, "object", "M42", NULL));

    RefBuffer fits = {0};
    OK(ref_image_fits_bytes(img, NULL, &fits));
    CHECK(fits.len % 2880 == 0 && memcmp(fits.data, "SIMPLE  =", 9) == 0);
    ref_buffer_free(&fits);

    RefFitsOptions o = ref_fits_options_default();
    o.compression = REF_FITS_RICE;
    o.tile_rows = 4;
    OK(ref_image_fits_bytes(img, &o, &fits));
    CHECK(fits.len % 2880 == 0);
    ref_buffer_free(&fits);
    o.compression = REF_FITS_GZIP;
    o.gzip_level = 12;
    FAILS(REF_ERR_INVALID_ARG, ref_image_fits_bytes(img, &o, &fits));
    o.compression = 99;
    FAILS(REF_ERR_INVALID_ARG, ref_image_fits_bytes(img, &o, &fits));

    char path[1024];
    snprintf(path, sizeof path, "%s/capi_test.fits", dir);
    remove(path);
    OK(ref_image_write_fits(img, path, NULL, false));
    FAILS(REF_ERR_IO, ref_image_write_fits(img, path, NULL, false));
    OK(ref_image_write_fits(img, path, NULL, true));
    remove(path);
    ref_image_free(img);
}

static void test_pipeline_runner(void) {
    enum { W = 64, H = 48 };
    static uint16_t raw[W * H];
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) raw[y * W + x] = (uint16_t)((x * 977 + y * 131) % 4096);
    RefFrame f = frame_of(raw, W, H, REF_PIXEL_U12, REF_COLOR_BAYER_RGGB);

    RefPipeline *p = NULL;
    OK(ref_pipeline_new(&p));
    OK(ref_pipeline_debayer(p, REF_DEMOSAIC_LINEAR));
    OK(ref_pipeline_to_luma(p));
    OK(ref_pipeline_convert(p, REF_PIXEL_U8));
    FAILS(REF_ERR_INVALID_ARG, ref_pipeline_debayer(p, 17));
    FAILS(REF_ERR_INVALID_ARG, ref_pipeline_convert(p, 17));
    size_t len = 0;
    OK(ref_pipeline_len(p, &len));
    CHECK(len == 3);

    /* Serial tiling: allocation-free once compiled. */
    RefStrategy st = ref_strategy_default();
    st.parallel = false;
    st.tile_rows = 16;
    RefRunner *r = NULL;
    OK(ref_pipeline_compile(p, &f, &st, &r));
    RefImageInfo in, out;
    OK(ref_runner_input_info(r, &in));
    OK(ref_runner_output_info(r, &out));
    CHECK(in.width == W && in.pixel_type == REF_PIXEL_U12);
    CHECK(out.width == W && out.height == H && out.channels == 1 && out.pixel_type == REF_PIXEL_U8);
    bool tiled = false;
    OK(ref_runner_is_tiled(r, &tiled));
    CHECK(tiled);
    size_t scratch = 0;
    OK(ref_runner_scratch_bytes(r, &scratch));
    CHECK(scratch > 0);

    /* First run creates the output; later runs reuse its buffer. */
    RefImage *dst = NULL;
    OK(ref_runner_run(r, &f, &dst));
    CHECK(dst != NULL);
    const void *first = NULL;
    OK(ref_image_data_const(dst, &first, NULL));
    for (int k = 0; k < 5; k++) OK(ref_runner_run(r, &f, &dst));
    const void *again = NULL;
    OK(ref_image_data_const(dst, &again, NULL));
    CHECK(first == again);

    /* The same as a one-shot apply. */
    RefImage *src = NULL, *applied = NULL;
    OK(ref_image_new(&f, &src));
    OK(ref_pipeline_apply(p, src, &applied));
    const void *a = NULL, *b = NULL;
    size_t na = 0, nb = 0;
    OK(ref_image_data_const(applied, &a, &na));
    OK(ref_image_data_const(dst, &b, &nb));
    CHECK(na == nb && na == W * H && memcmp(a, b, na) == 0);

    /* run_into a padded caller buffer. */
    static uint8_t user[H * (W + 8)];
    memset(user, 0xAB, sizeof user);
    OK(ref_runner_run_into(r, &f, user, sizeof user, W + 8));
    for (int y = 0; y < H; y++) {
        CHECK(memcmp(user + y * (W + 8), (const uint8_t *)b + y * W, W) == 0);
        CHECK(user[y * (W + 8) + W] == 0xAB);
    }
    FAILS(REF_ERR_BUFFER_TOO_SMALL, ref_runner_run_into(r, &f, user, 10, 0));

    /* run_image keeps metadata; in place works too. */
    OK(ref_image_set_frame_id(src, 9));
    RefImage *meta_out = NULL;
    OK(ref_runner_run_image(r, src, &meta_out));
    uint32_t id = 0;
    OK(ref_image_frame_id(meta_out, &id));
    CHECK(id == 9);
    RefImage *inplace = NULL;
    OK(ref_image_clone(src, &inplace));
    OK(ref_runner_run_image(r, inplace, &inplace));
    bool eq = false;
    OK(ref_image_equal(inplace, meta_out, &eq));
    CHECK(eq);

    /* JSON round trip gives the same results. */
    RefBuffer json = {0};
    OK(ref_pipeline_to_json(p, &json));
    CHECK(json.data[json.len] == '\0' && strstr((const char *)json.data, "Debayer") != NULL);
    RefPipeline *p2 = NULL;
    OK(ref_pipeline_from_json((const char *)json.data, &p2));
    ref_buffer_free(&json);
    RefImage *applied2 = NULL;
    OK(ref_pipeline_apply(p2, src, &applied2));
    CHECK(same_pixels(applied, applied2));
    FAILS(REF_ERR_DECODE, ref_pipeline_from_json("{\"ops\": [42]}", &p2));

    /* A pipeline written directly as JSON. */
    RefPipeline *pj = NULL;
    OK(ref_pipeline_from_json("{\"ops\":[{\"Debayer\":\"Linear\"},\"ToLuma\",{\"Convert\":\"U8\"}]}", &pj));
    RefImage *applied3 = NULL;
    if (pj) {
        OK(ref_pipeline_apply(pj, src, &applied3));
        CHECK(same_pixels(applied, applied3));
    }

    /* Invalid for this input: luma weights for the wrong channel count. */
    RefPipeline *bad = NULL;
    double weights[2] = {0.5, 0.5};
    OK(ref_pipeline_new(&bad));
    OK(ref_pipeline_debayer(bad, REF_DEMOSAIC_LINEAR));
    OK(ref_pipeline_to_luma_weights(bad, weights, 2));
    RefRunner *rbad = NULL;
    FAILS(REF_ERR_PIPELINE, ref_pipeline_compile(bad, &f, NULL, &rbad));
    CHECK(rbad == NULL);

    ref_image_free(applied3);
    ref_pipeline_free(pj);
    ref_pipeline_free(bad);
    ref_image_free(applied2);
    ref_pipeline_free(p2);
    ref_image_free(inplace);
    ref_image_free(meta_out);
    ref_image_free(applied);
    ref_image_free(src);
    ref_image_free(dst);
    ref_runner_free(r);
    ref_pipeline_free(p);
}

static void test_geometry(void) {
    enum { W = 8, H = 6 };
    float px[W * H];
    for (int i = 0; i < W * H; i++) px[i] = (float)i / (W * H);
    RefFrame f = frame_of(px, W, H, REF_PIXEL_F32, REF_COLOR_GRAY);
    RefImage *img = NULL;
    OK(ref_image_new(&f, &img));

    /* Shift one column right: output (c, r) reads input (c - 1, r). */
    const double shift[9] = {1, 0, -1, 0, 1, 0, 0, 0, 1};
    RefInterpolation interp = ref_interpolation_default();
    interp.order = 1;
    RefPipeline *pw = NULL;
    OK(ref_pipeline_new(&pw));
    OK(ref_pipeline_warp(pw, shift, W, H, &interp));
    RefImage *warped = NULL;
    OK(ref_pipeline_apply(pw, img, &warped));
    const float *w = NULL;
    OK(ref_image_data_const(warped, (const void **)&w, NULL));
    for (int y = 0; y < H; y++) {
        CHECK(w[y * W] == 0.0f); /* cval */
        for (int x = 1; x < W; x++) CHECK(fabsf(w[y * W + x] - px[y * W + x - 1]) < 1e-6f);
    }

    /* The same through explicit coordinates. */
    double rows[W * H], cols[W * H];
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            rows[y * W + x] = y;
            cols[y * W + x] = x - 1;
        }
    RefPipeline *pm = NULL;
    OK(ref_pipeline_new(&pm));
    OK(ref_pipeline_map_coordinates(pm, rows, cols, W, H, NULL));
    RefImage *mapped = NULL;
    OK(ref_pipeline_apply(pm, img, &mapped));
    bool eq = false;
    OK(ref_image_equal(warped, mapped, &eq));
    CHECK(eq);

    interp.order = 2;
    FAILS(REF_ERR_INVALID_ARG, ref_pipeline_warp(pw, shift, W, H, &interp));
    interp.order = 1;
    interp.mode = 9;
    FAILS(REF_ERR_INVALID_ARG, ref_pipeline_warp(pw, shift, W, H, &interp));
    FAILS(REF_ERR_NULL, ref_pipeline_warp(pw, NULLPTR(const double *), W, H, NULL));

    /* Crop, flips, rotation, resize: a quick shape check. */
    RefPipeline *pg = NULL;
    OK(ref_pipeline_new(&pg));
    OK(ref_pipeline_crop(pg, 1, 1, 6, 4));
    OK(ref_pipeline_flip_horizontal(pg));
    OK(ref_pipeline_flip_horizontal(pg));
    OK(ref_pipeline_rotate_90(pg));
    OK(ref_pipeline_resize_to_fit(pg, 2, 3, REF_RESIZE_LANCZOS3));
    OK(ref_pipeline_scale(pg, 2.0, 0.0));
    OK(ref_pipeline_optimize(pg));
    size_t len = 0;
    OK(ref_pipeline_len(pg, &len));
    CHECK(len == 4); /* the flip pair folds away */
    RefImage *g = NULL;
    OK(ref_pipeline_apply(pg, img, &g));
    RefImageInfo info;
    OK(ref_image_info(g, &info));
    CHECK(info.width == 2 && info.height == 3);

    ref_image_free(g);
    ref_pipeline_free(pg);
    ref_image_free(mapped);
    ref_pipeline_free(pm);
    ref_image_free(warped);
    ref_pipeline_free(pw);
    ref_image_free(img);
}

static void test_exposure(void) {
    enum { N = 100 * 100 };
    static uint16_t px[N];
    for (int i = 0; i < N; i++) px[i] = (uint16_t)(i % 1000 * 10); /* dim */
    RefFrame f = frame_of(px, 100, 100, REF_PIXEL_U16, REF_COLOR_GRAY);
    RefExposureOptions o = ref_exposure_options_default();
    o.pixel_exclusion = 10;
    RefExposureResult res;

    OK(ref_exposure_calculate_frame(&o, &f, 0.1, 1, &res));
    CHECK(res.exposure > 0.1 && !res.within_target && res.bin == 1);
    FAILS(REF_ERR_INVALID_ARG, ref_exposure_calculate_frame(&o, &f, -1.0, 1, &res));

    RefImage *img = NULL;
    OK(ref_image_new(&f, &img));
    OK(ref_image_set_exposure(img, 0.1));
    RefExposureResult res2;
    OK(ref_exposure_calculate(&o, img, 0.0, 1, &res2)); /* image's own exposure */
    CHECK(res2.exposure == res.exposure && res2.measured == res.measured);
    o.pixel_exclusion = N; /* excludes every pixel */
    FAILS(REF_ERR_EXPOSURE, ref_exposure_calculate(&o, img, 0.0, 1, &res2));
    o.pixel_exclusion = 10;

    /* In place on a caller buffer. */
    OK(ref_exposure_calculate_buffer(&o, px, N, REF_PIXEL_U16, 0.1, 1, &res2));
    CHECK(res2.exposure == res.exposure);
    FAILS(REF_ERR_INVALID_ARG,
          ref_exposure_calculate_buffer(&o, (uint8_t *)px + 1, N - 1, REF_PIXEL_U16, 0.1, 1, &res2));
    OK(ref_exposure_calculate_in_place(NULL, img, 0.0, 1, &res2));
    ref_image_free(img);
}

static void test_errors(void) {
    RefImage *img = NULL;
    FAILS(REF_ERR_NULL, ref_image_new(NULLPTR(const RefFrame *), &img));
    uint8_t px[4] = {0};
    RefFrame f = frame_of(px, 2, 2, REF_PIXEL_U8, 42);
    FAILS(REF_ERR_INVALID_ARG, ref_image_new(&f, &img));
    f.color = REF_COLOR_GRAY;
    f.pixel_type = 42;
    FAILS(REF_ERR_INVALID_ARG, ref_image_new(&f, &img));
    f.pixel_type = REF_PIXEL_U8;
    f.width = 0;
    FAILS(REF_ERR_INVALID_ARG, ref_image_new(&f, &img));
    f.width = 2;
    FAILS(REF_ERR_NULL, ref_image_new(&f, NULLPTR(RefImage **)));
    CHECK(img == NULL);
    FAILS(REF_ERR_NULL, ref_image_info(NULLPTR(const RefImage *), NULLPTR(RefImageInfo *)));
    /* Messages are per thread and stay until the next failure. */
    OK(ref_image_new(&f, &img));
    CHECK(strstr(ref_last_error(), "NULL") != NULL);
    ref_image_free(img);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : ".";
    test_library();
    test_image_basics();
    test_padded_frames();
    test_metadata();
    test_fits(dir);
    test_pipeline_runner();
    test_geometry();
    test_exposure();
    test_errors();
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("all C API checks passed\n");
    return 0;
}
