/*
 * Process a stream of camera frames with refimage's C API: debayer, convert to
 * 8-bit luma and half-size, reusing every buffer, then save the last raw frame as
 * FITS and suggest the next exposure.
 *
 *   cc stream.c $(pkg-config --cflags --libs refimage) -o stream && ./stream
 */
#include <refimage.h>

#include <stdio.h>
#include <stdlib.h>

#define CHECK(call)                                                               \
    do {                                                                          \
        RefStatus s_ = (call);                                                    \
        if (s_ != REF_OK) {                                                       \
            fprintf(stderr, "%s failed: %s (%s)\n", #call, ref_last_error(),      \
                    ref_status_name(s_));                                         \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

enum { W = 1920, H = 1080 };

/* Stand-in for a camera SDK callback: a 12-bit RGGB frame. */
static void capture(uint16_t *buf, int n) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) buf[y * W + x] = (uint16_t)((x + y + n * 16) % 4096);
}

int main(void) {
    printf("refimage %s (ABI %u)\n", ref_version(), ref_abi_version());
    uint16_t *raw = malloc(sizeof(uint16_t) * W * H);

    RefFrame frame = {0};
    frame.data = raw;
    frame.width = W;
    frame.height = H;
    frame.pixel_type = REF_PIXEL_U12;
    frame.color = REF_COLOR_BAYER_RGGB;

    /* Build the recipe once ... */
    RefPipeline *p = NULL;
    CHECK(ref_pipeline_new(&p));
    CHECK(ref_pipeline_debayer(p, REF_DEMOSAIC_LINEAR));
    CHECK(ref_pipeline_to_luma(p));
    CHECK(ref_pipeline_resize_to_fit(p, W / 2, H / 2, REF_RESIZE_BILINEAR));
    CHECK(ref_pipeline_convert(p, REF_PIXEL_U8));

    /* ... compile it for the frame shape (allocates every buffer) ... */
    RefRunner *runner = NULL;
    CHECK(ref_pipeline_compile(p, &frame, NULL, &runner));
    RefImageInfo out;
    CHECK(ref_runner_output_info(runner, &out));
    printf("output %zux%zu, %zu bytes\n", out.width, out.height, out.nbytes);

    /* ... and run it for every frame, straight into a display buffer. */
    uint8_t *display = malloc(out.nbytes);
    for (int n = 0; n < 30; n++) {
        capture(raw, n);
        CHECK(ref_runner_run_into(runner, &frame, display, out.nbytes, 0));
    }
    printf("display[0] = %u\n", display[0]);

    /* Keep the last raw frame, with metadata, as compressed FITS. */
    RefImage *img = NULL;
    CHECK(ref_image_new(&frame, &img));
    CHECK(ref_image_set_exposure(img, 0.05));
    CHECK(ref_image_set_string(img, "CAMERA", "example", "camera model"));
    RefFitsOptions fits = ref_fits_options_default();
    fits.compression = REF_FITS_RICE;
    CHECK(ref_image_write_fits(img, "last.fits", &fits, true));

    RefExposureResult exp;
    CHECK(ref_exposure_calculate(NULL, img, 0.0, 1, &exp));
    printf("next exposure %.4f s (measured %.3f of full scale)\n", exp.exposure, exp.measured);

    ref_image_free(img);
    ref_runner_free(runner);
    ref_pipeline_free(p);
    free(display);
    free(raw);
    return 0;
}
