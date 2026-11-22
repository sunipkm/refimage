// C++17 over refimage's C API: RAII handles, exceptions for errors, std containers for
// pixels. The same job as ../c/stream.c: debayer a stream of 12-bit frames to 8-bit luma at
// half size with every buffer reused, save the last raw frame as FITS, suggest the next
// exposure.
//
//   cmake -S . -B build -DCMAKE_PREFIX_PATH=<prefix> && cmake --build build && build/stream

#include <refimage.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ri
{

    /// A failed call, carrying its status and refimage's message.
    struct Error : std::runtime_error
    {
        RefStatus status;
        Error(RefStatus s, const std::string &what)
            : std::runtime_error(what + ": " + ref_last_error() + " (" + ref_status_name(s) + ")"),
              status(s) {}
    };

    /// Throw for anything but REF_OK.
    inline void check(RefStatus s, const char *what)
    {
        if (s != REF_OK)
            throw Error(s, what);
    }

    // Each handle owns its object and frees it with the matching ref_*_free.
    template <class T, void (*Free)(T *)>
    struct Deleter
    {
        void operator()(T *p) const noexcept { Free(p); }
    };
    using Image = std::unique_ptr<RefImage, Deleter<RefImage, ref_image_free>>;
    using Pipeline = std::unique_ptr<RefPipeline, Deleter<RefPipeline, ref_pipeline_free>>;
    using Runner = std::unique_ptr<RefRunner, Deleter<RefRunner, ref_runner_free>>;

    /// A frame borrowing `pixels` (never copied by a run).
    template <class T>
    RefFrame frame(const std::vector<T> &pixels, size_t width, size_t height, RefPixelType type,
                   RefColor color)
    {
        RefFrame f{};
        f.data = pixels.data();
        f.width = width;
        f.height = height;
        f.pixel_type = type;
        f.color = color;
        return f;
    }

    inline Pipeline new_pipeline()
    {
        RefPipeline *p = nullptr;
        check(ref_pipeline_new(&p), "ref_pipeline_new");
        return Pipeline(p);
    }

    inline Runner compile(const Pipeline &p, const RefFrame &like, const RefStrategy *strategy = nullptr)
    {
        RefRunner *r = nullptr;
        check(ref_pipeline_compile(p.get(), &like, strategy, &r), "ref_pipeline_compile");
        return Runner(r);
    }

    inline Image new_image(const RefFrame &f)
    {
        RefImage *img = nullptr;
        check(ref_image_new(&f, &img), "ref_image_new");
        return Image(img);
    }

} // namespace ri

int main()
try
{
    constexpr size_t W = 1920, H = 1080;
    std::printf("refimage %s (ABI %u)\n", ref_version(), ref_abi_version());

    std::vector<uint16_t> raw(W * H);
    const RefFrame frame = ri::frame(raw, W, H, REF_PIXEL_U12, REF_COLOR_BAYER_RGGB);

    auto p = ri::new_pipeline();
    ri::check(ref_pipeline_debayer(p.get(), REF_DEMOSAIC_LINEAR), "debayer");
    ri::check(ref_pipeline_to_luma(p.get()), "to_luma");
    ri::check(ref_pipeline_resize_to_fit(p.get(), W / 2, H / 2, REF_RESIZE_BILINEAR), "resize");
    ri::check(ref_pipeline_convert(p.get(), REF_PIXEL_U8), "convert");

    // Serial tiling: no allocation per frame once compiled.
    RefStrategy serial = ref_strategy_default();
    serial.parallel = false;
    auto runner = ri::compile(p, frame, &serial);

    RefImageInfo out{};
    ri::check(ref_runner_output_info(runner.get(), &out), "output_info");
    std::printf("output %zux%zu, %zu bytes\n", out.width, out.height, out.nbytes);

    std::vector<uint8_t> display(out.nbytes);
    for (int n = 0; n < 30; n++)
    {
        for (size_t i = 0; i < raw.size(); i++) // stand-in for a camera SDK
            raw[i] = static_cast<uint16_t>((i % W + i / W + n * 16) % 4096);
        ri::check(
            ref_runner_run_into(
                runner.get(),
                &frame,
                display.data(),
                display.size(),
                0),
            "run_into");
    }
    std::printf("display[0] = %u\n", display[0]);

    auto img = ri::new_image(frame);
    ri::check(ref_image_set_exposure(img.get(), 0.05), "set_exposure");
    ri::check(ref_image_set_string(img.get(), "CAMERA", "example", "camera model"), "set_string");
    RefFitsOptions fits = ref_fits_options_default();
    fits.compression = REF_FITS_RICE;
    ri::check(ref_image_write_fits(img.get(), "last.fits", &fits, true), "write_fits");

    RefExposureResult exp{};
    ri::check(ref_exposure_calculate(nullptr, img.get(), 0.0, 1, &exp), "exposure");
    std::printf("next exposure %.4f s (measured %.3f of full scale)\n", exp.exposure, exp.measured);

    // Errors become exceptions, with refimage's message.
    try
    {
        ri::check(ref_pipeline_debayer(p.get(), 42), "debayer");
    }
    catch (const ri::Error &e)
    {
        std::printf("caught: %s\n", e.what());
    }
    return 0; // the handles free themselves
}
catch (const std::exception &e)
{
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
