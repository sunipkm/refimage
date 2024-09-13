#![feature(test)]

#[cfg(test)]
mod tests {
    extern crate test;
    use rand::{thread_rng, Rng};
    use refimage::PixelStor;
    use test::{black_box, Bencher};

    #[bench]
    /// [`PixelStor::cast_u8`] to convert a single `u16` value to `u8`.
    fn bench_cast_u16(b: &mut Bencher) {
        let mut data = [0u16; 1];
        thread_rng().fill(&mut data[..]);
        b.iter(|| {
            // Inner closure, the actual test
            let res = data.iter().map(|&x| x.cast_u8()).collect::<Vec<u8>>();
            black_box(res);
        });
    }

    #[bench]
    /// [`PixelStor::cast_u8`] to convert a single `f32` value to `u8`.
    fn bench_cast_f32(b: &mut Bencher) {
        let mut data = [0u16; 1];
        thread_rng().fill(&mut data[..]);
        let data = data
            .iter()
            .map(|&x| x.to_f32() / u16::MAX as f32)
            .collect::<Vec<f32>>();
        b.iter(|| {
            // Inner closure, the actual test
            let res = data.iter().map(|&x| x.cast_u8()).collect::<Vec<u8>>();
            black_box(res);
        });
    }

    #[bench]
    /// [`PixelStor::floor_u8`] to convert a single `u16` value to `u8`.
    fn bench_floor_u16(b: &mut Bencher) {
        let mut data = [0u16; 1];
        thread_rng().fill(&mut data[..]);
        b.iter(|| {
            // Inner closure, the actual test
            let res = data.iter().map(|&x| x.cast_u8()).collect::<Vec<u8>>();
            black_box(res);
        });
    }

    #[bench]
    /// [`PixelStor::floor_u8`] to convert a single `f32` value to `u8`.
    fn bench_floor_f32(b: &mut Bencher) {
        let mut data = [0u16; 1];
        thread_rng().fill(&mut data[..]);
        let data = data
            .iter()
            .map(|&x| x.to_f32() / u16::MAX as f32)
            .collect::<Vec<f32>>();
        b.iter(|| {
            // Inner closure, the actual test
            let res = data.iter().map(|&x| x.cast_u8()).collect::<Vec<u8>>();
            black_box(res);
        });
    }

    #[bench]
    /// [`PixelStor::cast_u8`] to convert 1 million `u16` values to `u8`.
    fn bench_cast_u16_1mp(b: &mut Bencher) {
        let mut data = vec![0u16; 1024 * 1024];
        thread_rng().fill(&mut data[..]);

        b.iter(|| {
            // Inner closure, the actual test
            let res = data.iter().map(|&x| x.cast_u8()).collect::<Vec<u8>>();
            black_box(res);
        });
    }

    #[bench]
    /// [`PixelStor::cast_u8`] to convert 1 million `f32` values to `u8`.
    fn bench_cast_f32_1mp(b: &mut Bencher) {
        let mut data = vec![0u16; 1024 * 1024];
        thread_rng().fill(&mut data[..]);
        let data = data
            .iter()
            .map(|&x| x.to_f32() / u16::MAX as f32)
            .collect::<Vec<f32>>();

        b.iter(|| {
            // Inner closure, the actual test
            let res = data.iter().map(|&x| x.cast_u8()).collect::<Vec<_>>();
            black_box(res);
        });
    }

    #[bench]
    /// [`PixelStor::floor_u8`] to convert 1 million `u16` values to `u8`.
    fn bench_floor_u16_1mp(b: &mut Bencher) {
        let mut data = vec![0u16; 1024 * 1024];
        thread_rng().fill(&mut data[..]);

        b.iter(|| {
            // Inner closure, the actual test
            let res = data.iter().map(|&x| x.floor_u8()).collect::<Vec<u8>>();
            black_box(res);
        });
    }

    #[bench]
    /// [`PixelStor::floor_u8`] to convert 1 million `f32` values to `u8`.
    fn bench_floor_f3_1mp(b: &mut Bencher) {
        let mut data = vec![0u16; 1024 * 1024];
        thread_rng().fill(&mut data[..]);
        let data = data
            .iter()
            .map(|&x| x.to_f32() / u16::MAX as f32)
            .collect::<Vec<f32>>();

        b.iter(|| {
            // Inner closure, the actual test
            let res = data.iter().map(|&x| x.floor_u8()).collect::<Vec<_>>();
            black_box(res);
        });
    }
}
