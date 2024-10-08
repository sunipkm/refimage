#![feature(test)]



#[cfg(test)]
mod tests {
    extern crate test;
    extern crate paste;
    use rand::{thread_rng, Rng};
    use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};
    use refimage::PixelStor;
    use test::Bencher;

    macro_rules! generate_test{
        ($name:expr, $size:expr) => {
            ::paste::paste! {
                #[bench]
                fn [<bench_ $name _ $size>](b: &mut Bencher) {
                    let wid = $size;
                    let hei = $size;
                    let channels = 3;
                    let tot = wid * hei * channels;
                    let mut data = vec![0u16; tot];
                    thread_rng().fill(&mut data[..]);
                    let wts = [0.2126, 0.7152, 0.0722];
                    b.iter(|| {
                        [<luma_ $name>](wid, hei, channels, &mut data, &wts);
                    });
                }
            }
        };
    }

    generate_test!(par, 3072);
    generate_test!(par, 1024);
    generate_test!(par, 512);
    generate_test!(par, 256);
    generate_test!(par, 128);
    generate_test!(par, 64);

    generate_test!(seq, 3072);
    generate_test!(seq, 1024);
    generate_test!(seq, 512);
    generate_test!(seq, 256);
    generate_test!(seq, 128);
    generate_test!(seq, 64);

    fn luma_par(wid: usize, hei: usize, channels: usize, data: &mut [u16], wts: &[f64]) {
        let len = wid * hei;
        data.par_chunks_exact_mut(channels).for_each(|chunk| {
            let v = u16::from_f64(
                chunk
                    .iter()
                    .zip(wts.iter())
                    .fold(0f64, |acc, (px, &w)| acc + (*px).to_f64() * w),
            );
            chunk[0] = v;
        });
        for i in 0..len {
            data[i] = data[i * channels];
        }
    }

    fn luma_seq(wid: usize, hei: usize, channels: usize, data: &mut [u16], wts: &[f64]) {
        let len = wid * hei;
        for i in 0..len {
            let v = u16::from_f64(
                data[i * channels..(i + 1) * channels]
                    .iter()
                    .zip(wts.iter())
                    .fold(0f64, |acc, (px, &w)| acc + (*px).to_f64() * w),
            );
            data[i] = v;
        }
    }
}
