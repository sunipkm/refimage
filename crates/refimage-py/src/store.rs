//! The pixels and metadata of a `refimage.Image`: owned by this module, or **adopted** from
//! a numpy array that another extension module allocated (refimage-pyo3's zero-copy path).
//!
//! An adopted image is a [`GenericImageRef`] over the array's memory. Its `'static`
//! lifetime is a promise kept here: the array is held in `owner` for as long as the
//! borrow exists, and nothing outside this module ever sees the `'static` lifetime --
//! [`Store::get`] only hands out borrows tied to `&self`.

use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;
use refimage::pipeline::{Frame, Pipeline, PipelineError};
use refimage::{
    CalcOptExp, DynamicImageOwned, DynamicImageRef, GenericImage, GenericImageOwned,
    GenericImageRef, ImageProps, ImageRef, Metadata, OptimumExposure, OptimumExposureResult,
    PixelData, PixelType,
};
use refimage_pyo3::parts::{self, Header};

use crate::err;

pub(crate) struct Store {
    // Declared before `owner`, so it is dropped first: `Ref` borrows `owner`'s memory.
    img: GenericImage<'static>,
    /// The numpy array whose memory an adopted image uses (`None` when owned).
    owner: Option<Py<PyAny>>,
}

impl Store {
    pub(crate) fn owned(img: GenericImageOwned) -> Self {
        Self {
            img: GenericImage::Own(img),
            owner: None,
        }
    }

    /// Adopt `pixels` (checked against `header` by [`parts::read`]) as the image's buffer,
    /// without copying, when it is a C-contiguous, aligned, writable numpy array; `None`
    /// otherwise.
    pub(crate) fn adopt(header: Header, pixels: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        if !parts::adoptable(pixels)? {
            return Ok(None);
        }
        let (w, h, cs) = (header.width, header.height, header.color_space.clone());
        let n = w * h * header.channels();
        macro_rules! adopt {
            ($t:ty) => {{
                let a = pixels.cast::<PyArrayDyn<$t>>()?;
                if numpy::PyUntypedArrayMethods::len(a) != n {
                    return Ok(None);
                }
                // SAFETY: `a` is C-contiguous, aligned and holds `n` initialised `$t`s
                // (checked above and by `parts::read`). The memory stays valid while the
                // array lives, and `owner` keeps it alive for as long as the slice exists
                // (`img` is dropped before `owner`, and is never moved out of `Store`).
                // numpy views of the same array may alias it, exactly as numpy views of an
                // owned image alias its `Vec` through the buffer protocol.
                let data: &'static mut [$t] =
                    unsafe { std::slice::from_raw_parts_mut(a.data(), n) };
                ImageRef::new(data, w, h, cs).map_err(err)?
            }};
        }
        let image: DynamicImageRef<'static> = match header.pixel_type.storage() {
            PixelType::U8 => adopt!(u8).into(),
            PixelType::U16 => adopt!(u16)
                .with_bit_depth(parts::bit_depth_tag(header.pixel_type))
                .into(),
            _ => adopt!(f32).into(),
        };
        Ok(Some(Self {
            img: GenericImage::Ref(GenericImageRef::with_metadata(header.metadata, image)),
            owner: Some(pixels.clone().unbind()),
        }))
    }

    /// The image, borrowed for no longer than `self`.
    pub(crate) fn get(&self) -> &GenericImage<'_> {
        &self.img
    }

    pub(crate) fn metadata_mut(&mut self) -> &mut Metadata {
        self.img.metadata_mut()
    }

    /// The pixels as a pipeline frame (read in place).
    pub(crate) fn frame(&self) -> &(dyn Frame + '_) {
        match &self.img {
            GenericImage::Ref(g) => g.image(),
            GenericImage::Own(g) => g.image(),
        }
    }

    /// The pixel bytes, writable.
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        match &mut self.img {
            GenericImage::Ref(g) => match g.image_mut() {
                DynamicImageRef::U8(i) => i.as_mut_slice(),
                DynamicImageRef::U16(i) => bytemuck::cast_slice_mut(i.as_mut_slice()),
                DynamicImageRef::F32(i) => bytemuck::cast_slice_mut(i.as_mut_slice()),
                _ => &mut [],
            },
            GenericImage::Own(g) => g.image_mut().as_mut_raw_u8(),
        }
    }

    /// Re-tag 16-bit storage with `pt`'s bit depth (no change to the samples).
    pub(crate) fn retag(&mut self, pt: PixelType) {
        match &mut self.img {
            GenericImage::Ref(g) => {
                if let DynamicImageRef::U16(i) = g.image_mut() {
                    i.set_bit_depth(pt.bit_depth());
                }
            }
            GenericImage::Own(g) => {
                if let DynamicImageOwned::U16(i) = g.image_mut() {
                    i.set_bit_depth(pt.bit_depth());
                }
            }
        }
    }

    /// Write `src` into the pixels: in place when the shape and storage type match (no
    /// allocation, adopted buffers included), else `Err(())` -- the caller decides whether
    /// replacing the buffer is allowed, then calls [`replace`](Self::replace).
    pub(crate) fn write(&mut self, src: &DynamicImageRef<'_>) -> Result<(), ()> {
        let d = &self.img;
        let fits = d.width() == src.width()
            && d.height() == src.height()
            && d.color_space() == src.color_space()
            && d.pixel_type().storage() == src.pixel_type().storage();
        if !fits {
            return Err(());
        }
        let bytes = src.as_raw_u8();
        self.bytes_mut()[..bytes.len()].copy_from_slice(bytes);
        self.retag(src.pixel_type());
        Ok(())
    }

    /// Replace the pixels with a buffer of this module's own (releasing an adopted one).
    pub(crate) fn replace(&mut self, src: &DynamicImageRef<'_>) {
        let meta = self.img.metadata().clone();
        self.img = GenericImage::Own(GenericImageOwned::with_metadata(meta, src));
        self.owner = None;
    }

    /// A deep copy with a buffer of this module's own.
    pub(crate) fn to_owned_image(&self) -> GenericImageOwned {
        self.img.clone().into_owned()
    }

    /// Run `pipeline` once, returning a new image with this one's metadata.
    pub(crate) fn apply(&self, pipeline: &Pipeline) -> Result<GenericImageOwned, PipelineError> {
        match &self.img {
            GenericImage::Ref(g) => pipeline.apply(g),
            GenericImage::Own(g) => pipeline.apply(g),
        }
    }

    /// The optimum exposure, reordering the pixels in place.
    pub(crate) fn calc_opt_exp_in_place(
        &mut self,
        eval: &OptimumExposure,
        exposure: std::time::Duration,
        bin: u16,
    ) -> refimage::ExposureResult<OptimumExposureResult> {
        self.img.calc_opt_exp(eval, exposure, bin)
    }

    /// bincode bytes: the same as `bincode::serialize(&GenericImageOwned)`.
    pub(crate) fn serialize(&self) -> bincode::Result<Vec<u8>> {
        match &self.img {
            GenericImage::Ref(g) => bincode::serialize(g),
            GenericImage::Own(g) => bincode::serialize(g),
        }
    }

    /// The exchange-format header (shape, type and a copy of the metadata).
    pub(crate) fn header(&self) -> Header {
        Header::new(self.img.metadata().clone(), &self.img)
    }

    /// Identical pixels, shape, type and metadata.
    pub(crate) fn same(&self, other: &Store) -> bool {
        let (a, b) = (&self.img, &other.img);
        a.metadata() == b.metadata()
            && a.width() == b.width()
            && a.height() == b.height()
            && a.color_space() == b.color_space()
            && a.pixel_type() == b.pixel_type()
            && a.image().as_raw_u8() == b.image().as_raw_u8()
    }

    /// Run `f` on the image as something refimage's writers take (FITS).
    pub(crate) fn with_generic<R>(
        &self,
        f_ref: impl FnOnce(&GenericImageRef<'_>) -> R,
        f_own: impl FnOnce(&GenericImageOwned) -> R,
    ) -> R {
        match &self.img {
            GenericImage::Ref(g) => f_ref(g),
            GenericImage::Own(g) => f_own(g),
        }
    }
}
