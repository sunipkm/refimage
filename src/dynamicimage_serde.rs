use crate::imagetraits::ImageProps;
use crate::{ColorSpace, DynamicImageOwned, DynamicImageRef, ImageError, ImageOwned, PixelType};
use crate::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors from (de)serializing a [`DynamicImageOwned`] / [`DynamicImageRef`] to
/// or from the crate's internal wire format.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum SerdeError {
    /// The stored CRC32 does not match the payload.
    #[error("data checksum mismatch")]
    Checksum,
    /// The reconstructed image has a different channel count than recorded.
    #[error("channel count mismatch: recorded {expected}, image has {got}")]
    ChannelMismatch {
        /// Channel count recorded in the stream.
        expected: u8,
        /// Channel count of the reconstructed image.
        got: u8,
    },
    /// The pixel-type discriminant is not one of `{8, 16, -32}`.
    #[error("invalid pixel type discriminant")]
    InvalidPixelType,
    /// Reinterpreting the byte payload as the pixel type failed.
    #[error("byte cast failed: {0}")]
    Cast(&'static str),
    /// Rebuilding the image from the decoded fields failed.
    #[error(transparent)]
    Image(#[from] ImageError),
}

/// `Result` alias for [`SerdeError`].
pub type SerdeResult<T> = Result<T, SerdeError>;

#[derive(Serialize, Deserialize)]
struct SerialImage {
    width: u16,
    height: u16,
    channels: u8,
    cspace: ColorSpace,
    pixeltype: i8,
    compressed: bool,
    data: Vec<u8>,
    crc: u32,
}

impl<'a> TryFrom<&'a DynamicImageRef<'a>> for SerialImage {
    type Error = SerdeError;

    fn try_from(data: &'a DynamicImageRef<'a>) -> Result<Self, Self::Error> {
        let width = data.width();
        let height = data.height();
        let channels = data.channels();
        let cspace = data.color_space();
        let pixeltype: PixelType = (data).into();
        let data = data.as_raw_u8();
        let out = data.to_vec();
        let crc = crc32fast::hash(data);
        let compressed = false;

        Ok(SerialImage {
            width: width as _,
            height: height as _,
            channels,
            cspace: cspace as _,
            pixeltype: pixeltype as _,
            compressed,
            data: out,
            crc,
        })
    }
}

impl Serialize for DynamicImageRef<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SerialImage::try_from(self)
            .map_err(|_| serde::ser::Error::custom("Could not serialize DynamicImageRef"))
            .and_then(|img| img.serialize(serializer))
    }
}

impl TryFrom<&DynamicImageOwned> for SerialImage {
    type Error = SerdeError;

    fn try_from(data: &DynamicImageOwned) -> Result<Self, Self::Error> {
        let width = data.width();
        let height = data.height();
        let channels = data.channels();
        let cspace = data.color_space();
        let pixeltype: PixelType = (data).into();
        let data = data.as_raw_u8();
        let out = data.to_vec();
        let crc = crc32fast::hash(data);
        let compressed = false;
        Ok(SerialImage {
            width: width as _,
            height: height as _,
            channels,
            cspace: cspace as _,
            pixeltype: pixeltype as _,
            compressed,
            data: out,
            crc,
        })
    }
}

impl TryFrom<SerialImage> for DynamicImageOwned {
    type Error = SerdeError;

    fn try_from(data: SerialImage) -> Result<Self, Self::Error> {
        let width = data.width;
        let height = data.height;
        let channels = data.channels;
        let cspace = data.cspace;
        let pixeltype: PixelType = data
            .pixeltype
            .try_into()
            .map_err(|_| SerdeError::InvalidPixelType)?;
        #[allow(unused_mut)]
        let mut out = data.data;
        let crc = crc32fast::hash(&out);
        if data.crc != crc {
            return Err(SerdeError::Checksum);
        }
        fn check<T: crate::PixelStor>(img: &ImageOwned<T>, expected: u8) -> SerdeResult<()> {
            if img.channels() == expected {
                Ok(())
            } else {
                Err(SerdeError::ChannelMismatch {
                    expected,
                    got: img.channels(),
                })
            }
        }
        match pixeltype {
            PixelType::U8 => {
                let img = ImageOwned::new(out, width.into(), height.into(), cspace)?;
                check(&img, channels)?;
                Ok(DynamicImageOwned::U8(img))
            }
            PixelType::U16 => {
                let data = u8_slice_as_u16(&out).map_err(SerdeError::Cast)?;
                let img = ImageOwned::new(
                    data.as_slice().to_vec(),
                    width.into(),
                    height.into(),
                    cspace,
                )?;
                check(&img, channels)?;
                Ok(DynamicImageOwned::U16(img))
            }
            PixelType::F32 => {
                let data = u8_slice_as_f32(&out).map_err(SerdeError::Cast)?;
                let img = ImageOwned::new(
                    data.as_slice().to_vec(),
                    width.into(),
                    height.into(),
                    cspace,
                )?;
                check(&img, channels)?;
                Ok(DynamicImageOwned::F32(img))
            }
            other => Err(SerdeError::Image(ImageError::InvalidPixelType(other as i8))),
        }
    }
}

impl Serialize for DynamicImageOwned {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SerialImage::try_from(self)
            .map_err(|_| serde::ser::Error::custom("Could not serialize DynamicImageOwned"))
            .and_then(|img| img.serialize(serializer))
    }
}

impl<'de> Deserialize<'de> for DynamicImageOwned {
    fn deserialize<D>(deserializer: D) -> Result<DynamicImageOwned, D::Error>
    where
        D: Deserializer<'de>,
    {
        SerialImage::deserialize(deserializer).and_then(|img| {
            DynamicImageOwned::try_from(img)
                .map_err(|_| serde::de::Error::custom("Could not deserialize DynamicImageOwned"))
        })
    }
}

enum DtypeContainer<'a, T> {
    Slice(&'a [T]),
    Vec(Vec<T>),
}

impl<T> DtypeContainer<'_, T> {
    fn as_slice(&self) -> &[T] {
        match self {
            DtypeContainer::Slice(slice) => slice,
            DtypeContainer::Vec(vec) => vec,
        }
    }
}

type ByteResult<T> = Result<T, &'static str>;

fn u8_slice_as_f32(buf: &[u8]) -> ByteResult<DtypeContainer<'_, f32>> {
    let res = bytemuck::try_cast_slice(buf);
    match res {
        Ok(slc) => Ok(DtypeContainer::<'_, f32>::Slice(slc)),
        Err(err) => {
            match err {
                bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                    // If the buffer is not aligned for a f32 slice, copy the buffer into a new Vec<f32>
                    let mut vec = vec![0.0; buf.len() / 4];
                    for (i, chunk) in buf.chunks_exact(4).enumerate() {
                        let f32_val = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        vec[i] = f32_val;
                    }
                    Ok(DtypeContainer::Vec(vec))
                }
                _ => Err(crate::imageref::cast_msg(err)),
            }
        }
    }
}

fn u8_slice_as_u16(buf: &[u8]) -> ByteResult<DtypeContainer<'_, u16>> {
    let res = bytemuck::try_cast_slice(buf);
    match res {
        Ok(slc) => Ok(DtypeContainer::<u16>::Slice(slc)),
        Err(err) => {
            match err {
                bytemuck::PodCastError::TargetAlignmentGreaterAndInputNotAligned => {
                    // If the buffer is not aligned for a f32 slice, copy the buffer into a new Vec<f32>
                    let mut vec = vec![0; buf.len() / 2];
                    for (i, chunk) in buf.chunks_exact(2).enumerate() {
                        let u16_val = u16::from_ne_bytes([chunk[0], chunk[1]]);
                        vec[i] = u16_val;
                    }
                    Ok(DtypeContainer::Vec(vec))
                }
                _ => Err(crate::imageref::cast_msg(err)),
            }
        }
    }
}

mod test {

    #[test]
    fn generate_pycode_dynamicimagedata() {
        use serde_reflection::{Tracer, TracerConfig};
        use std::path::Path;

        let mut tracer = Tracer::new(TracerConfig::default());
        if let Err(v) = tracer.trace_simple_type::<super::SerialImage>() {
            eprintln!("Tracer Error: {:?}", v);
            return;
        }
        if let Ok(registry) = tracer.registry() {
            let mut src = Vec::new();
            let cfg =
                serde_generate::CodeGeneratorConfig::new("refimage::DynamicImageRef".to_string())
                    .with_encodings(vec![serde_generate::Encoding::Bincode]);

            let rgen = serde_generate::python3::CodeGenerator::new(&cfg);
            if let Err(v) = rgen.output(&mut src, &registry) {
                eprintln!("Output Error: {:?}", v);
                return;
            }
            let outdir = Path::new(&"serde-interop/python3/dynamicimagedata");
            if let Err(v) = std::fs::create_dir_all(outdir) {
                match v.kind() {
                    std::io::ErrorKind::AlreadyExists => {}
                    _ => {
                        eprintln!("Error creating directory: {:?}", v);
                        return;
                    }
                }
            }
            std::fs::write(outdir.join("DynamicImageRef.py"), src)
                .expect("Could not write to file.");
        }
    }

    #[test]
    fn generate_pycode_dynamicimageowned() {
        use serde_reflection::{Tracer, TracerConfig};
        use std::path::Path;

        let mut tracer = Tracer::new(TracerConfig::default());
        if let Err(v) = tracer.trace_simple_type::<super::SerialImage>() {
            eprintln!("Tracer Error: {:?}", v);
            return;
        }
        if let Ok(registry) = tracer.registry() {
            let mut src = Vec::new();
            let cfg =
                serde_generate::CodeGeneratorConfig::new("refimage::DynamicImageOwned".to_string())
                    .with_encodings(vec![serde_generate::Encoding::Bincode]);

            let rgen = serde_generate::python3::CodeGenerator::new(&cfg);
            if let Err(v) = rgen.output(&mut src, &registry) {
                eprintln!("Output Error: {:?}", v);
                return;
            }
            let outdir = Path::new(&"serde-interop/python3/dynamicimageowned");
            if let Err(v) = std::fs::create_dir_all(outdir) {
                match v.kind() {
                    std::io::ErrorKind::AlreadyExists => {}
                    _ => {
                        eprintln!("Error creating directory: {:?}", v);
                        return;
                    }
                }
            }
            std::fs::write(outdir.join("DynamicImageOwned.py"), src)
                .expect("Could not write to file.");
        }
    }
}
