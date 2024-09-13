use crate::{
    ColorSpace, DataStor, DynamicImageData, DynamicImageOwned, ImageData, ImageOwned, PixelType,
};
use crate::{Deserializer, Serializer};
#[cfg(feature = "serde_flate")]
use flate2::{write::ZlibDecoder, write::ZlibEncoder, Compress, Compression};
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde_flate")]
use std::io::Write;

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

impl<'a> TryFrom<&'a DynamicImageData<'a>> for SerialImage {
    type Error = &'static str;

    fn try_from(data: &'a DynamicImageData<'a>) -> Result<Self, Self::Error> {
        let width = data.width();
        let height = data.height();
        let channels = data.channels();
        let cspace = data.color_space();
        let pixeltype: PixelType = (data).into();
        let data = data.as_raw_u8();
        let out;
        let crc = crc32fast::hash(data);
        let compressed;
        #[cfg(feature = "serde_flate")]
        {
            let mut encoder = ZlibEncoder::new_with_compress(
                Vec::new(),
                Compress::new(Compression::fast(), true),
            );
            encoder
                .write_all(data)
                .map_err(|_| "Could not write data to compressor.")?;
            out = encoder
                .finish()
                .map_err(|_| "Could not finalize compression.")?;
            compressed = true;
        }
        #[cfg(not(feature = "serde_flate"))]
        {
            out = data.to_vec();
            compressed = false;
        }
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

impl<'b> TryFrom<SerialImage> for DynamicImageData<'b> {
    type Error = &'static str;

    fn try_from(data: SerialImage) -> Result<Self, Self::Error> {
        let width = data.width;
        let height = data.height;
        let channels = data.channels;
        let cspace = data.cspace;
        let pixeltype = data.pixeltype.try_into()?;
        #[allow(unused_mut)]
        let mut out;
        #[cfg(feature = "serde_flate")]
        {
            if !data.compressed {
                return Err("Data is not compressed.");
            }
            out = Vec::new();
            let mut decoder = ZlibDecoder::new(out);
            decoder
                .write_all(&data.data)
                .map_err(|_| "Could not decompress the data.")?;
            out = decoder
                .finish()
                .map_err(|_| "Could not finalize the decompression.")?;
        }
        #[cfg(not(feature = "serde_flate"))]
        {
            if data.compressed {
                return Err("Data is compressed.");
            }
            out = data.data;
        }
        let crc = crc32fast::hash(&out);
        if data.crc != crc {
            return Err("Invalid data checksum");
        }
        match pixeltype {
            PixelType::U8 => {
                let img = ImageData::new(
                    DataStor::from_owned(out),
                    width.into(),
                    height.into(),
                    cspace,
                )?;
                if img.channels() != channels {
                    return Err("Data length does not match image size.");
                }
                Ok(DynamicImageData::U8(img))
            }
            PixelType::U16 => {
                let data = u8_slice_as_u16(&out).map_err(|_| "Could not cast u8 slice as u16")?;
                let img = ImageData::new(
                    DataStor::from_owned(data.as_slice().to_vec()),
                    width.into(),
                    height.into(),
                    cspace,
                )?;
                if img.channels() != channels {
                    return Err("Data length does not match image size.");
                }
                Ok(DynamicImageData::U16(img))
            }
            PixelType::F32 => {
                let data = u8_slice_as_f32(&out).map_err(|_| "Could not cast u8 slice as f32")?;
                let img = ImageData::new(
                    DataStor::from_owned(data.as_slice().to_vec()),
                    width.into(),
                    height.into(),
                    cspace,
                )?;
                if img.channels() != channels {
                    return Err("Data length does not match image size.");
                }
                Ok(DynamicImageData::F32(img))
            }
        }
    }
}

impl Serialize for DynamicImageData<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SerialImage::try_from(self)
            .map_err(|_| serde::ser::Error::custom("Could not serialize DynamicImageData"))
            .and_then(|img| img.serialize(serializer))
    }
}

impl<'de: 'a, 'a> Deserialize<'de> for DynamicImageData<'a> {
    fn deserialize<D>(deserializer: D) -> Result<DynamicImageData<'a>, D::Error>
    where
        D: Deserializer<'de>,
    {
        SerialImage::deserialize(deserializer).and_then(|img| {
            DynamicImageData::try_from(img)
                .map_err(|_| serde::de::Error::custom("Could not deserialize DynamicImageData"))
        })
    }
}

impl TryFrom<&DynamicImageOwned> for SerialImage {
    type Error = &'static str;

    fn try_from(data: &DynamicImageOwned) -> Result<Self, Self::Error> {
        let width = data.width();
        let height = data.height();
        let channels = data.channels();
        let cspace = data.color_space();
        let pixeltype: PixelType = (data).into();
        let data = data.as_raw_u8();
        let out;
        let crc = crc32fast::hash(data);
        let compressed;
        #[cfg(feature = "serde_flate")]
        {
            let mut encoder = ZlibEncoder::new_with_compress(
                Vec::new(),
                Compress::new(Compression::fast(), true),
            );
            encoder
                .write_all(data)
                .map_err(|_| "Could not write data to compressor.")?;
            out = encoder
                .finish()
                .map_err(|_| "Could not finalize compression.")?;
            compressed = true;
        }
        #[cfg(not(feature = "serde_flate"))]
        {
            out = data.to_vec();
            compressed = false;
        }
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
    type Error = &'static str;

    fn try_from(data: SerialImage) -> Result<Self, Self::Error> {
        let width = data.width;
        let height = data.height;
        let channels = data.channels;
        let cspace = data.cspace;
        let pixeltype = data.pixeltype.try_into()?;
        #[allow(unused_mut)]
        let mut out;
        #[cfg(feature = "serde_flate")]
        {
            if !data.compressed {
                return Err("Data is not compressed.");
            }
            out = Vec::new();
            let mut decoder = ZlibDecoder::new(out);
            decoder
                .write_all(&data.data)
                .map_err(|_| "Could not decompress the data.")?;
            out = decoder
                .finish()
                .map_err(|_| "Could not finalize the decompression.")?;
        }
        #[cfg(not(feature = "serde_flate"))]
        {
            if data.compressed {
                return Err("Data is compressed.");
            }
            out = data.data;
        }
        let crc = crc32fast::hash(&out);
        if data.crc != crc {
            return Err("Invalid data checksum");
        }
        match pixeltype {
            PixelType::U8 => {
                let img = ImageOwned::new(out, width.into(), height.into(), cspace)?;
                if img.channels() != channels {
                    return Err("Data length does not match image size.");
                }
                Ok(DynamicImageOwned::U8(img))
            }
            PixelType::U16 => {
                let data = u8_slice_as_u16(&out).map_err(|_| "Could not cast u8 slice as u16")?;
                let img = ImageOwned::new(
                    data.as_slice().to_vec(),
                    width.into(),
                    height.into(),
                    cspace,
                )?;
                if img.channels() != channels {
                    return Err("Data length does not match image size.");
                }
                Ok(DynamicImageOwned::U16(img))
            }
            PixelType::F32 => {
                let data = u8_slice_as_f32(&out).map_err(|_| "Could not cast u8 slice as f32")?;
                let img = ImageOwned::new(
                    data.as_slice().to_vec(),
                    width.into(),
                    height.into(),
                    cspace,
                )?;
                if img.channels() != channels {
                    return Err("Data length does not match image size.");
                }
                Ok(DynamicImageOwned::F32(img))
            }
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

type ByteResult<T> = Result<T, String>;

fn u8_slice_as_f32(buf: &[u8]) -> ByteResult<DtypeContainer<f32>> {
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
                _ => {
                    // If the buffer is not the correct length for a f32 slice, err.
                    Err(err.to_string())
                }
            }
        }
    }
}

fn u8_slice_as_u16(buf: &[u8]) -> ByteResult<DtypeContainer<u16>> {
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
                _ => {
                    // If the buffer is not the correct length for a f32 slice, err.
                    Err(err.to_string())
                }
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
                serde_generate::CodeGeneratorConfig::new("refimage::DynamicImageData".to_string())
                    .with_encodings(vec![serde_generate::Encoding::Bincode]);

            let gen = serde_generate::python3::CodeGenerator::new(&cfg);
            if let Err(v) = gen.output(&mut src, &registry) {
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
            std::fs::write(outdir.join("DynamicImageData.py"), src)
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

            let gen = serde_generate::python3::CodeGenerator::new(&cfg);
            if let Err(v) = gen.output(&mut src, &registry) {
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
