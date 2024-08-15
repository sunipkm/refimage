use crate::{ColorSpace, DataStor, DynamicImageData, ImageData, PixelType};

struct DynamicImage<'a> {
    width: u16,
    height: u16,
    channels: u8,
    cspace: ColorSpace,
    pixeltype: PixelType,
    data: &'a [u8],
}

impl<'a> From<&'a DynamicImageData<'a>> for DynamicImage<'a> {
    fn from(data: &'a DynamicImageData<'a>) -> DynamicImage<'a> {
        let width = data.width();
        let height = data.height();
        let channels = data.channels();
        let cspace = data.color_space();
        let pixeltype: PixelType = (data).into();
        let data = data.as_raw_u8();
        DynamicImage {
            width: width as _,
            height: height as _,
            channels,
            cspace,
            pixeltype,
            data,
        }
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

impl<'a, 'b> TryFrom<DynamicImage<'a>> for DynamicImageData<'b> {
    type Error = &'static str;

    fn try_from(data: DynamicImage<'a>) -> Result<Self, Self::Error> {
        let width = data.width;
        let height = data.height;
        let channels = data.channels;
        let cspace = data.cspace;
        let pixeltype = data.pixeltype;
        match pixeltype {
            PixelType::U8 => {
                let img = ImageData::new(
                    DataStor::from_owned(data.data.to_vec()),
                    width,
                    height,
                    cspace,
                )?;
                if img.channels() != channels {
                    return Err("Data length does not match image size.");
                }
                Ok(DynamicImageData::U8(img))
            }
            PixelType::U16 => {
                let data =
                    u8_slice_as_u16(data.data).map_err(|_| "Could not cast u8 slice as u16")?;
                let img = ImageData::new(
                    DataStor::from_owned(data.as_slice().to_vec()),
                    width,
                    height,
                    cspace,
                )?;
                if img.channels() != channels {
                    return Err("Data length does not match image size.");
                }
                Ok(DynamicImageData::U16(img))
            }
            PixelType::F32 => {
                let data =
                    u8_slice_as_f32(data.data).map_err(|_| "Could not cast u8 slice as f32")?;
                let img = ImageData::new(
                    DataStor::from_owned(data.as_slice().to_vec()),
                    width,
                    height,
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
