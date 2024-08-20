use std::{
    fmt::Display, path::{Path, PathBuf}, time::{Duration, SystemTime, UNIX_EPOCH}
};

use chrono::DateTime;
use fitsio::{
    errors::Error as FitsError,
    hdu::FitsHdu,
    images::{ImageDescription, ImageType, WriteImage},
    FitsFile,
};

use crate::{
    metadata::{LineItem, TIMESTAMP_KEY},
    DynamicImageData, GenericImage, GenericLineItem, ImageData, PixelStor, PixelType,
};

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
/// Compression algorithms used in FITS files.
pub enum FitsCompression {
    /// No compression.
    None,
    /// GZIP compression.
    Gzip,
    /// Rice compression.
    Rice,
    /// HCOMPRESS compression.
    Hcompress,
    /// HCOMPRESS with smoothing.
    Hsmooth,
    /// BZIP2 compression.
    Bzip2,
    /// PLIO compression.
    Plio,
}

impl FitsCompression {
    fn extension(&self) -> &str {
        match self {
            FitsCompression::None => "fits",
            FitsCompression::Gzip => "fits[compress G]",
            FitsCompression::Rice => "fits[compress R]",
            FitsCompression::Hcompress => "fits[compress H]",
            FitsCompression::Hsmooth => "fits[compress HS]",
            FitsCompression::Bzip2 => "fits[compress B]",
            FitsCompression::Plio => "fits[compress P]",
        }
    }
}

impl From<Option<FitsCompression>> for FitsCompression {
    fn from(opt: Option<FitsCompression>) -> Self {
        opt.unwrap_or(FitsCompression::None)
    }
}

impl Display for FitsCompression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitsCompression::None => "uncomp",
            FitsCompression::Gzip => "gzip",
            FitsCompression::Rice => "rice",
            FitsCompression::Hcompress => "hcompress",
            FitsCompression::Hsmooth => "hscompress",
            FitsCompression::Bzip2 => "bzip2",
            FitsCompression::Plio => "plio",
        }
        .fmt(f)
    }
}

impl GenericImage<'_> {
    /// Write the image data to a FITS file.
    pub fn write_fits(
        &self,
        path: &Path,
        compress: FitsCompression,
        overwrite: bool,
    ) -> Result<PathBuf, FitsError> {
        if path.exists() && path.is_dir() {
            return Err(FitsError::Message("Path is a directory".to_string()));
        }

        let timestamp = match self.get_key(TIMESTAMP_KEY) {
            Some(val) => {
                let val = val.get_value_systemtime().ok_or(FitsError::Message(
                    "Could not convert timestamp to SystemTime".to_owned(),
                ))?;
                val.duration_since(UNIX_EPOCH)
                    .map_err(|err| {
                        FitsError::Message(format!(
                            "Could not convert SystemTime to duration since epoch: {}",
                            err
                        ))
                    })?
                    .as_millis()
            }
            None => SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
        };

        // let cameraname = match self.get_key(CAMERANAME_KEY) {
        //     Some(val) => val.get_value_string().unwrap_or_default(),
        //     None => "",
        // };

        let datestamp = DateTime::from_timestamp_millis(timestamp as i64).ok_or(
            FitsError::Message("Could not convert timestamp to NaiveDateTime".to_owned()),
        )?;
        let datestamp = datestamp.format("%Y-%m-%dT%H:%M:%S%.6f").to_string();

        let mut path = PathBuf::from(path);
        path.set_extension((FitsCompression::None).extension()); // Default extension
        if overwrite && path.exists() {
            // There seems to be a bug in FITSIO, overwrite() the way called here does nothing
            std::fs::remove_file(&path)?;
        }

        let fpath = path.clone();
        path.set_extension(compress.extension());

        let (hdu, mut fptr) = self.get_image().write_fits(path, compress)?;
        let lineitem = LineItem {
            name: "DATE-OBS".to_string(),
            value: datestamp,
            comment: Some("Date and time of FITS file data".to_string()),
        };
        lineitem.write_key(&hdu, &mut fptr)?;
        for item in self.get_metadata().iter() {
            item.write_key(&hdu, &mut fptr)?;
        }
        Ok(fpath)
    }
}

impl<'a> DynamicImageData<'a> {
    fn write_fits(
        &self,
        path: PathBuf,
        compress: FitsCompression,
    ) -> Result<(FitsHdu, FitsFile), FitsError> {
        match self {
            DynamicImageData::U8(data) => data.write_fits(path, compress, PixelType::U8),
            DynamicImageData::U16(data) => data.write_fits(path, compress, PixelType::U16),
            DynamicImageData::F32(data) => data.write_fits(path, compress, PixelType::F32),
        }
    }
}

impl<'a, T: PixelStor + WriteImage> ImageData<'a, T> {
    /// Write the image data to a FITS file.
    fn write_fits(
        &self,
        path: PathBuf,
        compress: FitsCompression,
        pxltype: PixelType,
    ) -> Result<(FitsHdu, FitsFile), FitsError> {
        let data = self.data.as_slice();
        let desc = ImageDescription {
            data_type: pxltype.into(),
            dimensions: if self.channels() > 1 {
                &[self.height() as _, self.width() as _, self.channels() as _]
            } else {
                &[self.height() as _, self.width() as _]
            },
        };
        
        let mut fptr = FitsFile::create(path);
        
        if compress == FitsCompression::None {
            fptr = fptr.with_custom_primary(&desc);
        }
        let mut fptr = fptr.open()?;

        let hdu = if compress == FitsCompression::None {
            fptr.primary_hdu()?
        } else {
            let hdu = fptr.primary_hdu()?;
            hdu.write_key(&mut fptr, "COMPRESSED_IMAGE", "T")?;
            hdu.write_key(&mut fptr, "COMPRESSION_ALGO", compress.to_string())?;
            fptr.create_image("IMAGE", &desc)?
        };

        hdu.write_image(&mut fptr, data)?;
        Ok((hdu, fptr))
    }
}

impl GenericLineItem {
    fn write_key(&self, hdu: &FitsHdu, fptr: &mut FitsFile) -> Result<(), FitsError> {
        match self {
            GenericLineItem::U8(item) => item.write_key(hdu, fptr),
            GenericLineItem::U16(item) => item.write_key(hdu, fptr),
            GenericLineItem::U32(item) => item.write_key(hdu, fptr),
            GenericLineItem::U64(item) => item.write_key(hdu, fptr),
            GenericLineItem::I8(item) => item.write_key(hdu, fptr),
            GenericLineItem::I16(item) => item.write_key(hdu, fptr),
            GenericLineItem::I32(item) => item.write_key(hdu, fptr),
            GenericLineItem::I64(item) => item.write_key(hdu, fptr),
            GenericLineItem::F32(item) => item.write_key(hdu, fptr),
            GenericLineItem::F64(item) => item.write_key(hdu, fptr),
            GenericLineItem::String(item) => item.write_key(hdu, fptr),
            GenericLineItem::SystemTime(item) => item.write_key(hdu, fptr),
            GenericLineItem::Duration(item) => item.write_key(hdu, fptr),
        }
    }
}

pub(crate) trait WriteKey {
    fn write_key(&self, hdu: &FitsHdu, fptr: &mut FitsFile) -> Result<(), FitsError>;
}

macro_rules! write_int_key_impl {
    ($type:ty) => {
        impl WriteKey for LineItem<$type> {
            fn write_key(&self, hdu: &FitsHdu, fptr: &mut FitsFile) -> Result<(), FitsError> {
                match &self.comment {
                    Some(cmt) => hdu.write_key(fptr, &self.name, (self.value, cmt.as_str())),
                    None => hdu.write_key(fptr, &self.name, self.value),
                }
            }
        }
    };
}

write_int_key_impl!(u8);
write_int_key_impl!(u16);
write_int_key_impl!(u32);
write_int_key_impl!(u64);
write_int_key_impl!(i8);
write_int_key_impl!(i16);
write_int_key_impl!(i32);
write_int_key_impl!(i64);
write_int_key_impl!(f32);
write_int_key_impl!(f64);

impl WriteKey for LineItem<String> {
    fn write_key(&self, hdu: &FitsHdu, fptr: &mut FitsFile) -> Result<(), FitsError> {
        match &self.comment {
            Some(cmt) => hdu.write_key(fptr, &self.name, (self.value.as_str(), cmt.as_str())),
            None => hdu.write_key(fptr, &self.name, self.value.as_str()),
        }
    }
}

impl WriteKey for LineItem<SystemTime> {
    fn write_key(&self, hdu: &FitsHdu, fptr: &mut FitsFile) -> Result<(), FitsError> {
        let timestamp = self.value.duration_since(UNIX_EPOCH).map_err(|err| {
            FitsError::Message(format!(
                "Could not convert SystemTime to duration since epoch: {}",
                err
            ))
        })?;
        match &self.comment {
            Some(cmt) => hdu.write_key(
                fptr,
                &self.name,
                (timestamp.as_micros() as u64, cmt.as_str()),
            ),
            None => hdu.write_key(fptr, &self.name, timestamp.as_micros() as u64),
        }
    }
}

impl WriteKey for LineItem<Duration> {
    fn write_key(&self, hdu: &FitsHdu, fptr: &mut FitsFile) -> Result<(), FitsError> {
        match &self.comment {
            Some(cmt) => hdu.write_key(
                fptr,
                &self.name,
                (self.value.as_micros() as u64, cmt.as_str()),
            ),
            None => hdu.write_key(fptr, &self.name, self.value.as_micros() as u64),
        }
    }
}

impl From<PixelType> for ImageType {
    fn from(pixeltype: PixelType) -> Self {
        match pixeltype {
            PixelType::U8 => ImageType::UnsignedByte,
            PixelType::U16 => ImageType::UnsignedShort,
            PixelType::F32 => ImageType::Float,
        }
    }
}
