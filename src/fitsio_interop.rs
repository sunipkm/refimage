use std::{
    fs::remove_file,
    io,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use chrono::DateTime;
use fitsio::{
    errors::Error as FitsError,
    hdu::FitsHdu,
    images::{ImageDescription, ImageType, WriteImage},
    FitsFile,
};

use crate::{
    metadata::{LineItem, CAMERANAME_KEY, PROGRAMNAME_KEY, TIMESTAMP_KEY},
    DynamicImageData, GenericImage, GenericLineItem, ImageData, PixelStor, PixelType,
};

impl GenericImage<'_> {
    /// Write the image data to a FITS file.
    pub fn write_fits(
        &self,
        dir_prefix: &Path,
        file_prefix: &str,
        progname: Option<&str>,
        compress: bool,
        overwrite: bool,
    ) -> Result<PathBuf, FitsError> {
        if !dir_prefix.exists() {
            return Err(FitsError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Directory {:?} does not exist", dir_prefix),
            )));
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

        let cameraname = match self.get_key(CAMERANAME_KEY) {
            Some(val) => val.get_value_string().unwrap_or_default(),
            None => "",
        };

        let datestamp = DateTime::from_timestamp_millis(timestamp as i64).ok_or(
            FitsError::Message("Could not convert timestamp to NaiveDateTime".to_owned()),
        )?;
        let datestamp = datestamp.format("%Y%m%d_%H%M%S").to_string();

        let file_prefix = if file_prefix.trim().is_empty() {
            cameraname
        } else {
            file_prefix
        };

        let fpath = dir_prefix.join(Path::new(&format!("{}_{}.fits", file_prefix, datestamp)));

        if fpath.exists() {
            if !overwrite {
                return Err(FitsError::Io(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("File {:?} already exists", fpath),
                )));
            } else {
                let res = remove_file(fpath.clone());
                if let Err(msg) = res {
                    return Err(FitsError::Io(io::Error::new(
                        io::ErrorKind::Other,
                        format!("Could not remove file {:?}: {}", fpath, msg),
                    )));
                }
            }
        }

        let path = Path::new(dir_prefix).join(Path::new(&format!(
            "{}_{}.fits{}",
            file_prefix,
            timestamp,
            if compress { "[compress]" } else { "" }
        )));

        let mut fptr = self.get_image().write_fits(path)?;
        let hdu = fptr.primary_hdu()?;
        if let Some(progname) = progname {
            if !progname.is_empty() {
                let lineitem = LineItem {
                    name: PROGRAMNAME_KEY.to_owned(),
                    value: progname.to_owned(),
                    comment: Some("Name of the program that created the file".to_owned()),
                };
                lineitem.write_key(&hdu, &mut fptr)?;
            }
        }
        for item in self.get_metadata().iter() {
            item.write_key(&hdu, &mut fptr)?;
        }
        Ok(fpath)
    }
}

impl<'a> DynamicImageData<'a> {
    fn write_fits(&self, path: PathBuf) -> Result<FitsFile, FitsError> {
        match self {
            DynamicImageData::U8(data) => data.write_fits(path, PixelType::U8),
            DynamicImageData::U16(data) => data.write_fits(path, PixelType::U16),
            DynamicImageData::F32(data) => data.write_fits(path, PixelType::F32),
        }
    }
}

impl<'a, T: PixelStor + WriteImage> ImageData<'a, T> {
    /// Write the image data to a FITS file.
    fn write_fits(&self, path: PathBuf, pxltype: PixelType) -> Result<FitsFile, FitsError> {
        let data = self.data.as_slice();
        let desc = ImageDescription {
            data_type: pxltype.into(),
            dimensions: &[self.width as _, self.height as _],
        };
        let mut fptr = FitsFile::create(path).with_custom_primary(&desc).open()?;
        let hdu = fptr.primary_hdu()?;
        hdu.write_image(&mut fptr, data)?;
        Ok(fptr)
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
            PixelType::U8 => ImageType::Byte,
            PixelType::U16 => ImageType::UnsignedShort,
            PixelType::F32 => ImageType::Float,
        }
    }
}
