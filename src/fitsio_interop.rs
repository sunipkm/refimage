use std::{
    fmt::Display,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use chrono::DateTime;
pub use fitsio::errors::Error as FitsError;
use fitsio::{
    hdu::FitsHdu,
    images::{ImageDescription, ImageType, WriteImage},
    FitsFile,
};

use crate::{
    metadata::{GenericValue, TIMESTAMP_KEY},
    BayerPattern, ColorSpace, DynamicImageRef, DynamicImageOwned, GenericImageRef, GenericImageOwned,
    GenericLineItem, ImageRef, ImageOwned, PixelStor, PixelType,
};

#[derive(Debug, Clone, PartialEq, Hash)]
#[non_exhaustive]
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
    /// Custom compression algorithm.
    ///
    /// The custom settings are passed as a string.
    ///
    /// For more information, visit the [relevant FITSIO documentation page](https://heasarc.gsfc.nasa.gov/fitsio/c/c_user/node41.html).
    ///
    /// # Format
    /// `[compress NAME T1,T2; q[z] QLEVEL, s HSCALE]`, where
    /// - `NAME`: Algorithm name:  GZIP, Rice, HCOMPRESS, HSCOMPRSS or PLIO
    ///   may be abbreviated to the first letter (or HS for HSCOMPRESS).
    /// - `T1, T2`: Tile dimension (e.g. 100,100 for square tiles 100 pixels wide).
    /// - `QLEVEL`: Quantization level for floating point FITS images.
    /// - `HSCALE`: `HCOMPRESS` scale factor; default = 0 which is lossless.
    ///
    /// # Example
    /// - `[compress]`: Use the default compression algorithm (Rice)
    ///   and the default tile size (row by row).
    /// - `[compress G|R|P|H]`: Use the specified compression algorithm; only the first letter
    ///   of the algorithm should be given.
    /// - `[compress R 100,100]`: Use Rice and 100 x 100 pixel tiles.
    /// - `[compress R; q 10.0]`: Quantization level = `(RMS - noise) / 10`.
    /// - `[compress R; qz 10.0]`: Quantization level = `(RMS - noise) / 10`, using the
    ///   `SUBTRACTIVE_DITHER_2` quantization method.
    /// - `[compress HS; s 2.0]`: `HSCOMPRESS` (with smoothing) and scale = `(2.0 * RMS) - noise`.
    Custom(String),
}

impl FitsCompression {
    fn extension(&self) -> String {
        match self {
            FitsCompression::None => "fits".into(),
            FitsCompression::Gzip => "fits[compress G]".into(),
            FitsCompression::Rice => "fits[compress R]".into(),
            FitsCompression::Hcompress => "fits[compress H]".into(),
            FitsCompression::Hsmooth => "fits[compress HS]".into(),
            FitsCompression::Bzip2 => "fits[compress B]".into(),
            FitsCompression::Plio => "fits[compress P]".into(),
            FitsCompression::Custom(val) => format!("fits{}", val),
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
            FitsCompression::None => "uncomp".into(),
            FitsCompression::Gzip => "gzip".into(),
            FitsCompression::Rice => "rice".into(),
            FitsCompression::Hcompress => "hcompress".into(),
            FitsCompression::Hsmooth => "hscompress".into(),
            FitsCompression::Bzip2 => "bzip2".into(),
            FitsCompression::Plio => "plio".into(),
            FitsCompression::Custom(val) => val.clone(),
        }
        .fmt(f)
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "fitsio")))]
/// Trait for writing objects to FITS files.
pub trait FitsWrite {
    #[cfg_attr(docsrs, doc(cfg(feature = "fitsio")))]
    /// Write the image, with metadata, to a FITS file.
    ///
    /// # Arguments
    /// - `path`: The path to write the FITS file to.
    /// - `compress`: The compression algorithm to use ([`FitsCompression`]).
    /// - `overwrite`: Whether to overwrite the file if it already exists.
    ///
    /// # Returns
    /// The path to the written FITS file.
    ///
    /// # Errors
    /// This function returns errors from the FITS library if the file could not be written.
    fn write_fits(
        &self,
        path: &Path,
        compress: FitsCompression,
        overwrite: bool,
    ) -> Result<PathBuf, FitsError>;
}

impl FitsWrite for GenericImageRef<'_> {
    fn write_fits(
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
                let val = val
                    .get_value()
                    .get_value_systemtime()
                    .ok_or(FitsError::Message(
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

        let lineitem = GenericLineItem {
            value: GenericValue::String(datestamp),
            comment: Some("Date and time of FITS file data".to_string()),
        };
        lineitem.write_key("DATE-OBS", &hdu, &mut fptr)?;

        let lineitem = GenericLineItem {
            value: self.get_image().color_space().into(),
            comment: Some("Color space of the image".to_string()),
        };
        lineitem.write_key("COLOR_SPACE", &hdu, &mut fptr)?;

        for (name, value) in self.get_metadata().iter() {
            value.write_key(name, &hdu, &mut fptr)?;
        }
        Ok(fpath)
    }
}

impl FitsWrite for GenericImageOwned {
    fn write_fits(
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
                let val = val
                    .get_value()
                    .get_value_systemtime()
                    .ok_or(FitsError::Message(
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

        let lineitem = GenericLineItem {
            value: GenericValue::String(datestamp),
            comment: Some("Date and time of FITS file data".to_string()),
        };
        lineitem.write_key("DATE-OBS", &hdu, &mut fptr)?;

        let lineitem = GenericLineItem {
            value: self.get_image().color_space().into(),
            comment: Some("Color space of the image".to_string()),
        };
        lineitem.write_key("COLOR_SPACE", &hdu, &mut fptr)?;

        for (name, value) in self.get_metadata().iter() {
            value.write_key(name, &hdu, &mut fptr)?;
        }
        Ok(fpath)
    }
}

impl<'a> DynamicImageRef<'a> {
    fn write_fits(
        &self,
        path: PathBuf,
        compress: FitsCompression,
    ) -> Result<(FitsHdu, FitsFile), FitsError> {
        use DynamicImageRef::*;
        match self {
            U8(data) => data.write_fits(path, compress, PixelType::U8),
            U16(data) => data.write_fits(path, compress, PixelType::U16),
            F32(data) => data.write_fits(path, compress, PixelType::F32),
        }
    }
}

impl DynamicImageOwned {
    fn write_fits(
        &self,
        path: PathBuf,
        compress: FitsCompression,
    ) -> Result<(FitsHdu, FitsFile), FitsError> {
        use DynamicImageOwned::*;
        match self {
            U8(data) => data.write_fits(path, compress, PixelType::U8),
            U16(data) => data.write_fits(path, compress, PixelType::U16),
            F32(data) => data.write_fits(path, compress, PixelType::F32),
        }
    }
}

impl<'a, T: PixelStor + WriteImage> ImageRef<'a, T> {
    /// Write the image data to a FITS file.
    fn write_fits(
        &self,
        path: PathBuf,
        compress: FitsCompression,
        pxltype: PixelType,
    ) -> Result<(FitsHdu, FitsFile), FitsError> {
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

        hdu.write_image(&mut fptr, self.data)?;
        Ok((hdu, fptr))
    }
}

impl<T: PixelStor + WriteImage> ImageOwned<T> {
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

pub(crate) trait WriteKey {
    fn write_key(&self, key: &str, hdu: &FitsHdu, fptr: &mut FitsFile) -> Result<(), FitsError>;
}

impl WriteKey for GenericLineItem {
    fn write_key(&self, key: &str, hdu: &FitsHdu, fptr: &mut FitsFile) -> Result<(), FitsError> {
        if let Some(cmt) = &self.comment {
            match &self.value {
                GenericValue::U8(v) => hdu.write_key(fptr, key, (*v, cmt.as_str())),
                GenericValue::U16(v) => hdu.write_key(fptr, key, (*v, cmt.as_str())),
                GenericValue::F32(v) => hdu.write_key(fptr, key, (*v, cmt.as_str())),
                GenericValue::String(ref v) => hdu.write_key(fptr, key, (v.as_str(), cmt.as_str())),
                GenericValue::SystemTime(v) => {
                    let v = v
                        .duration_since(UNIX_EPOCH)
                        .map_err(|err| FitsError::Message(err.to_string()))?;
                    let cmt_ = format!("{cmt} (s from EPOCH)");
                    hdu.write_key(fptr, key, (v.as_secs(), cmt_.as_str()))?;
                    let cmt_ = format!("{cmt} (ns from EPOCH)");
                    hdu.write_key(fptr, key, (v.subsec_nanos(), cmt_.as_str()))
                }
                GenericValue::U32(v) => hdu.write_key(fptr, key, (*v, cmt.as_str())),
                GenericValue::U64(v) => hdu.write_key(fptr, key, (*v, cmt.as_str())),
                GenericValue::I8(v) => hdu.write_key(fptr, key, (*v, cmt.as_str())),
                GenericValue::I16(v) => hdu.write_key(fptr, key, (*v, cmt.as_str())),
                GenericValue::I32(v) => hdu.write_key(fptr, key, (*v, cmt.as_str())),
                GenericValue::I64(v) => hdu.write_key(fptr, key, (*v, cmt.as_str())),
                GenericValue::F64(v) => hdu.write_key(fptr, key, (*v, cmt.as_str())),
                GenericValue::ColorSpace(color_space) => {
                    hdu.write_key(fptr, key, (str_from_cspace(color_space), cmt.as_str()))
                }
                GenericValue::Duration(duration) => {
                    let cmt_ = format!("{cmt} (s)");
                    hdu.write_key(fptr, key, (duration.as_secs(), cmt_.as_str()))?;
                    let cmt_ = format!("{cmt} (ns)");
                    hdu.write_key(fptr, key, (duration.subsec_nanos(), cmt_.as_str()))
                }
            }
        } else {
            match &self.value {
                GenericValue::U8(v) => hdu.write_key(fptr, key, *v),
                GenericValue::U16(v) => hdu.write_key(fptr, key, *v),
                GenericValue::F32(v) => hdu.write_key(fptr, key, *v),
                GenericValue::String(ref v) => hdu.write_key(fptr, key, v.as_str()),
                GenericValue::SystemTime(v) => {
                    let v = v
                        .duration_since(UNIX_EPOCH)
                        .map_err(|err| FitsError::Message(err.to_string()))?;
                    hdu.write_key(fptr, key, (v.as_secs(), "s from EPOCH"))?;
                    hdu.write_key(fptr, key, (v.subsec_nanos(), "ns from EPOCH"))
                }
                GenericValue::U32(v) => hdu.write_key(fptr, key, *v),
                GenericValue::U64(v) => hdu.write_key(fptr, key, *v),
                GenericValue::I8(v) => hdu.write_key(fptr, key, *v),
                GenericValue::I16(v) => hdu.write_key(fptr, key, *v),
                GenericValue::I32(v) => hdu.write_key(fptr, key, *v),
                GenericValue::I64(v) => hdu.write_key(fptr, key, *v),
                GenericValue::F64(v) => hdu.write_key(fptr, key, *v),
                GenericValue::ColorSpace(color_space) => {
                    hdu.write_key(fptr, key, str_from_cspace(color_space))
                }
                GenericValue::Duration(duration) => {
                    hdu.write_key(fptr, key, (duration.as_secs(), "(s)"))?;
                    hdu.write_key(fptr, key, (duration.subsec_nanos(), "(ns)"))
                }
            }
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

fn str_from_cspace(cspace: &ColorSpace) -> String {
    let val = match cspace {
        ColorSpace::Gray => "GRAY",
        ColorSpace::GrayAlpha => "GRAYA",
        ColorSpace::Rgb => "RGB",
        ColorSpace::Rgba => "RGBA",
        ColorSpace::Bayer(BayerPattern::Bggr) => "BGGR",
        ColorSpace::Bayer(BayerPattern::Gbrg) => "GBRG",
        ColorSpace::Bayer(BayerPattern::Grbg) => "GRBG",
        ColorSpace::Bayer(BayerPattern::Rggb) => "RGGB",
        ColorSpace::Custom(val) => &format!("C({})", val),
    };
    val.to_string()
}

mod test {

    #[cfg(test)]
    fn files_equal(file1: &str, file2: &str) -> bool {
        use std::{
            fs::File,
            io::{BufReader, Read},
        };
        if let Result::Ok(file1) = File::open(file1) {
            let mut reader1 = BufReader::new(file1);
            if let Result::Ok(file2) = File::open(file2) {
                let mut reader2 = BufReader::new(file2);
                let mut buf1 = [0; 1024];
                let mut buf2 = [0; 1024];
                while let Result::Ok(n1) = reader1.read(&mut buf1) {
                    if n1 > 0 {
                        if let Result::Ok(n2) = reader2.read(&mut buf2) {
                            if n1 == n2 && buf1 == buf2 {
                                continue;
                            }
                            return false;
                        }
                    } else {
                        break;
                    }
                }
                return true;
            };
        };
        false
    }

    #[test]
    fn test_fitsio() {
        use crate::FitsWrite;
        let mut data = vec![1u8, 2, 3, 4, 5, 6];
        let img = crate::ImageRef::new(&mut data, 3, 2, crate::ColorSpace::Gray)
            .expect("Failed to create ImageRef");
        let img = crate::DynamicImageRef::from(img);
        let mut img = crate::GenericImageRef::new(std::time::SystemTime::now(), img);
        img.insert_key("CAMERA", "Canon EOS 5D Mark IV").unwrap();
        img.insert_key(
            "TESTING_THIS_LONG_KEY_VERY_VERY_VERY_VERYLONG",
            "This is a long key",
        )
        .unwrap();
        img.write_fits(
            std::path::Path::new("test.fits"),
            crate::FitsCompression::Custom("[compress R 2,3]".into()),
            true,
        )
        .expect("Could not write FITS file");
        let img: crate::GenericImageOwned = img.into();
        img.write_fits(
            std::path::Path::new("test2.fits"),
            crate::FitsCompression::Custom("[compress R 2,3]".into()),
            true,
        )
        .expect("Could not write FITS file");
        assert!(files_equal("test.fits", "test2.fits"));
    }
}
