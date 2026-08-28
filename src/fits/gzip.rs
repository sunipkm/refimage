//! Minimal gzip-stream encoder (RFC 1952) for the `GZIP_1` tile-compression type.
//!
//! `astropy.io.fits` decompresses `GZIP_1` tiles with `zlib.decompress(data, 31)`, i.e.
//! it expects a full gzip member: 10-byte header, raw DEFLATE body, CRC-32 and ISIZE
//! trailer.

use miniz_oxide::deflate::compress_to_vec;

/// gzip-compress `data` into one gzip member.
pub(super) fn gzip(data: &[u8]) -> Vec<u8> {
    // level 6 — cfitsio's default effort.
    let deflated = compress_to_vec(data, 6);

    let mut out = Vec::with_capacity(deflated.len() + 18);
    out.extend_from_slice(&[
        0x1f, 0x8b, // magic
        0x08, // CM = deflate
        0x00, // FLG
        0x00, 0x00, 0x00, 0x00, // MTIME = 0
        0x00, // XFL
        0xff, // OS = unknown
    ]);
    out.extend_from_slice(&deflated);
    out.extend_from_slice(&crc32fast::hash(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use miniz_oxide::inflate::decompress_to_vec;

    #[test]
    fn roundtrips_through_raw_inflate() {
        let src: Vec<u8> = (0..4096u32).map(|i| (i * 31) as u8).collect();
        let comp = gzip(&src);
        assert_eq!(&comp[..2], &[0x1f, 0x8b]);
        // Body between the 10-byte header and the 8-byte trailer is raw DEFLATE.
        let body = &comp[10..comp.len() - 8];
        assert_eq!(decompress_to_vec(body).unwrap(), src);
        let isize_le = &comp[comp.len() - 4..];
        assert_eq!(
            u32::from_le_bytes(isize_le.try_into().unwrap()),
            src.len() as u32
        );
    }
}
