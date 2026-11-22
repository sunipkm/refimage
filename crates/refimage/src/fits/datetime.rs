//! UTC timestamp → FITS `DATE-OBS` string and `_S` / `_NS` card values.

use chrono::{DateTime, Utc};

/// Format `t` as `YYYY-MM-DDThh:mm:ss.nnnnnnnnn` (UTC, 9 fractional digits, no
/// trailing `Z` — FITS `DATE-OBS` is implicitly UTC).
pub(super) fn to_iso8601(t: DateTime<Utc>) -> String {
    t.format("%Y-%m-%dT%H:%M:%S%.9f").to_string()
}

/// Split into `(seconds, nanoseconds)` since the Unix epoch for the `_S` / `_NS`
/// metadata cards. `seconds` is negative for pre-1970 timestamps.
pub(super) fn epoch_parts(t: DateTime<Utc>) -> (i64, u32) {
    (t.timestamp(), t.timestamp_subsec_nanos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_instants() {
        assert_eq!(
            to_iso8601(DateTime::from_timestamp(0, 0).unwrap()),
            "1970-01-01T00:00:00.000000000"
        );
        // 2026-08-28T12:34:56.000000123 UTC
        let t = DateTime::from_timestamp(1_787_920_496, 123).unwrap();
        assert_eq!(to_iso8601(t), "2026-08-28T12:34:56.000000123");
    }

    #[test]
    fn pre_epoch_is_fine() {
        let t = DateTime::from_timestamp(-1, 0).unwrap();
        assert_eq!(to_iso8601(t), "1969-12-31T23:59:59.000000000");
        assert_eq!(epoch_parts(t), (-1, 0));
    }
}
