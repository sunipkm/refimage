//! `SystemTime` → ISO-8601 UTC string, without pulling in a date library.

use std::time::{SystemTime, UNIX_EPOCH};

use super::{FitsError, FitsResult};

/// Format `t` as `YYYY-MM-DDThh:mm:ss.nnnnnnnnn` (UTC, 9 fractional digits).
///
/// Errors if `t` is before the Unix epoch.
pub(super) fn to_iso8601(t: SystemTime) -> FitsResult<String> {
    let d = t
        .duration_since(UNIX_EPOCH)
        .map_err(|_| FitsError::TimestampBeforeEpoch)?;
    let secs = d.as_secs() as i64;
    let nanos = d.subsec_nanos();

    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (hh, mm, ss) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    let (y, m, day) = civil_from_days(days);

    Ok(format!(
        "{y:04}-{m:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}.{nanos:09}"
    ))
}

/// Split seconds-from-epoch into `(secs, nanos)` for the `_S` / `_NS` metadata cards.
pub(super) fn epoch_parts(t: SystemTime) -> FitsResult<(i64, u32)> {
    let d = t
        .duration_since(UNIX_EPOCH)
        .map_err(|_| FitsError::TimestampBeforeEpoch)?;
    Ok((d.as_secs() as i64, d.subsec_nanos()))
}

/// Civil date from a count of days since 1970-01-01 (Howard Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn known_instants() {
        assert_eq!(
            to_iso8601(UNIX_EPOCH).unwrap(),
            "1970-01-01T00:00:00.000000000"
        );
        // 2026-08-28T12:34:56.000000123 UTC
        let t = UNIX_EPOCH + Duration::new(1_787_920_496, 123);
        assert_eq!(to_iso8601(t).unwrap(), "2026-08-28T12:34:56.000000123");
    }

    #[test]
    fn before_epoch_errors() {
        assert!(matches!(
            to_iso8601(UNIX_EPOCH - Duration::from_secs(1)),
            Err(FitsError::TimestampBeforeEpoch)
        ));
    }
}
