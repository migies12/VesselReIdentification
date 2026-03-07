# Data source ("skylight" or "elasticsearch")
DEFAULT_DATA_SOURCE = "es"

# Vessel filtering
MIN_IMAGES_PER_VESSEL = 3

# Data population / backfill
BACKFILL_LOOKBACK_DAYS = 540
BACKFILL_EVENT_TYPES = ["eo_sentinel2", "eo_landsat_8_9", "sar_sentinel1"]
BACKFILL_MIN_ESTIMATED_LENGTH = 150

# Cloud filtering
DRY_RUN = True

BRIGHTNESS_THRESHOLD = 115
SATURATION_THRESHOLD = 65
COVERAGE_THRESHOLD = 0.05

LUMINANCE_R = 0.3
LUMINANCE_G = 0.6
LUMINANCE_B = 0.1
