from datetime import datetime
from enum import Enum
from pathlib import Path


class Variables(Enum):
    year_report: int = datetime.now().year
    year_max_analysis: int = year_report - 1
    year_min_analysis: int = 1980


class Directories(Enum):
    main: Path = Path("/Users/ftar3919/Documents/lancet_countdown_europe")
    data: Path = main / "data"
    data_era: Path = data / "era5"
    data_era_hourly: Path = data_era / "hourly_temperature_2m"
    data_era_daily_summaries: Path = data_era / "era5_land_daily_summary"
