import os
from datetime import datetime
from enum import Enum
from pathlib import Path

from matplotlib import pyplot as plt

# Figure settings
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (7, 4)
plt.rcParams["figure.titlesize"] = "medium"
plt.rcParams["axes.titlesize"] = "medium"
plt.rcParams["savefig.bbox"] = "tight"


class Variables(Enum):
    year_report: int = datetime.now().year
    year_max_analysis: int = year_report - 1
    year_min_analysis: int = 1980
    year_min_comparison: int = 2015
    year_max_comparison: int = year_max_analysis
    year_reference_start: int = 1991
    year_reference_end: int = 2000
    quantiles = [0.95]


class Dirs(Enum):
    main: Path = Path(
        "/Users/ftar3919/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)/Academia/Datasets/lancet_countdown_europe"
    )
    adp: Path = Path("/mnt/scratch/frederico/lancet_countdown_europe")
    gadi: Path = Path("/home/562/ft8695/lancet_countdown_europe")
    results: Path = main / "results" / str(Variables.year_report.value)
    results_interim: Path = results / "interim"
    data: Path = main / "data"
    data_pop_gpw: Path = data / "population" / "gpwv4"
    data_era: Path = data / "era5"
    data_era_hourly: Path = data_era / "hourly_temperature_2m"
    data_era_daily_summaries: Path = data_era / "era5_land_daily_summary"
    data_era_heatwaves: Path = (
        data_era / "heatwaves" / f"results_{Variables.year_report.value}"
    )
    data_era_heatwaves_days: Path = data_era_heatwaves / "days"
    data_era_heatwaves_counts: Path = data_era_heatwaves / "counts"
    data_era_quantiles: Path = data_era / "era5_land_daily_quantiles"
    shapefiles: Path = main / "shapefiles"
    rasters: Path = Path(os.getcwd()) / "data" / "rasters"
    figures: Path = results / "figures"
    figures_test: Path = figures / "test"


if __name__ == "__main__":
    os.makedirs(Dirs.data_era_heatwaves_days.value, exist_ok=True)
    os.makedirs(Dirs.data_era_heatwaves_counts.value, exist_ok=True)
    Dirs.results.value.mkdir(exist_ok=True, parents=True)
    Dirs.results_interim.value.mkdir(exist_ok=True, parents=True)
    Dirs.data_pop_gpw.value.mkdir(exist_ok=True, parents=True)
    Dirs.figures.value.mkdir(exist_ok=True, parents=True)
