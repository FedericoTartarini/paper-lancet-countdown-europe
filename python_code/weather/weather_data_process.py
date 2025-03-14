import glob
import os
from pathlib import Path

import xarray as xr
from icecream import ic
from tqdm import tqdm

from my_config import Dirs


summary_variables = ["t_max", "t_min", "t_mean"]


def generate_daily_summary(file, move_source: bool = True) -> None:

    daily = xr.open_dataset(file)
    daily = daily.rename({"valid_time": "time"})
    daily = daily.resample(time="1D")

    for var in summary_variables:
        if var == "t_min":
            daily_summary = daily.min()
        elif var == "t_max":
            daily_summary = daily.max()
        elif var == "t_mean":
            daily_summary = daily.mean()

        daily_summary = daily_summary.rename({"t2m": var})

        file = Path(file)
        year = file.name.split("-")[0]
        month = file.name.split("-")[1].split("_")[0]

        summary_file = Dirs.data_era_daily_summaries.value / year / var.split("_")[1]
        summary_file.mkdir(parents=True, exist_ok=True)

        summary_file = summary_file / f"era5-land_global_daily_{var}_{year}{month}.nc"

        daily_summary.to_netcdf(
            summary_file,
            encoding={
                var: {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999},
            },
        )

    # # I am moving the file to the OneDrive folder to have a backup and save space on the computer
    # if move_source:
    #     shutil.move(
    #         source_file,
    #         source_file.replace(str(dir_era_hourly), dir_one_drive_era_hourly),
    #     )


def generate_list_of_files_to_process():
    list_of_files = []
    for file in glob.glob(str(Dirs.data_era_hourly.value) + "/*.nc"):

        file = Path(file)
        year = file.name.split("-")[0]
        month = file.name.split("-")[1].split("_")[0]

        var = summary_variables[0]
        summary_file = (
            Dirs.data_era_daily_summaries.value
            / year
            / var.split("_")[1]
            / f"era5-land_global_daily_{var}_{year}{month}.nc"
        )

        # checking that the file is fully downloaded before processing it
        size_gb = os.path.getsize(file) / 1025**3  # Convert bytes to GB
        if size_gb > 2 and not summary_file.exists():
            list_of_files.append(file)

    return list_of_files


def main():
    files = generate_list_of_files_to_process()

    for file in tqdm(files):
        ic(file)
        generate_daily_summary(file=file)


if __name__ == "__main__":
    main()
    # pass
