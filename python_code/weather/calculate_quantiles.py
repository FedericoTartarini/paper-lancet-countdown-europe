from pathlib import Path

import dask
import xarray as xr
from dask.diagnostics import ProgressBar
from icecream import ic

from my_config import Dirs, Variables

if __name__ == "__main__":

    for t_var in ["t_max", "t_min"]:

        file_list = []
        for file in Dirs.data_era_daily_summaries.value.rglob("*.nc"):
            file = Path(file)

            year = int(file.parts[-3])
            var = t_var.split("_")[1]

            if var not in file.parts:
                continue

            if (
                Variables.year_reference_start.value
                <= year
                <= Variables.year_reference_end.value
            ):
                ic(file)
                file_list.append(file)

        file_list = sorted(file_list)

        daily_temperatures = xr.open_mfdataset(
            file_list, combine="by_coords", chunks={"latitude": 100, "longitude": 100}
        )

        daily_temperatures = daily_temperatures.chunk({"time": -1})

        climatology_quantiles = (
            Dirs.data_era_quantiles.value
            / f'daily_{t_var}_quantiles_{"_".join([str(int(100*q)) for q in Variables.quantiles.value])}_{Variables.year_reference_start.value}-{Variables.year_reference_end.value}.nc'
        )

        daily_quantiles = daily_temperatures.quantile(
            Variables.quantiles.value, dim="time"
        )

        with dask.config.set(scheduler="processes"), ProgressBar():
            daily_quantiles = daily_quantiles.compute()
            daily_quantiles.to_netcdf(climatology_quantiles)
