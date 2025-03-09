import cdsapi
from icecream import ic
from concurrent.futures import ThreadPoolExecutor

from my_config import Directories, Variables
from python_code.secrets import copernicus_api_key


def download_year_era5(
    year: int, out_file: str, only_europe: bool = False, month=False
) -> None:

    c = cdsapi.Client(
        key=copernicus_api_key,
        url="https://cds.climate.copernicus.eu/api",
    )

    dataset = "reanalysis-era5-land"

    request = {
        "variable": "2m_temperature",
        "year": year,
        "month": month,
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
        ],
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "format": "grib",
    }

    if only_europe:
        request["area"] = [90, -80, 30, 90]

    c.retrieve(
        dataset,
        request,
        str(out_file),
    )


def download(info: tuple, only_europe: bool = False) -> None:

    out_file = info[0]
    m = info[1]
    y = info[2]

    ic(f"Downloading ERA5 data for year: {y} and month: {m}")
    download_year_era5(y, out_file, only_europe, m)


def generate_list_of_monthly_files_download():
    list_of_monthly_files = []
    for y in range(Variables.year_min_analysis.value, Variables.year_report.value):
        for m in range(1, 13):
            out_file = (
                Directories.data_era_hourly.value
                / f"{y}-{str(m).zfill(2)}_temperature.grib"
            )
            summary_file = Directories.data_era_daily_summaries.value / str(y)

            if not out_file.exists() and not summary_file.exists():
                ic(f"File missing: {y} and month: {str(m).zfill(2)}")
                list_of_monthly_files.append((out_file, str(m).zfill(2), y))
    return list_of_monthly_files


if __name__ == "__main__":

    list_of_m_files = generate_list_of_monthly_files_download()

    # for file_info in list_of_m_files:
    #     download(file_info, only_europe=False)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(download, file_info, only_europe=False)
            for file_info in list_of_m_files
        ]
        for future in futures:
            future.result()

    # download(only_europe=False, monthly=True)


# # var = 'max'
# # daily_folder = TEMPERATURE_SUMMARY_FOLDER / f'{year}' / var
# # daily_folder.mkdir(parents=True, exist_ok=True)
#
# # summary_file = daily_folder / f'era5-land_global_daily_t{var}_{year}{str(month).zfill(2)}.nc'
#
# # hourly_temperatures = xr.open_dataset(out_file, chunks=dict(step=-1, longitude=10))
#
# # ref = xr.open_dataset('/home/jonathanchambers/Shared/Data/weather/era5_land/era5_land_daily_summary/1979/max/era5-land_global_daily_tmax_197901.nc')
#
# # encoding = ref.tmax.encoding.copy()
# # encoding.pop('source')
#
# # daily_max = hourly_temperatures.max(dim='step').drop('surface').drop('number')
# # daily_max = daily_max.rename({'t2m': f't{var}'})
# # daily_max.attrs['frequency'] = 'day'
#
# # daily_max.to_netcdf(summary_file,
# #                     encoding={f't{var}':encoding})
#
#
# # ```shell
# # grib_to_netcdf -k 3 -d 1 2021-01_temperature_2m.grib -o 2021-01_temperature_2m.nc
# #
# # cdo -daymax -chname,t2m,tmax 2021-01_temperature_2m.nc era5-land_global_daily_tmax_19790101.nc
# #
# # ```
#
# # base_path / 'hourly_nc'
#
# import subprocess
# from tqdm.notebook import tqdm
#
# temperature_summary_folder
#
# year = max_year
# base_path = Path("~/Scratch").expanduser() / "proc_era5_land_tmp"
# # daily_folder = base_path / 'daily'
# daily_folder = temperature_summary_folder
#
#
# grib_to_netcdf = (
#     "/home/jonathanchambers/Scratch/.conda/envs/science2/bin/grib_to_netcdf"
# )
#
# for month in tqdm(list(range(1, 13))):
#     month = str(month).zfill(2)
#
#     # input_file = base_path / f'{year}-{month}_temperature_2m.grib'
#     # hourly_nc_file = base_path / 'hourly_nc' / f'{year}-{month}_temperature_2m.nc'
#     input_file = subdaily_temperatures_folder / f"{year}-{month}_temperature_2m.grib"
#     hourly_nc_file = (
#             data_src
#             / "weather"
#             / "era5_land"
#             / "hourly_nc"
#             / f"{year}-{month}_temperature_2m.nc"
#     )
#
#     to_netcdf = [
#         grib_to_netcdf,
#         "-k",
#         "3",
#         "-d",
#         "3",
#         str(input_file),
#         "-o",
#         str(hourly_nc_file),
#     ]
#
#     if not hourly_nc_file.exists():
#         subprocess.run(to_netcdf, shell=False, check=True)
#
#     for var in tqdm(["mean", "min", "max"]):
#         summary_file = (
#                 daily_folder
#                 / f"{year}"
#                 / f"{var}"
#                 / f"era5-land_global_daily_t{var}_{year}{str(month).zfill(2)}.nc"
#         )
#
#         daily_summary = [
#             "cdo",
#             f"-day{var}",
#             f"-chname,t2m,t{var}",
#             str(hourly_nc_file),
#             str(summary_file),
#         ]
#
#         if not summary_file.exists():
#             subprocess.run(daily_summary, shell=False, check=True)
#
# year = max_year
# base_path = Path("~/Scratch").expanduser() / "proc_era5_land_tmp"
# # daily_folder = base_path / 'daily'
# daily_folder = temperature_summary_folder
#
#
# grib_to_netcdf = (
#     "/home/jonathanchambers/Scratch/.conda/envs/science2/bin/grib_to_netcdf"
# )
#
# for month in tqdm(list(range(1, 2))):
#     month = str(month).zfill(2)
#
#     # input_file = base_path / f'{year}-{month}_temperature_2m.grib'
#     # hourly_nc_file = base_path / 'hourly_nc' / f'{year}-{month}_temperature_2m.nc'
#     input_file = subdaily_temperatures_folder / f"{year}-{month}_temperature_2m.grib"
#     hourly_nc_file = (
#             data_src
#             / "weather"
#             / "era5_land"
#             / "hourly_nc"
#             / f"{year}-{month}_temperature_2m.nc"
#     )
#
#     to_netcdf = [
#         grib_to_netcdf,
#         "-k",
#         "3",
#         "-d",
#         "3",
#         str(input_file),
#         "-o",
#         str(hourly_nc_file),
#     ]
#
#     if not hourly_nc_file.exists():
#         subprocess.run(to_netcdf, shell=False, check=True)
#
#     for var in tqdm(["max"]):
#         summary_file = (
#                 daily_folder
#                 / f"{year}"
#                 / f"{var}"
#                 / f"era5-land_global_daily_t{var}_{year}{str(month).zfill(2)}.nc"
#         )
#
#         daily_summary = [
#             "cdo",
#             f"-day{var}",
#             f"-chname,t2m,t{var}",
#             str(hourly_nc_file),
#             str(summary_file),
#         ]
#
#         # if not summary_file.exists():
#         subprocess.run(daily_summary, shell=False, check=True)
#
# new_encoding = {
#     "zlib": True,
#     "shuffle": True,
#     "complevel": 1,
#     "chunksizes": (1, 1801, 3600),
#     "original_shape": (31, 1801, 3600),
#     "dtype": np.dtype("int32"),
#     "missing_value": -9999,
#     "_FillValue": -9999,
#     "scale_factor": 0.0016127533931795336,
#     "add_offset": 264.7066521193972,
# }
#
# out_folder = Path(
#     "/home/jonathanchambers/Shared/Data/weather/era5_land/era5_land_daily_summary/2022/"
# )
# for month in tqdm(list(range(1, 13))):
#     month = str(month).zfill(2)
#     for var in tqdm(["mean", "min", "max"]):
#         summary_file = (
#                 daily_folder
#                 / f"t{var}"
#                 / f"era5-land_global_daily_t{var}_{year}{str(month).zfill(2)}.nc"
#         )
#         re_save_file = (
#                 out_folder
#                 / var
#                 / f"era5-land_global_daily_t{var}_{year}{str(month).zfill(2)}.nc"
#         )
#         data = xr.open_dataset(summary_file)
#         data.to_netcdf(re_save_file, encoding={f"t{var}": new_encoding})
#
# # subprocess.run(task_list['daymax'], shell=False, check=True)
# # subprocess.run(task_list['to_netcdf'], shell=False, check=True)
