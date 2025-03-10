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
        "data_format": "netcdf",
        "download_format": "unarchived",
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
                / f"{y}-{str(m).zfill(2)}_temperature.nc"
            )
            summary_file = Directories.data_era_daily_summaries.value / str(y)

            if not out_file.exists() and not summary_file.exists():
                ic(f"File missing: {y} and month: {str(m).zfill(2)}")
                list_of_monthly_files.append((out_file, str(m).zfill(2), y))
    return list_of_monthly_files


if __name__ == "__main__":

    list_of_m_files = generate_list_of_monthly_files_download()

    # for file_info in list_of_m_files:
    #     print(file_info)
    #     download(info=file_info, only_europe=False)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(download, file_info, only_europe=False)
            for file_info in list_of_m_files
        ]
        for future in futures:
            future.result()
