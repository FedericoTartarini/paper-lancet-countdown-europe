# Lancet Countdown 2025 Heatwave Indicator - Europe

This repository contains the code needed to generate the heatwave indicator for the Lancet Countdown Europe report.
This code allows you to:
1. download the raw weather and population data
2. process, clean, and combine the data
3. generate the figures for the report and appendix.

## How to run the code

1. Clone the repository
2. Install the required packages using conda
3. Get your Personal Access Token from your profile on the CDS portal at the address: https://cds.climate.copernicus.eu/profile
4. Update [my_config.py](my_config.py), change the paths and the dates if needed.

## Weather data

The weather data used in this analysis is the ERA5 land reanalysis data from the Copernicus Climate Data Store (CDS).
The data is available at https://cds.climate.copernicus.eu
To download the data you need to:

1. Register on their portal and save the Personal Access Token in [secrets.py](python_code/secrets.py). Create the file if the file does not exist and save the token as `copernicus_api_key = "XX"`.
2. Download the data using [weather_data_download.py](python_code/weather/weather_data_download.py)
3. Preprocess the data using [weather_data_process.py](python_code/weather/weather_data_process.py)
4. Calculate the quantiles using [calculate_quantiles.py](python_code/weather/calculate_quantiles.py)
5. Calculate the heatwaves occurrences using [calculate_heatwaves.py](python_code/weather/calculate_heatwaves.py)

# To implement
## Population data
- [ ] Download the data.
- [ ] Process the data, for example, regrid the data to the ERA5 grid.
- [ ] Combine the age groups.
- [ ] Generate the rasterized data.
- [ ] Calculate the exposure to heatwaves.
- [ ] Calculate the change in exposure to heatwaves.
- [ ] Calculate the exposure due to climate change and pop growth.
- [ ] Generate most of the results.

## NCI super computer
Guide to connect to the NCI super computer and copy the files. https://opus.nci.org.au/spaces/Help/pages/12583143/How+to+login+to+Gadi

1. Connect to the NCI super computer `ssh ft8695@gadi.nci.org.au`
2. Copy the files over `scp /Users/ftar3919/Documents/lancet_countdown_europe/data/era5/era5_land_daily_summary/1991/max/era5-land_global_daily_tmax_199101.nc ft8695@gadi-dm.nci.org.au:/home/562/ft8695/lancet_countdown_europe/data/era5/era5_land_daily_summary/1991/max/`