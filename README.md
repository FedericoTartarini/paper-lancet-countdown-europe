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

## Shapes to grids
1. Use the file [shapes_to_grid.py](python_code/calculations/shapes_to_grid.py) to convert the shapefiles to grids.

# To implement
- [ ] Get the new demographic file.

# Other info
To update the list of dependencies use:
```bash
conda list -e > requirements_conda.txt
pip list --format=freeze > requirements.txt
```