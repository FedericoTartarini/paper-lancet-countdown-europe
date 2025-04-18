"""
# Heatwave exposure in Europe

Use ERA5-Land data

https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/population-distribution-demography/geostat

https://ec.europa.eu/eurostat/data/database?node_code=demo_r_minf
Demographic data at regional level include statistics on the population at the end of the calendar year and on live births and deaths during that year, according to the official classification for statistics at regional level (NUTS - nomenclature of territorial units for statistics). These data are broken down by NUTS 2 and 3 levels. The current online demographic data refers to the NUTS 2016 classification, which subdivides the territory of the European Union into
"""

import math

import cartopy
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import xarray as xr
import xesmf as xe
from matplotlib.ticker import MultipleLocator
from rasterio import features
from tqdm import tqdm

from my_config import Dirs, Variables

xr.set_options(keep_attrs=True)


def get_nuts_2021():
    shapefile = Dirs.shapefiles.value / "NUTS_RG_03M_2021_4326.geojson"
    nuts_2021 = gpd.read_file(shapefile)
    nuts_2021 = nuts_2021[nuts_2021.LEVL_CODE == 2]
    nuts_2021["id"] = nuts_2021.NUTS_ID.values
    nuts_2021["id_as_int"] = (
        nuts_2021.id.str.encode("utf-8")
        .apply(lambda b: np.frombuffer(b, dtype=">u4")[0])
        .astype(np.uint32)
    )

    nuts_2021 = nuts_2021.set_index("id")
    return nuts_2021


def transform_raster_size(data):
    north = data.latitude[0].item()
    west = data.longitude[0].item()

    y_grid_size = north - data.latitude[1].item()
    x_grid_size = data.longitude[1].item() - west

    return rasterio.transform.from_origin(west, north, x_grid_size, y_grid_size)


def get_nuts_level_2(return_bounds_slices=False):
    nuts = gpd.read_file(
        Dirs.shapefiles.value / "NUTS_RG_03M_2016_4326 clipped.geojson.zip"
    )
    nuts2 = nuts[nuts.LEVL_CODE == 2].copy()
    nuts2["id_as_int"] = (
        nuts2.id.str.encode("utf-8")
        .apply(lambda b: np.frombuffer(b, dtype=">u4")[0])
        .astype(np.uint32)
    )
    nuts_bounds = nuts.geometry.total_bounds
    lat_slice = slice(nuts_bounds[3], nuts_bounds[1])
    lon_slice = slice(nuts_bounds[0], nuts_bounds[2])

    if return_bounds_slices:
        return nuts, nuts2, nuts_bounds, lat_slice, lon_slice

    return nuts, nuts2


def area_of_pixel(pixel_size, center_lat):
    """Calculate m^2 area of a wgs84 square pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397

    Parameters:
        pixel_size (float): length of side of pixel in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        Area of square pixel of side length `pixel_size` centered at
        `center_lat` in m^2.

    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = np.sqrt(1 - (b / a) ** 2)
    area_list = []
    for f in [center_lat + pixel_size / 2, center_lat - pixel_size / 2]:
        zm = 1 - e * np.sin(np.radians(f))
        zp = 1 + e * np.sin(np.radians(f))
        area = (
            np.pi
            * b**2
            * (np.log(zp / zm) / (2 * e) + np.sin(np.radians(f)) / (zp * zm))
        )

        area_list.append(area)
    return (pixel_size / 360.0) * (area_list[0] - area_list[1])


def _plot_year(year, data):
    data.sel(year=year).plot(robust=True)
    plt.show()


def prep_eurostat_table(data):
    # Annoyingly Eurostat decided to include data quality indicator codes (as letters) along side the numbers:
    # https://ec.europa.eu/eurostat/data/database/information. At least are separated with a space, so can split on space
    # and keep first part.

    cols = [c.strip() for c in data.columns]
    cols[0] = "key"
    data.columns = cols

    for c in data:
        if hasattr(data[c], "str"):
            data[c] = data[c].astype(str)
            data[c] = data[c].apply(lambda s: s.split(" ")[0])

    data.index = pd.MultiIndex.from_tuples(data.key.str.split(","))
    data = data.drop("key", axis=1)

    data.columns = data.columns.astype(int)
    data = data.replace(":", np.nan).astype(float)

    data = data[sorted(data.columns)]
    return data


def calculate_heatwave_days():
    # import data
    hw_file = sorted(Dirs.data_era_heatwaves_days.value.glob("*.nc"))
    heatwave_days = xr.open_mfdataset(hw_file)

    # NUTS shapefiles
    nuts, _, nuts_bounds, lat_slice, lon_slice = get_nuts_level_2(
        return_bounds_slices=True
    )

    # Subset the weather data
    # NOTE: since input data is 'image style' it's indexed from top-left so latitude is in decreasing order and lat
    # label ranges need to be in order [max val: min val]
    # NOTE: since we cross 0 longitude and ERA data is on 0-360 grid
    # need to stitch together a new dataset and change coords to be on -180 to 180

    lon_slice1 = slice(360 + nuts_bounds[0], 360)
    lon_slice2 = slice(0, nuts_bounds[2])

    # analyze the heatwave days
    heatwaves_days_reference = heatwave_days.sel(
        year=slice(
            Variables.year_reference_start.value, Variables.year_reference_end.value
        )
    ).mean(dim="year")
    heatwave_days_delta = heatwave_days - heatwaves_days_reference

    part1 = heatwave_days_delta.sel(latitude=lat_slice, longitude=lon_slice1)
    part1["longitude"] = part1.longitude - 360

    heatwave_days_delta_eu = xr.concat(
        [
            part1,
            heatwave_days_delta.sel(latitude=lat_slice, longitude=lon_slice2),
        ],
        "longitude",
    ).load()

    heatwave_days_delta_eu.heatwaves_days.mean(dim="year").plot()
    plt.show()

    heatwave_days_delta_eu.to_netcdf(
        Dirs.results_interim.value / "heatwave_days_delta_eu.nc"
    )

    part1 = heatwave_days.sel(latitude=lat_slice, longitude=lon_slice1)
    part1["longitude"] = part1.longitude - 360

    heatwave_days_eu = xr.concat(
        [
            part1,
            heatwave_days.sel(latitude=lat_slice, longitude=lon_slice2),
        ],
        "longitude",
    ).load()

    heatwave_days_eu.heatwaves_days.mean(dim="year").plot()
    plt.show()

    heatwave_days_eu.to_netcdf(Dirs.results_interim.value / "heatwave_days_eu.nc")


def nuts2_id_as_int():

    heatwave_days_delta_eu = xr.open_dataset(
        Dirs.results_interim.value / "heatwave_days_delta_eu.nc", decode_cf=True
    )

    # Generate NUTS code grid
    # - split out each NUTS level
    # - assign numeric code
    # - burn to grid based on ref grid
    nuts, nuts2 = get_nuts_level_2()

    transform = transform_raster_size(heatwave_days_delta_eu)

    image = features.rasterize(
        [(row.geometry, row.id_as_int) for idx, row in nuts2.iterrows()],
        out_shape=heatwave_days_delta_eu.heatwaves_days.shape[1:],
        transform=transform,
        dtype=np.uint32,
    )

    # Make NetCDF matching the heatwave data
    raster_nuts_grid = xr.DataArray(
        image,
        {
            "latitude": heatwave_days_delta_eu.latitude,
            "longitude": heatwave_days_delta_eu.longitude,
        },
        name="eu_nuts2_id_as_int",
    )
    raster_nuts_grid.to_netcdf(
        Dirs.rasters.value / "eu_nuts2_id_as_int.nc",
        encoding={"eu_nuts2_id_as_int": {"_FillValue": 0}},
    )

    # Overview of NUTS regions
    plt.imshow(raster_nuts_grid)
    plt.show()

    nuts_2021 = get_nuts_2021()
    image = features.rasterize(
        [(row.geometry, row.id_as_int) for idx, row in nuts_2021.iterrows()],
        out_shape=heatwave_days_delta_eu.heatwaves_days.shape[1:],
        transform=transform,
        dtype=np.uint32,
    )

    # Make NetCDF matching the heatwave data
    raster_nuts_2021_grid = xr.DataArray(
        image,
        {
            "latitude": heatwave_days_delta_eu.latitude,
            "longitude": heatwave_days_delta_eu.longitude,
        },
        name="eu_nuts2_id_as_int",
    )

    raster_nuts_2021_grid.to_netcdf(
        Dirs.rasters.value / "eu_nuts2021_id_as_int.nc",
        encoding={"eu_nuts2_id_as_int": {"_FillValue": 0}},
    )

    # Overview of NUTS regions
    plt.imshow(raster_nuts_2021_grid)
    plt.show()


def re_grid_population():
    """
    Load the GPWv4 5min data, slice, and regrid

    > NOTE: Using GPW data because GEOSTAT is on EU grid reference and it's a huge pain to re-grid it, for little gain in
    the end (especially since I already have the preparation routines for GPW). Eventually the advantage of going to
    GEOSTAT is that it's based on several years worth of census data while (I now know) GPW is modelled on 2010 + UN
    population change projections -> so the GEOSTAT should be more accurate (I don't think GPW bothered to add extra
    corrections). Other alternative is Landscan which *does* do yearly data ingestion with various things. Can see with
    the group discussions about using it -> doesn't go back very far which is a shame for the heatwaves work. Finally,
    I think it's also worth the get the regridding working - GPW is a good start since I know the data, and it's on a
    'sensible' grid. Good way to get into using the ESMF regrid routines instead of homebaked/half-baked methods of my
    own.
    """
    population_file = (
        Dirs.data_pop_gpw.value / "gpw_v4_population_count_adjusted_rev11_2pt5_min.nc"
    )
    population_dense_file = (
        Dirs.data_pop_gpw.value / "gpw_v4_population_density_adjusted_rev11_2pt5_min.nc"
    )
    population_var = "UN WPP-Adjusted Population Count, v4.11 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes"
    dense_var = "UN WPP-Adjusted Population Density, v4.11 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes"

    population = xr.open_dataset(population_file)
    population_dense = xr.open_dataset(population_dense_file)

    nuts, _, nuts_bounds, lat_slice, lon_slice = get_nuts_level_2(
        return_bounds_slices=True
    )

    heatwave_days_delta_eu = xr.open_dataset(
        Dirs.results_interim.value / "heatwave_days_delta_eu.nc", decode_cf=True
    )
    reference_grid = heatwave_days_delta_eu.sel(year=2000, drop=True).drop_vars(
        "heatwaves_days"
    )

    # Regrid using population density

    # Use 'conservative' method to regrid on population density then re-multiply by pixel areas to get totals. Result is
    # -3.6% different w.r.t the original population count. Think this is FINE for now - would like to cross-check with
    # the 30second data (1km) using just grouping of points without fancy regrid, but it's a ton of extra data to mess
    # with.
    # NUTS shapefiles

    reference_grid["latitude"] = reference_grid.latitude.astype(float)
    reference_grid["longitude"] = reference_grid.longitude.astype(float)
    pixel_size = 0.1

    # Areas in km2 - calculation is in m2
    areas = area_of_pixel(pixel_size, reference_grid.latitude.values) / (1000 * 1000)
    areas = xr.DataArray(
        areas, dims=["latitude"], coords={"latitude": reference_grid.latitude}
    )
    ds_out = xr.Dataset(
        {
            "latitude": (["latitude"], reference_grid.latitude.data),
            "longitude": (["longitude"], reference_grid.longitude.data),
        }
    )

    ref_slice = population_dense.sel(raster=1, drop=True).sel(
        latitude=lat_slice, longitude=lon_slice
    )

    # NOTE: CF extension pretty flaky, doesn't
    regridder = xe.Regridder(ref_slice, ds_out, method="conservative")

    rasters_i = [1, 2, 3, 4, 5]
    population_dense = population_dense.sel(raster=rasters_i, drop=True).sel(
        latitude=lat_slice, longitude=lon_slice
    )
    pop_dense_regrid = regridder(population_dense[dense_var])
    pop_regrid = pop_dense_regrid * areas

    pop_regrid = pop_regrid.rename({"raster": "year"})

    pop_regrid["year"] = [2000, 2005, 2010, 2015, 2020]
    p1 = population.sel(latitude=lat_slice, longitude=lon_slice, raster=rasters_i).sum(
        dim=["latitude", "longitude"]
    )
    p2 = pop_regrid.sum(dim=["latitude", "longitude"])
    p1[population_var].plot()
    plt.tight_layout()
    plt.show()

    p1 = p1.to_dataframe()
    p2 = p2.to_dataframe(name="regrid")
    df = p2.copy()
    p1.index = [2000, 2005, 2010, 2015, 2020]
    df["original"] = p1

    f, ax = plt.subplots(constrained_layout=True)
    df.plot.bar(ax=ax)
    plt.tight_layout()
    plt.show()

    print(df.original / df.regrid)

    # Interpolate pop counts to get yearly
    pop_regrid = pop_regrid.interp(
        {
            "year": np.arange(
                Variables.year_reference_start.value, Variables.year_report.value
            )
        },
        kwargs=dict(fill_value="extrapolate"),
    )

    f, ax = plt.subplots(constrained_layout=True)
    pop_regrid.mean(dim=["latitude", "longitude"]).plot(ax=ax)
    plt.show()

    f, ax = plt.subplots(constrained_layout=True)
    pop_regrid.sel(year=Variables.year_max_analysis.value).plot(robust=True, ax=ax)
    plt.show()

    pop_regrid.to_netcdf(Dirs.results_interim.value / "population_regridded.nc")


def demographics_gpw_nuts_weighted():

    pop_regrid = xr.open_dataset(Dirs.results_interim.value / "population_regridded.nc")

    nuts, nuts2, nuts_bounds, lat_slice, lon_slice = get_nuts_level_2(
        return_bounds_slices=True
    )

    # Regrid GPW demographics, From the info file, over-65s is layer 15
    # todo need to get the new demographics file
    demog_file = (
        Dirs.data_pop_gpw.value
        / "gpw_v4_basic_demographic_characteristics_rev11_bt_2010_dens_2pt5_min.nc"
    )
    var = "Basic Demographic Characteristics, v4.10 (2010): Both, Density, 2.5 arc-minutes"

    demog = (
        xr.open_dataset(demog_file)
        .sel(raster=slice(2, 15), drop=True)
        .sel(latitude=lat_slice, longitude=lon_slice)[var]
    )
    demog["raster"] = np.arange(0, 65 + 1, 5)
    demog = demog.rename({"raster": "age_band_lower_bound"})

    demog.sel(age_band_lower_bound=65).plot(robust=True)
    plt.show()

    # Apply matched NUTS2 demographic data
    eu_demog = pd.read_table(
        Dirs.rasters.value / "demo_r_pjangroup.tsv", na_values=[": ", ": c", ":"]
    )
    eu_demog = prep_eurostat_table(eu_demog)
    eu_demog = eu_demog.droplevel(0).loc["T"]
    eu_demog.index.names = ["age_group", "nuts"]
    eu_demog = eu_demog[sorted(eu_demog.columns)]
    eu_demog_total = (
        eu_demog.to_xarray()
        .to_array("year")
        .sel(age_group="TOTAL")
        .ffill("year")
        .bfill("year")
    )

    eu_demog_x = eu_demog.to_xarray().to_array("year").ffill("year").bfill("year")
    over_65_names = ["Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y_GE85"]
    eu_demog_over65 = eu_demog_x.sel(age_group=over_65_names).sum(dim="age_group")
    eu_demog_x = eu_demog_x.drop_sel(
        age_group=over_65_names + ["TOTAL", "UNK", "Y_GE75", "Y_GE80"]
    )
    # Generate age band lower bound with same order as agr group label in Eurostat data
    age_band_lower_bound = [
        g.item().replace("Y", "").split("-")[0] for g in eu_demog_x.age_group
    ]
    i = age_band_lower_bound.index("_LT5")
    age_band_lower_bound[i] = 0
    age_band_lower_bound = list(map(int, age_band_lower_bound))
    eu_demog_x["age_group"] = age_band_lower_bound
    eu_demog_x = eu_demog_x.sortby("age_group")
    eu_demog_x = xr.concat(
        [eu_demog_x, eu_demog_over65.expand_dims({"age_group": [65]})], dim="age_group"
    )
    eu_demog_x = eu_demog_x.rename({"age_group": "age_band_lower_bound"})
    eu_demog_f = eu_demog_x.sel(nuts=nuts2.id.values) / eu_demog_total.sel(
        nuts=nuts2.id.values
    ).drop_vars("age_group")

    eu_demog_f = eu_demog_f.interp(
        {
            "year": np.arange(
                Variables.year_reference_start.value, Variables.year_report.value
            )
        },
        kwargs=dict(fill_value="extrapolate"),
    )

    eu_demog_f = eu_demog_f.sel(year=pop_regrid.year.data)

    demographics = xr.DataArray(
        np.zeros(
            (
                len(pop_regrid.year),
                len(pop_regrid.latitude),
                len(pop_regrid.longitude),
                len(eu_demog_f.age_band_lower_bound),
            )
        ),
        coords=[
            pop_regrid.year,
            pop_regrid.latitude,
            pop_regrid.longitude,
            eu_demog_f.age_band_lower_bound,
        ],
        name="demographics",
    )

    raster_nuts_grid = xr.open_dataarray(
        Dirs.rasters.value / "eu_nuts2_id_as_int.nc"
    ).squeeze()

    for _, row in tqdm(nuts2.iterrows(), total=len(nuts2)):
        mask = raster_nuts_grid == row.id_as_int

        demog = pop_regrid.where(mask) * eu_demog_f.sel(nuts=row.id, drop=True)

        demographics = xr.where(mask, demog, demographics)

    # demographics.name = "demographic_count"
    demographics.to_netcdf(
        Dirs.results_interim.value / "demographics_gpw_nuts_weighted.nc"
    )


def calculate_infants():

    demographics = xr.open_dataarray(
        Dirs.results_interim.value / "demographics_gpw_nuts_weighted.nc"
    )
    pop_regrid = xr.open_dataset(Dirs.results_interim.value / "population_regridded.nc")

    mask = demographics.sum(dim="age_band_lower_bound").max(dim="year") > 0

    data_plot = (pop_regrid * mask).sum(dim=["latitude", "longitude"])
    data_plot = data_plot.rename({"__xarray_dataarray_variable__": "demographic_count"})
    data_plot.demographic_count.plot()

    demographics.sum(dim=["latitude", "longitude", "age_band_lower_bound"]).plot(
        linestyle="dashed"
    )
    plt.show()

    nuts, nuts2 = get_nuts_level_2()

    """
    # Estimate the distribution of newborns by:
    
    > NOTE: for Eurostat data, instead use directly the number of births (otherwise birth rate is given per woman). use 
    the live births by age and agg as needed otherwise is given at NUTS3 level instead of NUTS2 (could be extended later)
    
    births = country_pop * CBR
    deaths = IMR * births
    infants = births - deaths = births * (1 - (IMR / 1000)
    
    (by NUTS region)
    
    > NOTE: make sure to forward and backfill data since missing for many countries
    
    Get the gridded number of under 5s for each country
    
    Calculate a gridded weight by dividing by the sum of the under 5s for each country-> using the GPW data so just for 2010
    
    $\Sigma w_i * c_i = c_{total}$
    $\Sigma w_i = 1$
    
    Calculate the total number of newborns for a country as CBR * country pop (technically should be related to the total 
    Births value, but the latter seems to be total over 5 years so anyway need to dick about to estimate for one year)
    
    Calculate the gridded number of newborns per cell using the weights
    
    $w_i * b$
    """

    eu_imr = pd.read_table(Dirs.rasters.value / "demo_r_minfind.tsv.gz")
    eu_imr = prep_eurostat_table(eu_imr).loc["RT"]
    eu_births = pd.read_table(Dirs.rasters.value / "demo_r_fagec.tsv.gz")
    eu_births = prep_eurostat_table(eu_births).loc["NR"].loc["TOTAL"]
    eu_imr = eu_imr.ffill(axis=1).bfill(axis=1)
    eu_births = eu_births.ffill(axis=1).bfill(axis=1)
    eu_infants = eu_births * (1 - (eu_imr / 1000))

    eu_infants.index.name = "nuts"
    eu_infants_x = eu_infants.to_xarray().to_array("year")

    eu_infants_x = eu_infants_x.interp(
        {
            "year": np.arange(
                Variables.year_reference_start.value, Variables.year_report.value
            )
        },
        kwargs=dict(fill_value="extrapolate"),
    )
    infants = xr.DataArray(
        np.zeros(
            (
                len(pop_regrid.year),
                len(pop_regrid.latitude),
                len(pop_regrid.longitude),
            )
        ),
        coords=[
            pop_regrid.year,
            pop_regrid.latitude,
            pop_regrid.longitude,
        ],
        name="infants",
    )

    raster_nuts_grid = xr.open_dataarray(
        Dirs.rasters.value / "eu_nuts2_id_as_int.nc"
    ).squeeze()

    for _, row in tqdm(nuts2.iterrows(), total=len(nuts2)):
        mask = raster_nuts_grid == row.id_as_int

        n_infants = eu_infants_x.sel(year=demographics.year, nuts=row.id, drop=True)

        # extract demographics and turn into spatial weights
        weights = demographics.sel(age_band_lower_bound=0).where(mask)
        weights = weights / weights.sum()

        # multiply wieghts with infant number
        weighted_infances = n_infants * weights
        infants = xr.where(mask, weighted_infances, infants)

    infants.to_netcdf(Dirs.results_interim.value / "infants_number.nc")

    infants.sel(year=Variables.year_max_analysis.value).plot(robust=True)
    plt.show()


def calculate_heatwaves_exposure():
    # import data
    infants = xr.open_dataset(Dirs.results_interim.value / "infants_number.nc")
    infants = infants.eu_nuts2_id_as_int

    heatwave_days_delta_eu = xr.open_dataset(
        Dirs.results_interim.value / "heatwave_days_delta_eu.nc", decode_cf=True
    )

    demographics = xr.open_dataarray(
        Dirs.results_interim.value / "demographics_gpw_nuts_weighted.nc"
    )

    heatwave_days_eu = xr.open_dataset(
        Dirs.results_interim.value / "heatwave_days_eu.nc"
    )

    # TODO also need to do the per-nuts calculations with the mask. Can use 2021 version for consistency.

    # Calculate exposures -> changes and totals
    hw_exposure_change = xr.concat(
        [
            (infants * heatwave_days_delta_eu.heatwaves_days).expand_dims(
                {"age_band": ["LT_1"]}
            ),
            (
                demographics.sel(age_band_lower_bound=65)
                * heatwave_days_delta_eu.heatwaves_days
            ).expand_dims({"age_band": ["GT_65"]}),
        ],
        "age_band",
    ).drop_vars("age_band_lower_bound")

    hw_exposure_change.to_netcdf(
        Dirs.results_interim.value / "heatwave_exposure_change_total_eu.nc"
    )

    hw_exposure = xr.concat(
        [
            (infants * heatwave_days_eu).expand_dims({"age_band": ["LT_1"]}),
            (demographics.sel(age_band_lower_bound=65) * heatwave_days_eu).expand_dims(
                {"age_band": ["GT_65"]}
            ),
        ],
        "age_band",
    ).drop_vars("age_band_lower_bound")

    hw_exposure.to_netcdf(Dirs.results_interim.value / "heatwave_exposure_total_eu.nc")

    # no longer exporting these files since they are not used
    # mask = hw_exposure.sum(dim="age_band").max(dim="year") > 0
    #
    # hw_exposure_change.where(mask).to_dataframe().dropna().to_csv(
    #     Dirs.results.value / "heatwave_exposure_change_grid_eu.csv.zip"
    # )
    #
    # hw_exposure.where(mask).to_dataframe().dropna().to_csv(
    #     Dirs.results.value / "heatwave_exposure_total_grid_eu.csv"
    # )


def calculate_exposures_nuts_2021():
    # import data
    hw_exposure_change = xr.open_dataarray(
        Dirs.results_interim.value / "heatwave_exposure_change_total_eu.nc"
    )
    hw_exposure = xr.open_dataarray(
        Dirs.results_interim.value / "heatwave_exposure_total_eu.nc"
    )
    infants = xr.open_dataset(Dirs.results_interim.value / "infants_number.nc")
    infants = infants.eu_nuts2_id_as_int

    demographics = xr.open_dataarray(
        Dirs.results_interim.value / "demographics_gpw_nuts_weighted.nc"
    )

    heatwave_days_eu = xr.open_dataset(
        Dirs.results_interim.value / "heatwave_days_eu.nc"
    )

    nuts_2021 = get_nuts_2021()

    raster_nuts_2021_grid = xr.open_dataarray(
        Dirs.rasters.value / "eu_nuts2021_id_as_int.nc"
    ).squeeze()

    summary_change = {}
    for nuts_code, row in tqdm(nuts_2021.iterrows(), total=len(nuts_2021)):
        mask = raster_nuts_2021_grid == row.id_as_int

        exp = hw_exposure_change.where(mask).sum(dim=["latitude", "longitude"])
        exp.name = nuts_code
        summary_change[nuts_code] = exp

    summary_change = xr.merge(summary_change.values()).to_array("nuts")
    summary_change.name = "exposures_to_change"

    vuln_demog = xr.concat(
        [
            infants.expand_dims({"age_band": ["LT_1"]}),
            demographics.sel(age_band_lower_bound=65).expand_dims(
                {"age_band": ["GT_65"]}
            ),
        ],
        "age_band",
    ).drop_vars("age_band_lower_bound")

    nuts_hw = {}
    nuts_demog = {}
    nuts_exposures = {}
    nuts_exposures_norm = {}

    # heatwave_days_eu = heatwave_days_eu.heatwaves_days
    # hw_exposure = hw_exposure.heatwaves_days

    for nuts_code, row in tqdm(nuts_2021.iterrows(), total=len(nuts_2021)):
        mask = raster_nuts_2021_grid == row.id_as_int
        hws = heatwave_days_eu.where(mask).mean(dim=["latitude", "longitude"])
        hws.name = nuts_code
        nuts_hw[nuts_code] = hws

        dem = vuln_demog.where(mask).sum(dim=["latitude", "longitude"])
        dem.name = nuts_code
        nuts_demog[nuts_code] = dem

        exp = hw_exposure.where(mask).sum(dim=["latitude", "longitude"])
        exp.name = nuts_code
        nuts_exposures[nuts_code] = exp
        nuts_exposures_norm[nuts_code] = exp / dem

    nuts_hw = xr.merge(nuts_hw.values()).to_array("nuts")
    nuts_hw.name = "heatwave_days"

    nuts_demog = xr.merge(nuts_demog.values()).to_array("nuts")
    nuts_demog.name = "population"

    nuts_exposures = xr.merge(nuts_exposures.values()).to_array("nuts")
    nuts_exposures.name = "exposures"

    nuts_exposures_norm = xr.merge(nuts_exposures_norm.values()).to_array("nuts")
    nuts_exposures_norm.name = "days_heatwave"

    age_band = "GT_65"

    plot_data = nuts_2021.to_crs("EPSG:3035")
    plot_data["plot_var"] = (
        nuts_demog.sel(year=2021, age_band=age_band).to_dataframe().population
    )
    plot_data = plot_data.dropna(subset=["plot_var"])

    ax = plot_data.plot("plot_var", legend=True)
    plt.tight_layout()
    plt.show()

    age_band = "LT_1"

    plot_data = nuts_2021.to_crs("EPSG:3035")
    plot_data["plot_var"] = (
        nuts_demog.sel(year=2021, age_band=age_band).to_dataframe().population
    )
    plot_data = plot_data.dropna(subset=["plot_var"])

    ax = plot_data.plot("plot_var", legend=True)
    plt.tight_layout()
    plt.show()

    # Export Summary
    summary_change.to_netcdf(
        Dirs.results_interim.value / "exposure_change_by_nuts2021.nc"
    )
    nuts_exposures.to_netcdf(
        Dirs.results_interim.value / "exposure_totals_by_nuts2021.nc"
    )
    nuts_exposures_norm.to_netcdf(
        Dirs.results_interim.value / "exposure_norm_by_nuts2021.nc"
    )
    nuts_demog.to_netcdf(Dirs.results_interim.value / "demographics_by_nuts2021.nc")
    nuts_hw.to_netcdf(Dirs.results_interim.value / "heatwave_days_by_nuts2021.nc")
    vuln_demog.to_netcdf(
        Dirs.results_interim.value / "vulnerability_demographics_by_nuts2021.nc"
    )
    summary_change.to_dataframe().reset_index().to_csv(
        Dirs.results.value / "exposure_change_by_nuts2021.csv"
    )
    nuts_exposures.to_dataframe().reset_index().to_csv(
        Dirs.results.value / "exposure_totals_by_nuts2021.csv"
    )
    nuts_exposures_norm.to_dataframe().reset_index().to_csv(
        Dirs.results.value / "exposure_norm_by_nuts2021.csv"
    )


def plot_heatwave_exposure():

    hw_exposure_change = xr.open_dataarray(
        Dirs.results_interim.value / "heatwave_exposure_change_total_eu.nc"
    )
    hw_exposure = xr.open_dataarray(
        Dirs.results_interim.value / "heatwave_exposure_total_eu.nc"
    )

    exposure_change_eu = hw_exposure_change.sum(
        dim=["latitude", "longitude"]
    ).to_dataframe("exposures")
    exposure_change_eu = exposure_change_eu.unstack().T.loc["exposures"]
    exposure_change_eu.to_csv(Dirs.results.value / "exposure_change_eu.csv")
    print("Exposure change EU \n", exposure_change_eu.tail(5))

    exposure_total_eu = hw_exposure.sum(dim=["latitude", "longitude"]).to_dataframe(
        "exposures"
    )
    exposure_total_eu = exposure_total_eu.unstack().T.loc["exposures"]
    exposure_total_eu.to_csv(Dirs.results.value / "exposure_total_eu.csv")
    print("Exposure Total EU \n", exposure_total_eu.tail(5))

    f, axs = plt.subplots(
        2,
        1,
        squeeze=True,
        sharex=True,
    )
    plot_data = (
        hw_exposure_change.sum(dim=["latitude", "longitude"])
        .to_dataframe("exposures")
        .unstack()
        .T.loc["exposures"]
    )
    plot_data_inf = plot_data.LT_1 / 1e6
    plot_data_inf_baseline = plot_data_inf[
        plot_data_inf.index <= Variables.year_reference_end.value
    ]
    plot_data_inf.plot.bar(ax=axs[0])
    plot_data_inf_baseline.plot.bar(ax=axs[0], color="lightblue")
    axs[0].set(title="Infants", ylabel="", xlabel="Year")
    axs[0].grid(ls="--", color="lightgrey", linewidth=0.5, axis="y")

    plot_data_old = plot_data.GT_65 / 1e6
    plot_data_old_baseline = plot_data_old[
        plot_data_old.index <= Variables.year_reference_end.value
    ]
    plot_data_old.plot.bar(ax=axs[1])
    plot_data_old_baseline.plot.bar(ax=axs[1], color="lightblue")
    axs[1].set(title="Over 65", ylabel="", xlabel="Year")
    plt.tight_layout(rect=(0.02, 0, 1, 1))
    f.text(
        0.01,
        0.5,
        "Excess heatwave person-days [million]",
        va="center",
        rotation="vertical",
    )
    axs[1].set_xlim(-0.5, plot_data.shape[0] + 0.5)
    axs[1].set_xticks(np.arange(0, plot_data.shape[0], 1))
    axs[1].set_xticklabels(plot_data.index)
    axs[1].grid(ls="--", color="lightgrey", linewidth=0.5, axis="y")
    sns.despine()
    f.savefig(Dirs.figures.value / "excess heatwaves person days.png")
    plt.show()

    f, axs = plt.subplots(
        2,
        1,
        squeeze=True,
        sharex=True,
    )
    plot_data = (
        hw_exposure.sum(dim=["latitude", "longitude"])
        .to_dataframe(name="exposures")
        .unstack()
        .T.loc["exposures"]
    )
    plot_data_inf = plot_data.LT_1 / 1e6
    plot_data_inf_baseline = plot_data_inf[
        plot_data_inf.index <= Variables.year_reference_end.value
    ]
    plot_data_inf.plot.bar(ax=axs[0])
    plot_data_inf_baseline.plot.bar(ax=axs[0], color="lightblue")
    axs[0].set(title="Infants", ylabel="", xlabel="Year")
    axs[0].grid(ls="--", color="lightgrey", linewidth=0.5, axis="y")

    plot_data_old = plot_data.GT_65 / 1e6
    plot_data_old_baseline = plot_data_old[
        plot_data_old.index <= Variables.year_reference_end.value
    ]
    plot_data_old.plot.bar(ax=axs[1])
    plot_data_old_baseline.plot.bar(ax=axs[1], color="lightblue")
    axs[1].set(title="Over 65", ylabel="", xlabel="Year")
    axs[1].set_xlim(-0.5, plot_data.shape[0] + 0.5)
    axs[1].set_xticks(np.arange(0, plot_data.shape[0], 1))
    axs[1].set_xticklabels(plot_data.index)
    axs[1].grid(ls="--", color="lightgrey", linewidth=0.5, axis="y")
    plt.tight_layout(rect=(0.02, 0, 1, 1))
    f.text(
        0.01, 0.5, "Heatwave person-days [million]", va="center", rotation="vertical"
    )
    sns.despine()
    f.savefig(Dirs.figures.value / "total heatwaves person days.png")
    plt.show()


def calc_normed_exposure_eu_level():

    heatwave_exposure = xr.open_dataarray(
        Dirs.results_interim.value / "heatwave_exposure_total_eu.nc"
    )
    demographics = xr.open_dataarray(
        Dirs.results_interim.value / "demographics_gpw_nuts_weighted.nc"
    )

    # Aggregate all EU
    ttl = (
        heatwave_exposure.sum(dim=["latitude", "longitude"])
        .to_dataframe("exposures")
        .unstack()
        .T.loc["exposures"]
        # .to_csv(Dirs.results.value / 'exposure_total_eu.csv')
    )

    dmg = demographics.sel(age_band_lower_bound=65)
    dmg = dmg.expand_dims({"age_band": ["GT_65"]})
    dmg = dmg.sum(dim=["latitude", "longitude"]).to_dataframe(name="demographic_count")

    nrm = ttl["GT_65"] / dmg.loc["GT_65", "demographic_count"]
    nrm = nrm.to_frame(name="exposures")
    nrm.to_csv(Dirs.results.value / "normed_exposures_gt_65_eu.csv")


def calc_normed_exposure_country_level():

    nuts_2021 = get_nuts_2021()
    nuts_exposures = xr.open_dataset(
        Dirs.results_interim.value / "exposure_totals_by_nuts2021.nc"
    )
    nuts_demog = xr.open_dataarray(
        Dirs.results_interim.value / "demographics_by_nuts2021.nc"
    )

    # Aggregate by Europe countries LCDE groupings

    # To avoid going crazy, just do the averages by NUTS regions rather than re-rasterizing at the region level. should
    # be a good enough approximation. For the total exposures its a straight sum so no issues there.
    country_demog = (
        nuts_demog.to_dataset()
        .merge(nuts_2021[["NUTS_NAME", "CNTR_CODE"]].rename_axis("nuts"))
        .set_coords("CNTR_CODE")
        .to_dataframe()
        .reset_index()
    )
    country_exposures = (
        nuts_exposures.merge(nuts_2021[["NUTS_NAME", "CNTR_CODE"]].rename_axis("nuts"))
        .set_coords("CNTR_CODE")
        .to_dataframe()
        .reset_index()
    )
    tots_age_band = country_exposures.groupby(["year", "age_band", "CNTR_CODE"]).sum()
    tots_age_band.to_csv(Dirs.results.value / "total_exposures_by_age_band_country.csv")
    tots = country_exposures.groupby(["year", "CNTR_CODE"]).sum()
    tots.to_csv(Dirs.results.value / "total_exposures_country.csv")
    nrm = (
        country_exposures.groupby(["year", "CNTR_CODE"]).sum().exposures
        / country_demog.groupby(["year", "CNTR_CODE"]).sum().population
    )
    nrm = nrm.to_frame(name="exposures")
    nrm.to_csv(Dirs.results.value / "normed_exposures_country.csv")

    # nrm = pd.read_csv(Dirs.results.value / "normed_exposures_country.csv")

    print(nrm)

    nrm_age_band = (
        country_exposures.groupby(["year", "age_band", "CNTR_CODE"]).sum().exposures
        / country_demog.groupby(["year", "age_band", "CNTR_CODE"]).sum().population
    )
    nrm_age_band = nrm_age_band.to_frame(name="exposures")

    nrm_age_band.to_csv(Dirs.results.value / "normed_exposures_by_age_band_country.csv")


def calc_normed_exposure_region_level():

    nuts_exposures = xr.open_dataset(
        Dirs.results_interim.value / "exposure_totals_by_nuts2021.nc"
    )
    nuts_2021 = get_nuts_2021()
    nuts_demog = xr.open_dataarray(
        Dirs.results_interim.value / "demographics_by_nuts2021.nc"
    )

    reg_colors = {
        "Southern Europe": "#EE7733",
        "Western Europe": "#009988",
        "Eastern Europe": "#EE3377",
        "Northern Europe": "#33BBEE",
        "Western Asia": "#CC3311",
        "Central Asia": "#0077BB",
    }

    # just do the averages by NUTS regions rather than re-rasterizing at the region level. should be a good enough
    # approximation. For the total exposures its a straigh sum so no issues there.
    country_names_groups = pd.read_excel(
        Dirs.rasters.value / "[LCDE 2024] Country names and groupings.xlsx", header=1
    )
    country_names_groups["European sub-region (UN geoscheme)"] = country_names_groups[
        "European sub-region (UN geoscheme)"
    ].str.strip()
    c = country_names_groups.set_index("ISO 3166-1 Alpha-2 code").rename_axis(
        "CNTR_CODE"
    )[
        [
            "Country name",
            "European sub-region (UN geoscheme)",
            "WB income group (2023)",
            "HDI index (2023)",
        ]
    ]
    region_demog = (
        nuts_demog.to_dataset()
        .merge(nuts_2021[["NUTS_NAME", "CNTR_CODE"]].rename_axis("nuts"))
        .set_coords("CNTR_CODE")
        .to_dataframe()
        .reset_index()
        .merge(c, left_on="CNTR_CODE", right_index=True)
    )
    region_exposures = (
        nuts_exposures.merge(nuts_2021[["NUTS_NAME", "CNTR_CODE"]].rename_axis("nuts"))
        .set_coords("CNTR_CODE")
        .to_dataframe()
        .reset_index()
        .merge(c, left_on="CNTR_CODE", right_index=True)
    )

    tots_age_band = region_exposures.groupby(
        ["year", "age_band", "European sub-region (UN geoscheme)"]
    ).sum()
    tots_age_band.to_csv(
        Dirs.results.value / "total_exposures_by_age_band_eu_sub_region.csv"
    )
    tots = region_exposures.groupby(
        ["year", "European sub-region (UN geoscheme)"]
    ).sum()
    tots.to_csv(Dirs.results.value / "total_exposures_eu_sub_region.csv")

    ax = sns.lineplot(
        data=tots.reset_index(),
        x="year",
        y="exposures",
        hue="European sub-region (UN geoscheme)",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        title=None,
        frameon=False,
    )
    sns.despine()
    plt.tight_layout()
    plt.show()

    nrm = (
        region_exposures.groupby(["year", "European sub-region (UN geoscheme)"])
        .sum()
        .exposures
        / region_demog.groupby(["year", "European sub-region (UN geoscheme)"])
        .sum()
        .population
    )
    nrm = nrm.to_frame(name="exposures")
    nrm.to_csv(Dirs.results.value / "normed_exposures_eu_sub_region.csv")
    nrm = pd.read_csv(Dirs.results.value / "normed_exposures_eu_sub_region.csv")

    print(nrm)

    nrm_age_band = (
        region_exposures.groupby(
            ["year", "age_band", "European sub-region (UN geoscheme)"]
        )
        .sum()
        .exposures
        / region_demog.groupby(
            ["year", "age_band", "European sub-region (UN geoscheme)"]
        )
        .sum()
        .population
    )
    nrm_age_band = nrm_age_band.to_frame(name="exposures")

    nrm_age_band.to_csv(
        Dirs.results.value / "normed_exposures_by_age_band_eu_sub_region.csv"
    )
    nrm_age_band = pd.read_csv(
        Dirs.results.value / "normed_exposures_by_age_band_eu_sub_region.csv"
    ).set_index(["year", "age_band", "European sub-region (UN geoscheme)"])

    sns.color_palette(reg_colors.values())
    ax = sns.lineplot(
        data=nrm.reset_index(),
        x="year",
        y="exposures",
        hue="European sub-region (UN geoscheme)",
    )
    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        title=None,
        frameon=False,
    )
    sns.despine()
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(
        2,
        1,
        sharey=True,
        sharex=True,
        figsize=(7, 5),
    )

    plot_data = (
        nrm_age_band.to_xarray()
        # .rolling(year=10)
        # .mean()
        .sel(age_band="GT_65", drop=True).to_dataframe()
    )
    df_summary = plot_data.unstack(level=1)
    df_summary.columns = df_summary.columns.droplevel(level=0)
    print(df_summary.tail(5).to_markdown())

    ax = sns.lineplot(
        data=plot_data.reset_index(),
        x="year",
        y="exposures",
        hue="European sub-region (UN geoscheme)",
        palette=reg_colors,
        ax=axs[0],
    )
    ax.set(
        title="Over 65",
        ylabel="",
        xlabel="",
    )
    ax.grid(ls="--", color="lightgray")
    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=3,
        title=None,
        frameon=False,
    )

    plot_data = (
        nrm_age_band.to_xarray()
        # .rolling(year=10)
        # .mean()
        .sel(age_band="LT_1", drop=True).to_dataframe()
    )
    df_summary = plot_data.unstack(level=1)
    df_summary.columns = df_summary.columns.droplevel(level=0)
    print(df_summary.tail(5).to_markdown())
    ax = sns.lineplot(
        data=plot_data.reset_index(),
        x="year",
        y="exposures",
        hue="European sub-region (UN geoscheme)",
        palette=reg_colors,
        ax=axs[1],
        legend=False,
    )
    ax.set(
        title="Infants",
        ylabel="",
        xlabel="Year",
        yticks=range(0, 41, 5),
    )
    ax.grid(ls="--", color="lightgray")
    plt.tight_layout(rect=(0.02, 0, 1, 1))
    fig.text(0.01, 0.5, "Mean heatwave days", va="center", rotation="vertical")
    sns.despine()

    # fig.savefig(Dirs.figures.value / "norm_hw_exposure_eu_sub_region.png")
    fig.savefig(Dirs.figures.value / "norm_hw_exposure_eu_sub_region.pdf")
    plt.show()

    plot_data = (
        nrm_age_band.to_xarray()
        .rolling(year=10)
        .mean()
        # .sel(age_band='GT_65', drop=True)
        .to_dataframe()
        .reset_index()
    )

    sns.relplot(
        data=plot_data,
        kind="line",
        x="year",
        y="exposures",
        hue="European sub-region (UN geoscheme)",
        col="age_band",
        palette=reg_colors,
    )
    plt.tight_layout()
    plt.show()

    plot_data = (
        nrm_age_band.to_xarray()
        .rolling(year=10)
        .mean()
        .sel(age_band="GT_65", drop=True)
        .to_dataframe()
    )

    ax = sns.lineplot(
        data=plot_data.reset_index(),
        x="year",
        y="exposures",
        hue="European sub-region (UN geoscheme)",
        palette=reg_colors,
    )
    ax.set_box_aspect(1)
    ax.set(
        title="10 year rolling mean of heatwave exposure, over-65s",
        ylabel="Mean heatwave days",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.savefig(Dirs.results.value / "norm_hw_exposure_over_62_eu_sub_region.png")
    plt.tight_layout()
    plt.show()

    # plot_data = (nrm_age_band
    #              .to_xarray()
    #              .rolling(year=10)
    #              .mean()
    #              .sel(age_band='LT_1', drop=True)
    #              .to_dataframe())
    # ax = sns.lineplot(data=plot_data.reset_index(),
    #                   x='year', y='exposures', hue='European sub-region (UN geoscheme)',
    #                  palette=reg_colors)
    # ax.set(
    #     title='10 year rolling mean of heatwave exposure, infants',
    #     ylabel='Mean heatwave days')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.savefig(Dirs.results.value / 'norm_hw_exposure_infant_eu_sub_region.png')
    # # plot_data
    # # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


def calc_normed_exposure_eea_level():
    nuts_2021 = get_nuts_2021()
    nuts_exposures = xr.open_dataset(
        Dirs.results_interim.value / "exposure_totals_by_nuts2021.nc"
    )
    nuts_demog = xr.open_dataarray(
        Dirs.results_interim.value / "demographics_by_nuts2021.nc"
    )

    # Aggregate by Europe EEA region

    # To avoid going crazy, just do the averages by NUTS regions rather than re-rasterizing at the region level. should be a good enough approximation. For the total exposures its a straigh sum so no issues there.
    country_names_groups = pd.read_excel(
        Dirs.rasters.value / "[LCDE 2024] Country names and groupings.xlsx", header=1
    )
    country_names_groups["EEA sub-region division"] = country_names_groups[
        "EEA sub-region division"
    ].str.strip()
    c = country_names_groups.set_index("ISO 3166-1 Alpha-2 code").rename_axis(
        "CNTR_CODE"
    )[
        [
            "Country name",
            "EEA sub-region division",
            "WB income group (2023)",
            "HDI index (2023)",
        ]
    ]
    region_demog = (
        nuts_demog.to_dataset()
        .merge(nuts_2021[["NUTS_NAME", "CNTR_CODE"]].rename_axis("nuts"))
        .set_coords("CNTR_CODE")
        .to_dataframe()
        .reset_index()
        .merge(c, left_on="CNTR_CODE", right_index=True)
    )
    region_exposures = (
        nuts_exposures.merge(nuts_2021[["NUTS_NAME", "CNTR_CODE"]].rename_axis("nuts"))
        .set_coords("CNTR_CODE")
        .to_dataframe()
        .reset_index()
        .merge(c, left_on="CNTR_CODE", right_index=True)
    )
    tots_age_band = region_exposures.groupby(
        ["year", "age_band", "EEA sub-region division"]
    ).sum()
    tots_age_band.to_csv(
        Dirs.results.value / "total_exposures_by_age_band_eu_eea_region.csv"
    )
    tots = region_exposures.groupby(["year", "EEA sub-region division"]).sum()
    tots.to_csv(Dirs.results.value / "total_exposures_eu_eea_region.csv")
    ax = sns.lineplot(
        data=tots.reset_index(), x="year", y="exposures", hue="EEA sub-region division"
    )
    plt.legend(
        bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
    )  # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    nrm = (
        region_exposures.groupby(["year", "EEA sub-region division"]).sum().exposures
        / region_demog.groupby(["year", "EEA sub-region division"]).sum().population
    )
    nrm = nrm.to_frame(name="exposures")
    nrm.to_csv(Dirs.results.value / "normed_exposures_eu_eea_region.csv")
    nrm_age_band = (
        region_exposures.groupby(["year", "age_band", "EEA sub-region division"])
        .sum()
        .exposures
        / region_demog.groupby(["year", "age_band", "EEA sub-region division"])
        .sum()
        .population
    )
    nrm_age_band = nrm_age_band.to_frame(name="exposures")

    nrm_age_band.to_csv(
        Dirs.results.value / "normed_exposures_by_age_band_eu_eea_region.csv"
    )


def main_results():

    # import data
    heatwave_exposure = xr.open_dataarray(
        Dirs.results_interim.value / "heatwave_exposure_total_eu.nc"
    )
    hw_file = sorted(Dirs.data_era_heatwaves_days.value.glob("*.nc"))
    heatwave_days = xr.open_mfdataset(hw_file).heatwaves_days

    heatwave_days_eu = xr.open_dataset(
        Dirs.results_interim.value / "heatwave_days_eu.nc"
    )
    heatwave_days_eu = heatwave_days_eu.heatwaves_days
    nuts_2021 = get_nuts_2021()
    nuts_exposures_norm = xr.open_dataset(
        Dirs.results_interim.value / "exposure_norm_by_nuts2021.nc"
    )
    nuts_exposures = xr.open_dataset(
        Dirs.results_interim.value / "exposure_totals_by_nuts2021.nc"
    )
    raster_nuts_grid = xr.open_dataarray(
        Dirs.rasters.value / "eu_nuts2_id_as_int.nc"
    ).squeeze()
    nuts_hw = xr.open_dataarray(
        Dirs.results_interim.value / "heatwave_days_by_nuts2021.nc"
    )
    vuln_demog = xr.open_dataarray(
        Dirs.results_interim.value / "vulnerability_demographics_by_nuts2021.nc"
    )

    years_baseline = slice(
        Variables.year_reference_start.value, Variables.year_reference_end.value
    )
    years_comparison = slice(
        Variables.year_min_comparison.value, Variables.year_max_comparison.value
    )

    years_baseline_2024_report = slice(2000, 2009)
    years_comparison_2024_report = slice(2012, 2021)

    def print_results(data, y_slice=None, year=None, text="all groups"):
        if y_slice:
            data_slice = data.sel(year=y_slice)
            text_intro = f"mean from {y_slice.start} to {y_slice.stop}"
        else:
            data_slice = data.sel(year=year)
            text_intro = f"mean for {year}"
        data_sum = data_slice.sum(dim=["latitude", "longitude"])
        if y_slice:
            data_sum = data_sum.mean(dim="year")
        data_sum = data_sum.item()
        print(
            text_intro,
            f"{text:}",
            f"{data_sum/1e9:.4f} billion",
        )
        return data_sum

    print("Heatwave days")
    val_base = print_results(data=heatwave_days_eu, y_slice=years_baseline)
    val_comp = print_results(data=heatwave_days_eu, y_slice=years_comparison)
    print(
        f"the number of heatwave days increased by {(val_comp - val_base) / val_base *100:.0f}%"
    )
    # print_results(data=heatwave_days_eu, y_slice=years_baseline_2024_report)
    # print_results(data=heatwave_days_eu, y_slice=years_comparison_2024_report)

    print("Heatwave person-days")
    val_base = print_results(
        data=heatwave_exposure.sum(dim="age_band"), y_slice=years_baseline
    )
    val_comp = print_results(
        data=heatwave_exposure.sum(dim="age_band"), y_slice=years_comparison
    )
    print(
        f"the number of heatwave person-days increased by {(val_comp - val_base) / val_base *100:.0f}%"
    )
    _ = print_results(
        data=heatwave_exposure.sum(dim="age_band"),
        year=Variables.year_max_analysis.value,
    )
    _ = print_results(
        data=heatwave_exposure.sum(dim="age_band"),
        year=Variables.year_max_analysis.value - 1,
    )
    val_base = print_results(
        data=heatwave_exposure.sum(dim="age_band"),
        y_slice=years_baseline_2024_report,
    )
    val_comp = print_results(
        data=heatwave_exposure.sum(dim="age_band"),
        y_slice=years_comparison_2024_report,
    )
    print(
        f"the number of heatwave person-days increased by {(val_comp - val_base) / val_base *100:.0f}%"
    )

    print_results(
        data=heatwave_exposure.sel(age_band="GT_65"),
        y_slice=years_baseline,
        text="over 65",
    )
    print_results(
        data=heatwave_exposure.sel(age_band="GT_65"),
        y_slice=years_comparison,
        text="over 65",
    )
    _ = print_results(
        data=heatwave_exposure.sel(age_band="GT_65"),
        year=Variables.year_max_analysis.value,
        text="over 65",
    )
    print_results(
        data=heatwave_exposure.sel(age_band="LT_1"),
        y_slice=years_baseline,
        text="under 1",
    )
    print_results(
        data=heatwave_exposure.sel(age_band="LT_1"),
        y_slice=years_comparison,
        text="under 1",
    )

    def _plot_heatwave_nuts(
        _nuts_2021,
        _nuts_exp_norm,
        _years,
        _title,
        _ax,
        _v_min=0,
        _v_max=33,
    ):
        _nuts_filter_year = _nuts_exp_norm.sel(year=_years)
        _nuts_filter_year = _nuts_filter_year.mean(dim="year")
        _nuts_filter_year = _nuts_filter_year.mean(dim="age_band").to_dataframe()
        _nuts_2021["plot_var"] = _nuts_filter_year.days_heatwave
        print("Max days heatwave", _nuts_2021["plot_var"].max())
        _ax = _nuts_2021.to_crs("EPSG:3035").plot(
            "plot_var", legend=False, cmap="Reds", ax=_ax, vmin=_v_min, vmax=_v_max
        )
        _plot_data = _nuts_2021.to_crs("EPSG:3035").dropna(subset=["plot_var"])
        _plot_data.plot(facecolor="none", linewidth=0.05, edgecolor="grey", ax=_ax)
        _ax.set_title(f"{_title} ({_years.start}-{_years.stop})")
        _ax.set_axis_off()

    # I am combining the data for both groups since results are almost the same
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))
    v_min, v_max = 0, 33

    _plot_heatwave_nuts(
        _nuts_2021=nuts_2021,
        _nuts_exp_norm=nuts_exposures_norm,
        _years=years_baseline,
        _title=f"Baseline",
        _ax=axs[0],
        _v_max=v_max,
        _v_min=v_min,
    )
    _plot_heatwave_nuts(
        _nuts_2021=nuts_2021,
        _nuts_exp_norm=nuts_exposures_norm,
        _years=years_comparison,
        _title=f"Comparison",
        _ax=axs[1],
        _v_max=v_max,
        _v_min=v_min,
    )

    plt.tight_layout()

    cax = plt.axes([0.2, 0.1, 0.6, 0.1])
    img = plt.imshow(np.array([[v_min, v_max]]), cmap="Reds")
    img.set_visible(False)
    # todo check the label below
    cb = plt.colorbar(
        orientation="horizontal", cax=cax, label="Mean heatwave days per year"
    )
    cb.outline.set_linewidth(0.25)
    plt.savefig(Dirs.figures.value / "mean_days_of_heatwave.png")
    plt.show()

    f, ax = plt.subplots(1, 1)
    v_min, v_max = 0, 65
    nuts_2021["plot_var"] = (
        nuts_exposures_norm.sel(year=Variables.year_max_analysis.value)
        .mean(dim="age_band")
        .to_dataframe()
        .days_heatwave
    )
    nuts_2021.to_crs("EPSG:3035").plot(
        "plot_var", legend=False, cmap="Reds", ax=ax, vmin=v_min, vmax=v_max
    )
    _plot_data = nuts_2021.to_crs("EPSG:3035").dropna(subset=["plot_var"])
    _plot_data.plot(facecolor="none", linewidth=0.05, edgecolor="grey", ax=ax)
    ax.set_axis_off()
    cax = plt.axes([0.85, 0.2, 0.1, 0.6])
    img = plt.imshow(np.array([[v_min, v_max]]), cmap="Reds", vmax=v_max)
    img.set_visible(False)
    # todo check the label below
    cb = plt.colorbar(
        orientation="vertical",
        cax=cax,
        label=f"Heatwave days per year in {Variables.year_max_analysis.value}",
    )
    cb.outline.set_linewidth(0.25)
    cb.ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        Dirs.figures.value
        / f"mean_days_of_heatwave_{Variables.year_max_analysis.value}.png"
    )
    plt.show()

    nuts_2021["plot_var"] = (
        nuts_exposures_norm.sel(
            year=Variables.year_max_analysis.value, age_band="GT_65"
        )
        .to_dataframe()
        .days_heatwave
    )
    nuts_2021.to_crs("EPSG:3035").plot("plot_var", legend=True, cmap="plasma")
    plt.tight_layout()
    plt.show()

    days_baseline = nuts_hw.sel(year=years_baseline).mean(dim="year")
    days_yr = nuts_hw.sel(year=years_comparison).mean(dim="year")
    plot_var = days_yr - days_baseline

    plot_data = nuts_2021.to_crs("EPSG:3035")
    plot_data["plot_var"] = plot_var.to_dataframe().heatwave_days
    plot_data = plot_data.dropna(subset=["plot_var"])

    ax = plot_data.plot(
        "plot_var",
        legend=True,
        # norm=divnorm,
        vmin=0,
        cmap="Reds",
        legend_kwds={
            "label": "[days]",
        },
    )
    plot_data.plot(facecolor="none", linewidth=0.05, edgecolor="grey", ax=ax)
    ax.set_axis_off()
    ax.set(
        title=f"Change in average heatwave days \n {years_comparison.start}-{years_comparison.stop} relative to baseline"
    )
    ax.figure.savefig(Dirs.results.value / "map_nuts3_days_change_mean.png")
    plt.tight_layout()
    plt.show()

    age_band = "GT_65"
    days_experienced_baseline = nuts_exposures_norm.sel(year=years_baseline).mean(
        dim="year"
    )
    days_experienced_yr = nuts_exposures_norm.sel(year=years_comparison).mean(
        dim="year"
    )

    plot_var = days_experienced_yr - days_experienced_baseline

    plot_data = nuts_2021.to_crs("EPSG:3035")
    plot_data["plot_var"] = plot_var.sel(age_band=age_band).to_dataframe().days_heatwave
    plot_data = plot_data.dropna(subset=["plot_var"])

    ax = plot_data.plot(
        "plot_var",
        legend=True,
        norm=colors.CenteredNorm(),
        cmap="RdBu_r",
        legend_kwds={
            "label": "[days]",
        },
    )
    plot_data.plot(facecolor="none", linewidth=0.05, edgecolor="grey", ax=ax)
    ax.set_axis_off()
    ax.set(
        title="Change in average heatwave days experienced\n by over-65s in 2013-2022 relative to baseline"
    )
    ax.figure.savefig(Dirs.results.value / "map_nuts3_days_change_over_65.png")
    plt.tight_layout()
    plt.show()

    age_band = "LT_1"

    days_experienced_baseline = nuts_exposures_norm.sel(year=years_baseline).mean(
        dim="year"
    )
    days_experienced_yr = nuts_exposures_norm.sel(year=years_comparison).mean(
        dim="year"
    )

    plot_var = days_experienced_yr - days_experienced_baseline

    plot_data = nuts_2021.to_crs("EPSG:3035")
    plot_data["plot_var"] = plot_var.sel(age_band=age_band).to_dataframe().days_heatwave
    plot_data = plot_data.dropna(subset=["plot_var"])

    ax = plot_data.plot(
        "plot_var",
        legend=True,
        norm=colors.CenteredNorm(),
        cmap="RdBu_r",
        legend_kwds={
            "label": "[days]",
        },
    )
    plot_data.plot(facecolor="none", linewidth=0.05, edgecolor="grey", ax=ax)
    ax.set_axis_off()
    ax.set(
        title="Change in average heatwave days experienced\n by infants in 2013-2022 relative to baseline"
    )
    ax.figure.savefig(Dirs.results.value / "map_nuts3_days_change_infants.png")
    plt.tight_layout()
    plt.show()

    days_experienced_baseline = nuts_exposures.sel(year=years_baseline).mean(dim="year")
    days_experienced_yr = nuts_exposures.sel(year=years_comparison).mean(dim="year")

    nuts_pct_change_person_days = (
        (
            (
                days_experienced_yr.sum(dim="age_band")
                - days_experienced_baseline.sum(dim="age_band")
            )
            / days_experienced_baseline.sum(dim="age_band")
        )
        .to_dataframe()
        .unstack()
    )
    print(nuts_pct_change_person_days.describe())

    days_experienced_baseline = nuts_exposures_norm.sel(year=years_baseline).mean(
        dim="year"
    )
    days_experienced_yr = nuts_exposures_norm.sel(year=years_comparison).mean(
        dim="year"
    )

    nuts_pct_change_days = (
        ((days_experienced_yr - days_experienced_baseline) / days_experienced_baseline)
        .to_dataframe()
        .unstack()
    )
    print(nuts_pct_change_days.T.describe())

    nuts_pct_change_days.T["LT_1"].replace([np.inf, -np.inf], np.nan).plot.hist()
    plt.show()
    nuts_pct_change_days.T["GT_65"].replace([np.inf, -np.inf], np.nan).plot.hist()
    plt.show()

    # Total and changes as exposure weighted days
    dem = vuln_demog.sum(dim=["latitude", "longitude"])
    exp = heatwave_exposure.sum(dim=["latitude", "longitude"])
    nrm = exp / dem
    print(exp.to_dataframe(name="heatwaves_days").unstack().T)

    nrm.to_dataframe(name="heatwaves_days").reset_index().to_csv(
        Dirs.results.value / "norm_days_heatwave_by_age_band.csv", index=False
    )

    print(nrm.to_dataframe(name="heatwaves_days").unstack().T)

    f, ax = plt.subplots(constrained_layout=True)
    data_plot = nrm.to_dataframe(name="heatwaves_days").unstack().T
    data_plot = data_plot.reset_index().drop(columns=["level_0"])
    data_plot.set_index("year", inplace=True)
    data_plot = data_plot.rename(columns={"LT_1": "Infants", "GT_65": "Over 65"})
    data_plot.plot(ax=ax)
    ylim = ax.get_ylim()
    ax.set(
        ylim=(0, math.ceil(ylim[1])),
        ylabel="Mean heatwave days",
        xlabel="Year",
    )
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(ls="--", color="lightgray")
    ax.legend(frameon=False)
    sns.despine()
    plt.savefig(Dirs.figures.value / "average_exposure_vulnerable_groups.png")
    plt.show()

    nrm.sel(year=years_baseline).mean(dim="year").to_dataframe(
        name="heatwaves_days"
    ).reset_index()
    nrm.sel(year=years_comparison).mean(dim="year").to_dataframe(
        name="heatwaves_days"
    ).reset_index()

    baseline = (
        nrm.sel(year=years_baseline)
        .mean(dim="year")
        .to_dataframe(name="heatwaves_days")
    )
    comparison = (
        nrm.sel(year=years_comparison)
        .mean(dim="year")
        .to_dataframe(name="heatwaves_days")
    )

    print((comparison - baseline) / baseline)

    comparison_max_analysis = nrm.sel(
        year=Variables.year_max_analysis.value
    ).to_dataframe(name="heatwaves_days")[["heatwaves_days"]]

    print((comparison_max_analysis - baseline) / baseline)

    plot_data = pd.concat(
        [
            (
                heatwave_days_eu.mean(dim=["latitude", "longitude"])
                .sel(year=slice(Variables.year_reference_start.value, None))
                .to_dataframe()
                .assign(age_band="European average")
                .reset_index()
            ),
            nrm.to_dataframe(name="heatwaves_days").reset_index(),
        ]
    )
    tbl_data = (
        plot_data.set_index(["age_band", "year"]).unstack().T.loc["heatwaves_days"]
    )
    print(tbl_data)

    f, ax = plt.subplots(constrained_layout=True)
    plot_data.age_band = plot_data.age_band.replace(
        {"LT_1": "Infants", "GT_65": "Over 65"}
    )
    sns.lineplot(data=plot_data, x="year", y="heatwaves_days", hue="age_band", ax=ax)
    # ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    # ax.set(title="heatwave days, average and by vulnerable group", ylabel="[days]")
    ylim = ax.get_ylim()
    ax.set(
        ylim=(0, math.ceil(ylim[1])),
        ylabel="Mean heatwave days per year",
        xlabel="Year",
    )
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(ls="--", color="lightgray")
    ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(Dirs.figures.value / "average_exposure_overall.png")
    plt.show()

    decade_change = (
        100
        * (
            heatwave_exposure.sel(year=years_comparison).mean(dim="year")
            - heatwave_exposure.sel(year=years_baseline).mean(dim="year")
        )
        / heatwave_exposure.sel(year=years_baseline).mean(dim="year")
    )
    # decade_change.to_netcdf(
    #     Dirs.results.value / "heatwave_exposure_map_percent_change_decades.nc"
    # )
    #
    # decade_change = xr.open_dataset(
    #     Dirs.results.value / "heatwave_exposure_map_percent_change_decades.nc"
    # )
    #
    # decade_change_table = decade_change.to_dataframe().dropna()
    # decade_change_table.reset_index().to_csv(
    #     Dirs.results.value / "heatwave_exposure_map_percent_change_decades.csv",
    #     index=False,
    # )
    # decade_change_table_sum = decade_change.sum(dim="age_band").to_dataframe().dropna()
    # decade_change_table_sum.reset_index().to_csv(
    #     Dirs.results.value
    #     / "heatwave_exposure_map_percent_change_decades_sum_age_bands.csv",
    #     index=False,
    # )
    # decade_change.sel(age_band="GT_65").max()
    # decade_change.sel(age_band="GT_65").quantile([0.95, 0.99])
    # heatwave_exposure.sel(year=2020).sum(dim=["latitude", "longitude"])
    # decade_change.mean(dim=["latitude", "longitude"])
    # decade_change.quantile(0.95, dim=["latitude", "longitude"])
    # decade_change_sum = (
    #     100
    #     * (
    #         heatwave_exposure.sel(year=years_comparison)
    #         .sum(dim="age_band")
    #         .mean(dim="year")
    #         - heatwave_exposure.sel(year=years_baseline)
    #         .sum(dim="age_band")
    #         .mean(dim="year")
    #     )
    #     / heatwave_exposure.sel(year=years_baseline)
    #     .sum(dim="age_band")
    #     .mean(dim="year")
    # )
    # decade_change_sum.mean(dim=["latitude", "longitude"])
    # decade_change_sum.quantile(0.95, dim=["latitude", "longitude"])

    def _plot_heatwave_exposure(plot_data, description_data):
        f = plt.figure()
        ax = plt.axes(projection=ccrs.epsg(3035), frameon=False)

        plot_data = plot_data.where(raster_nuts_grid > 0)

        plot_data.plot.pcolormesh(
            # norm=colors.CenteredNorm(),
            vmax=600,
            vmin=0,
            cmap="Reds",
            ax=ax,
            cbar_kwargs={"label": "% change"},
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(resolution="50m", color="grey", linewidth=0.5)
        ax.add_feature(cartopy.feature.BORDERS, color="grey", linewidth=0.5)
        ax.spines["geo"].set_visible(False)

        # ax.set(
        #     title=description_data
        #     + "\n"
        #     + f"{Variables.year_min_comparison.value}-{Variables.year_max_comparison.value} "
        #     f"compared to {Variables.year_reference_start.value}-{Variables.year_reference_end.value}"
        # )
        ax.set_title("")
        plt.tight_layout()
        filename = "_".join([x.lower() for x in description_data.split(" ")])
        f.savefig(Dirs.figures.value / f"{filename}.png", dpi=600)
        plt.show()

    # Usage
    _plot_heatwave_exposure(
        plot_data=decade_change.sum(dim="age_band"),
        description_data="Decadal change in heatwave exposure",
    )

    _plot_heatwave_exposure(
        plot_data=decade_change.sel(age_band="GT_65"),
        description_data=f"Decadal change in heatwave exposure of over 65",
    )

    _plot_heatwave_exposure(
        plot_data=decade_change.sel(age_band="LT_1"),
        description_data=f"Decadal change in heatwave exposure of infants",
    )


if __name__ == "__main__":
    pass

if __name__ == "__process__":

    # pre-analysis - run before calculating the main results
    calculate_heatwave_days()  # fast
    nuts2_id_as_int()  # fast
    re_grid_population()  # slow
    demographics_gpw_nuts_weighted()  # slow
    calculate_infants()  # medium
    calculate_heatwaves_exposure()  # fast
    calculate_exposures_nuts_2021()  # slow

    # plots and summaries
    calc_normed_exposure_eu_level()
    calc_normed_exposure_country_level()
    calc_normed_exposure_region_level()
    calc_normed_exposure_eea_level()
    plot_heatwave_exposure()
    main_results()
