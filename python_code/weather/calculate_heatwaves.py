"""
# Calculate heatwave occurrences

## Heatwave definition

Heatwaves are now defined as:

Tmin > 95percentile AND Tmax > 95percentile

For more than 2 consecutive days (i.e. total of 3 or more days).

This replaces the definition of only Tmin > 99percentile for more than 3 consecutive days (total of 4 or more days).

This is what is requested from the Lancet. To be honest it's not clear whether this produces a substantially 'better'
indicator since all heatwave indicators are arbitrary in absence of covariate data (i.e. impact data). Furthermore, we
know that the health impacts are mediated by many other things, so in any case we are truly interested just in the
trends i.e. demonstrating that there is a) more heatwaves and b) more exposure to heatwaves - this can be followed by
local studies but (as always) the point is to present a general risk factor trend.

> NOTE: considered just adding the newest year each time instead of re-calculating the whole thing. HOWEVER in
reality, the input data is still changing year to year, so far have needed to re-calculate anyway (e.g. change in
resolution, change from ERAI to ERA5, in the future probably use ERA5-Land, etc). Although it seems like a cool idea
to have a reproducible method where each year you just add one thing, in practice its better to have one 'frozen'
output corresponding to each publication, so that it's easy to go back later to find data corresponding to specific
results. Additionally, generating one file per year means you have a folder full of files that are harder to share,
and the outputs are in the end pretty small (<50MB in Float32)}."""

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from my_config import Variables, Directories

xr.set_options(keep_attrs=True)


def heatwaves_counts_multi_threshold(datasets_year, thresholds, days_threshold=2):
    """
    Accepts data as a (time, lat, lon) shaped boolean array.
    Iterates through the array in the time dimension comparing the current
    time slice to the previous one. For each cell, determines whether the
    cell is True (i.e. is over the heatwave thresholds) and whether this is
    the start, continuation, or end of a sequence of heatwave conditions.
    Accumulates the number of occurances and counts the total occurances.
    """
    datasets_year = [d.fillna(-9999) for d in datasets_year]
    # Init whole array to True
    threshold_exceeded = datasets_year[0] > thresholds[0]
    # For each threshold array, 'and' them together
    for dataset_year, thresh in zip(datasets_year[1:], thresholds[1:]):
        # for each (data, threshold) pair, add constraint the threshold excedance array
        threshold_exceeded = np.logical_and(threshold_exceeded, dataset_year > thresh)
    # Keep only the numpy array
    threshold_exceeded = threshold_exceeded.values

    # Init arrays, pre allocate to (hopefully) improve performance.
    out_shape: tuple = threshold_exceeded.shape[1:]

    last_slice: bool[:, :] = threshold_exceeded[0, :, :]
    curr_slice: bool[:, :] = threshold_exceeded[0, :, :]
    hw_ends: bool[:, :] = np.zeros(out_shape, dtype=bool)
    mask: bool[:, :] = np.zeros(out_shape, dtype=bool)

    # Init as int32 - value will never be > 365
    accumulator = np.zeros(threshold_exceeded.shape[1:], dtype=np.int32)
    counter = np.zeros(threshold_exceeded.shape[1:], dtype=np.int32)

    # Calculate the run length of the exceedances and count only the ones
    # over the length threshold
    for i in range(1, threshold_exceeded.shape[0]):
        last_slice = threshold_exceeded[i - 1, :, :]
        curr_slice = threshold_exceeded[i, :, :]

        # Add to the sequence length counter at all positions
        # above threshold at prev time step using boolean indexing
        accumulator[last_slice] += 1

        # End of sequence is where prev is true and current is false
        # True where prev and not current
        # Use pre-allicocated arrays for results
        np.logical_and(last_slice, np.logical_not(curr_slice), out=hw_ends)
        np.logical_and(hw_ends, (accumulator > days_threshold), out=mask)

        # Add 1 where the sequences are ending and are > 3
        counter[mask] += 1
        # Reset the accumulator where current slice is empty
        accumulator[np.logical_not(curr_slice)] = 0

    # Finally, 'close' the heatwaves that are ongoing at the end of the year
    # End of sequence is where last value of iteration is true and accumulator is over given length
    np.logical_and(curr_slice, (accumulator > days_threshold), out=mask)

    # Add the length of the accumulator where the sequences are ending and are > 3
    counter[mask] += 1

    # Convert np array to xr DataArray
    counter = xr.DataArray(
        counter,
        coords=[
            datasets_year[0].latitude.values,
            datasets_year[0].longitude.values,
        ],
        dims=["latitude", "longitude"],
        name="heatwave_count",
    )

    return counter


def heatwaves_days_multi_threshold(datasets_year, thresholds, days_threshold: int = 2):
    """
    Accepts data as a (time, lat, lon) shaped boolean array.
    Iterates through the array in the time dimension comparing the current
    time slice to the previous one. For each cell, determines whether the
    cell is True (i.e. is over the heatwave thresholds) and whether this is
    the start, continuation, or end of a sequence of heatwave conditions.
    Accumulates the number of days and counts the total lengths.
    """
    datasets_year = [d.fillna(-9999) for d in datasets_year]
    # Init array
    threshold_exceeded = datasets_year[0] > thresholds[0]
    # For each threshold array, 'and' them together
    for _data_year, _thresh in zip(datasets_year[1:], thresholds[1:]):
        # for each (data, threshold) pair, add constraint the threshold excedance array
        threshold_exceeded = np.logical_and(threshold_exceeded, _data_year > _thresh)

    # Keep only the numpy array
    # threshold_exceeded = threshold_exceeded.values

    # pre allocate arrays
    out_shape: tuple = threshold_exceeded.shape[1:]

    last_slice: bool[:, :] = threshold_exceeded[0, :, :]
    curr_slice: bool[:, :] = threshold_exceeded[0, :, :]
    hw_ends: bool[:, :] = np.zeros(out_shape, dtype=bool)
    mask: bool[:, :] = np.zeros(out_shape, dtype=bool)

    # Init as int32 - value will never be > 365
    accumulator = np.zeros(out_shape, dtype=np.int32)
    days = np.zeros(out_shape, dtype=np.int32)

    for i in range(1, threshold_exceeded.shape[0]):
        last_slice = threshold_exceeded[i - 1, :, :]
        curr_slice = threshold_exceeded[i, :, :]

        # Add to the sequence length counter at all positions
        # above threshold at prev time step using boolean indexing
        accumulator[last_slice] += 1

        # End of sequence is where prev is true and current is false
        # True where prev and not current
        # Use pre-allocated arrays for results
        np.logical_and(last_slice, np.logical_not(curr_slice), out=hw_ends)
        np.logical_and(hw_ends, (accumulator > days_threshold), out=mask)

        # Add the length of the accumulator where the sequences are ending and are > 3
        days[mask] += accumulator[mask]
        # Reset the accumulator where current slice is empty
        accumulator[np.logical_not(curr_slice)] = 0

    # Finally, 'close' the heatwaves that are ongoing at the end of the year
    # End of sequence is where last value of iteration is true and accumulator is over given length
    np.logical_and(curr_slice, (accumulator > days_threshold), out=mask)

    # Add the length of the accumulator where the sequences are ending and are > 3
    days[mask] += accumulator[mask]

    # Convert np array to xr DataArray
    days = xr.DataArray(
        days,
        coords=[
            datasets_year[0].latitude.values,
            datasets_year[0].longitude.values,
        ],
        dims=["latitude", "longitude"],
        name="heatwaves_days",
    )

    return days


def ds_for_year(year):
    temperature_files = []

    for p in Directories.data_era_daily_summaries.value.glob("*"):
        if str(year) in p.parts:
            temperature_files += [p for p in p.rglob("*.nc")]

    temperature_files.sort()
    ds = xr.open_mfdataset(temperature_files)
    ds = ds.drop("time_bnds")
    ds = ds.transpose("time", "latitude", "longitude")
    return ds


def apply_func_for_file(func, year, t_thresholds, t_var_names, days_threshold=2):
    ds = ds_for_year(year)

    datasets_year = [ds[name] for name in t_var_names]
    result = func(datasets_year, t_thresholds, days_threshold)

    # Add a year dimension matching the input file
    result = result.expand_dims(dim={"year": [year]})
    return year, result


def apply_func_and_save(
    func,
    year,
    output_folder,
    t_thresholds,
    t_var_names=["tmin", "tmax"],
    days_threshold=2,
    overwrite=False,
    filename_pattern="indicator_{year}.nc",
):
    output_file = output_folder / filename_pattern.format(year=year)
    if output_file.exists() is False and overwrite is False:
        year, result = apply_func_for_file(
            func,
            year,
            t_thresholds,
            t_var_names=t_var_names,
            days_threshold=days_threshold,
        )
        result.to_netcdf(output_file)
        return f"Created {output_file}"
    else:
        return f"Skipped {output_file}, already exists"


def apply_func_for_month(func, ds, month, t_thresholds, t_var_names, days_threshold=2):
    monthly_ds = ds.sel(time=ds["time"].dt.month == month)
    datasets_month = [monthly_ds[name] for name in t_var_names]

    result = func(datasets_month, t_thresholds, days_threshold)
    return result  # Returning only the monthly result


def apply_func_and_save_yearly(
    func,
    year,
    output_folder,
    t_thresholds,
    t_var_names,
    days_threshold=2,
    overwrite=False,
    filename_pattern="indicator_{year}.nc",
):
    ds = ds_for_year(year)
    yearly_results = []

    for month in range(1, 13):
        monthly_result = apply_func_for_month(
            func, ds, month, t_thresholds, t_var_names, days_threshold
        )
        yearly_results.append(monthly_result)

    # Combine all monthly results into one dataset
    combined_result = xr.concat(yearly_results, pd.Index(range(1, 13), name="month"))
    combined_result = combined_result.assign_coords({"year": year})

    # Save the combined yearly file
    output_file = output_folder / filename_pattern.format(year=year)
    if not output_file.exists() or overwrite:
        combined_result.to_netcdf(output_file)
        return f"Created {output_file}"
    else:
        return f"Skipped {output_file}, already exists"


if __name__ == "__main__":
    quantiles_files = list(Directories.data_era_quantiles.value.rglob("*.nc"))
    t_quantiles = xr.open_mfdataset(quantiles_files, compat="override")

    quantile = Variables.quantiles.value[0]

    t_min_threshold = t_quantiles.tmin.sel(
        quantile=quantile, drop=True, tolerance=0.001, method="nearest"
    )
    t_max_threshold = t_quantiles.tmax.sel(
        quantile=quantile, drop=True, tolerance=0.001, method="nearest"
    )
    t_thresholds = [t_min_threshold, t_max_threshold]

    # # the data calculated for monthly heatwaves is not used in the current analysis
    # res = Parallel(n_jobs=6, verbose=3)(
    #     delayed(apply_func_and_save_yearly)(
    #         heatwaves_days_multi_threshold,
    #         year,
    #         dir_results_heatwaves_monthly,
    #         t_thresholds,
    #         ["t_min", "t_max"],
    #     )
    #     for year, _ in temperature_files
    # )

    res = Parallel(n_jobs=6, verbose=3)(
        delayed(apply_func_and_save)(
            heatwaves_days_multi_threshold,
            year,
            Directories.data_era_heatwaves_days.value,
            t_thresholds,
            ["tmin", "tmax"],
        )
        for year in range(
            Variables.year_reference_start.value, Variables.year_report.value
        )
    )

    res = Parallel(n_jobs=5, verbose=2)(
        delayed(apply_func_and_save)(
            heatwaves_counts_multi_threshold,
            year,
            Directories.data_era_heatwaves_counts.value,
            t_thresholds,
            ["tmin", "tmax"],
        )
        for year in range(
            Variables.year_reference_start.value, Variables.year_report.value
        )
    )
