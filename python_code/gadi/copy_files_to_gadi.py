from icecream import ic
import pexpect
from my_config import Dirs, Variables
from python_code.secrets import gadi_password

for file in Dirs.data_era_daily_summaries.value.rglob("*.nc"):
    year = int(file.parts[-3])

    if "mean" in file.parts:
        continue

    if (
        Variables.year_reference_start.value
        <= year
        <= Variables.year_reference_end.value
    ):

        if year == 1991 and "min" in file.parts:
            # file = Path("/Users/ftar3919/Documents/lancet_countdown_europe/data/era5/era5_land_daily_summary/1991/max/era5-land_global_daily_tmax_199108.nc")

            destination = file.as_posix().replace(
                str(Dirs.main.value), str(Dirs.gadi.value)
            )

            ic(file)

            command = f"scp {str(file)} ft8695@gadi-dm.nci.org.au:{str(destination)}"
            child = pexpect.spawn(command)

            # Wait for the password prompt and send the password
            child.expect("password:")
            child.sendline(gadi_password)

            # Capture the output
            child.expect(pexpect.EOF)
            output = child.before.decode("utf-8")

            # Print the output
            print(output)
