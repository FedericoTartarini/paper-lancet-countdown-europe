from icecream import ic
import pexpect
from my_config import Dirs, Variables
from python_code.secrets import gadi_password, adp_password
from pprint import pprint


def move_files():
    for file in Dirs.data_era_daily_summaries.value.rglob("*.nc"):
        year = int(file.parts[-3])

        if "mean" in file.parts:
            continue

        if (
            Variables.year_reference_start.value
            <= year
            <= Variables.year_reference_end.value
        ):

            # if year == 1991 and "min" in file.parts:
            # file = Path("/Users/ftar3919/Documents/lancet_countdown_europe/data/era5/era5_land_daily_summary/1991/max/era5-land_global_daily_tmax_199108.nc")

            destination = file.as_posix().replace(
                str(Dirs.main.value), str(Dirs.adp.value)
            )

            ic(file.parts[-3:])

            command = f"scp {str(file)} frederico@adpgpu.staff.sydney.edu.au:{str(destination)}"

            child = pexpect.spawn(command)

            # Wait for the password prompt and send the password
            child.expect("password:")
            child.sendline(adp_password)

            # Capture the output
            child.expect(pexpect.EOF)
            output = child.before.decode("utf-8")

            # Print the output
            print(output)


def create_directories(directories):
    # Establish SSH connection
    ssh_command = "ssh frederico@adpgpu.staff.sydney.edu.au"
    child = pexpect.spawn(ssh_command)

    # Wait for the password prompt and send the password
    child.expect("password:")
    child.sendline(adp_password)

    # Wait for the shell prompt
    child.expect(r"\$")

    # Create directories
    for directory in directories:
        command = f"mkdir -p {directory}"
        child.sendline(command)
        child.expect(r"\$")

    # Close the SSH connection
    child.sendline("exit")
    child.expect(pexpect.EOF)


def create_list_directories(base_path_server):
    directories = []
    for year in range(
        Variables.year_reference_start.value, Variables.year_reference_end.value + 1
    ):
        directory = (
            base_path_server / "data" / "era5" / "era5_land_daily_summary" / str(year)
        )
        directories.append(directory)
        for var in ["min", "max", "mean"]:
            directories.append(directory / var)
    return directories


if __name__ == "__main__":
    # list_dirs_daily = create_list_directories(Dirs.adp.value)
    # pprint(list_dirs_daily)
    # create_directories(list_dirs_daily)
    move_files()
