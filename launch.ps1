# launch.ps1

# Retrieve the directory path where the path contains both the sample.py and launch.ps1 so that this PowerShell script can be invoked from any directory
$BASEFOLDER = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate the virtual environment
& "$BASEFOLDER\.venv\Scripts\Activate.ps1"

# Change to the base folder
Set-Location -Path $BASEFOLDER

# Run the Python script
python "$BASEFOLDER\Experiment_1\experiment_test_5.py"