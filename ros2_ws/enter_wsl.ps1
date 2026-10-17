# Open WSL already in this ros2_ws folder (Linux path under /mnt/...).
# PowerShell cannot source/build Linux ROS -- use this, then run bash commands.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

# E:\Desktop\Jessica\Camera\ros2_ws -> /mnt/e/Desktop/Jessica/Camera/ros2_ws
$drive = $Root.Substring(0, 1).ToLowerInvariant()
$unix = ($Root -replace '\\', '/') -replace '^[A-Za-z]:', "/mnt/$drive"

Write-Host ("Entering WSL at: {0}" -f $unix) -ForegroundColor Cyan
Write-Host 'Then run:  source /opt/ros/$ROS_DISTRO/setup.bash && ./build.sh'
Write-Host ''

$distros = @(wsl -l -q 2>$null | ForEach-Object { $_.ToString().Trim() } | Where-Object { $_ })
$prefer = @('Ubuntu', 'Ubuntu-24.04', 'Ubuntu-22.04')
$chosen = $null
foreach ($name in $prefer) {
    if ($distros -contains $name) {
        $chosen = $name
        break
    }
}

if ($chosen) {
    wsl -d $chosen --cd $unix
} else {
    # Default distro
    wsl --cd $unix
}
