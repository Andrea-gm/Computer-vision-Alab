$ErrorActionPreference = "Continue"
$log = "E:\Desktop\Jessica\Camera\ros2_ws\_wsl_install.log"
function Log([string]$m) {
    $line = "{0} {1}" -f (Get-Date -Format "o"), $m
    Add-Content -Path $log -Value $line
    Write-Output $line
}

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Log "elevated=$isAdmin"
if (-not $isAdmin) {
    Log "not elevated; abort"
    exit 1
}

foreach ($feature in @("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")) {
    Log "dism enable $feature"
    & dism.exe /online /enable-feature /featurename:$feature /all /norestart
    Log "dism $feature exit $LASTEXITCODE"
}

Log "wsl --install Ubuntu-24.04 --no-launch"
& wsl.exe --install --distribution Ubuntu-24.04 --no-launch
Log "wsl --install exit $LASTEXITCODE"
Log "finished"
exit 0
