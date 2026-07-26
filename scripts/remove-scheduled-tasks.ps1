# Cleanup: removes the SafeSignal-* Scheduled Tasks created earlier today.
# Abandoned as a persistence mechanism -- something on this machine
# (likely Amdocs' Zscaler endpoint agent) kills Task-Scheduler-spawned
# processes ~20-90s after they become network-active (open a port / SSM
# session), while the exact same scripts run indefinitely when launched
# ad-hoc from an interactive shell. Reverting to ad-hoc launches.
Get-ScheduledTask -TaskName "SafeSignal-*" -ErrorAction SilentlyContinue |
    ForEach-Object {
        Write-Host "Removing $($_.TaskName)"
        Unregister-ScheduledTask -TaskName $_.TaskName -Confirm:$false
    }
Write-Host "Done."
