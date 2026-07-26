# Registers ssh-reverse-tunnel-loop.ps1 as a Windows Scheduled Task that
# starts automatically at logon and restarts itself if it ever stops -- same
# pattern as install-ssm-tunnel-service.ps1. Run once to install; safe to
# re-run (it just replaces the existing task definition).

$taskName = "SafeSignal-SSH-ReverseTunnel-LangGraphAgent"
$scriptPath = "C:\Users\yitzhama\projects\final_project_safesignal\scripts\ssh-reverse-tunnel-loop.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$scriptPath`""

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force

Write-Output "Registered scheduled task '$taskName'. Starting it now..."
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 3
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State
