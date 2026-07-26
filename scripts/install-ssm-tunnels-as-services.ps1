# Registers the real backend (safesignal.py), the three ssm-tunnel-*-loop.ps1
# scripts, and the SSH reverse tunnel (EC2 -> this machine, used by
# telegram_bridge.py and ALERT_WEBHOOK_URL/REVIEW_QUEUE_URL) as persistent
# Windows Scheduled Tasks, so they survive logoff/reboot instead of dying
# silently whenever this machine restarts or the terminal that launched
# them closes (2026-07-26: all four tunnels were found dead at various
# points, one for 2+ days, with nobody noticing until a live-pipeline test
# needed them; safesignal.py itself was still just an ad-hoc `nohup`
# process with nothing to bring it back after a reboot).
#
# Uses Task Scheduler (built into Windows) rather than NSSM: NSSM requires
# downloading a binary from nssm.cc, which this network blocks outright
# (same Zscaler policy that blocks api.telegram.org elsewhere in this
# project -- confirmed via a hung `curl` to nssm.cc before writing this).
#
# Must be run from an elevated (Administrator) PowerShell -- registering a
# scheduled task in this environment fails with "Access is denied" without
# it (tested), likely a Group Policy restriction on this managed machine.

$ErrorActionPreference = "Stop"
$ProjectRoot = "C:\Users\yitzhama\projects\final_project_safesignal"
$ScriptsDir = Join-Path $ProjectRoot "scripts"
$CurrentUser = "$env:USERDOMAIN\$env:USERNAME"

$tasks = @(
    @{ Name = "SafeSignal-Backend";              Script = "safesignal-backend-loop.ps1" },
    @{ Name = "SafeSignal-SSM-Tunnel-Agent";     Script = "ssm-tunnel-loop.ps1" },
    @{ Name = "SafeSignal-SSM-Tunnel-RAG";       Script = "ssm-tunnel-rag-service-loop.ps1" },
    @{ Name = "SafeSignal-SSM-Tunnel-Screening"; Script = "ssm-tunnel-output-screening-loop.ps1" },
    @{ Name = "SafeSignal-SSH-Reverse-Tunnel";   Script = "ssh-reverse-tunnel-loop.ps1" }
)

foreach ($t in $tasks) {
    $name = $t.Name
    $scriptPath = Join-Path $ScriptsDir $t.Script
    if (-not (Test-Path $scriptPath)) {
        throw "Script not found: $scriptPath"
    }

    $existing = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($existing) {
        $info = $existing | Get-ScheduledTaskInfo
        # Leave already-running tasks alone -- re-registering forces a
        # stop/start cycle, and for the EC2-facing tunnels that means a new
        # SSM/SSH session; this instance only tolerates ~2 concurrent
        # sessions before things start failing (2026-07-26 incident), so
        # touching tunnels that are already fine risks breaking them for
        # no reason. Only re-register if it's missing or not currently up.
        if ($info.LastTaskResult -eq 267009) {
            Write-Host "$name already running -- leaving it alone."
            continue
        }
        Write-Host "$name exists but isn't running (result $($info.LastTaskResult)) -- removing before re-creating."
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
    }

    Write-Host "Registering task $name -> $scriptPath"

    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`"" `
        -WorkingDirectory $ProjectRoot

    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser

    # No execution time limit (Task Scheduler kills tasks after 3 days by
    # default -- fatal for a script meant to run forever), aggressive
    # restart-on-failure, and don't stack a second copy if one somehow
    # gets triggered while the first is still running.
    $settings = New-ScheduledTaskSettingsSet `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 999 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -MultipleInstances IgnoreNew `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries

    $principal = New-ScheduledTaskPrincipal -UserId $CurrentUser -LogonType Interactive -RunLevel Limited

    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal -Force | Out-Null
}

# --- Start all three immediately (don't wait for next logon) ---
foreach ($t in $tasks) {
    Start-ScheduledTask -TaskName $t.Name
}

Start-Sleep -Seconds 8

Write-Host "`n=== Task status ==="
Get-ScheduledTask -TaskName "SafeSignal-*" |
    Get-ScheduledTaskInfo |
    Select-Object TaskName, LastRunTime, LastTaskResult |
    Format-Table -AutoSize

Write-Host "`n=== Listening ports (18001 / 18002 / 18006) ==="
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -in 18001, 18002, 18006 } |
    Format-Table LocalAddress, LocalPort, OwningProcess -AutoSize
