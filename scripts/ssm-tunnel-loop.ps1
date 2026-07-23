# Self-restarting SSM port-forward tunnel: localhost:18001 -> langgraph_agent
# EC2 instance (i-0e10488a3bbbefef9) private port 8001, used because this
# instance's own public IP gets blocked by the corporate proxy (Zscaler)
# unpredictably -- the SSM tunnel routes through AWS's own API instead of a
# direct connection to the instance's public IP, sidestepping that entirely.
#
# Runs `aws ssm start-session` in an infinite loop so it comes back up
# whenever the session ends for any reason: AWS's own SSM idle timeout
# (~20 min with no traffic), a network blip, an instance reboot, etc. This
# is the Windows equivalent of a systemd service's "Restart=always" (this
# machine has no systemd) -- see scripts/install-ssm-tunnel-service.ps1 for
# the Scheduled Task that starts this script automatically at logon.
#
# n8n's "Invoke Decision Agent (Backend)" node depends on this tunnel being
# up (points at http://host.docker.internal:18001/api/v1/decision-agent).
# If n8n calls start failing with connection-refused on that node, check
# scripts/ssm-tunnel.log first before assuming the EC2 instance itself is
# down.

$env:Path += ";C:\Program Files\Amazon\SessionManagerPlugin\bin"
$awsExe = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
$logFile = "C:\Users\yitzhama\projects\final_project_safesignal\scripts\ssm-tunnel.log"

function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

Write-Log "=== ssm-tunnel-loop.ps1 started ==="

while ($true) {
    Write-Log "Starting SSM port-forward session..."
    try {
        & $awsExe ssm start-session `
            --region us-east-1 `
            --target i-0e10488a3bbbefef9 `
            --document-name AWS-StartPortForwardingSession `
            --parameters '{\"portNumber\":[\"8001\"],\"localPortNumber\":[\"18001\"]}' `
            2>&1 | Out-File -FilePath $logFile -Append -Encoding utf8
    } catch {
        Write-Log "Exception: $_"
    }
    Write-Log "Session ended (exit code $LASTEXITCODE) -- restarting in 3s"
    Start-Sleep -Seconds 3
}
