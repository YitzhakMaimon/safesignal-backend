# Self-restarting SSM port-forward tunnel: localhost:18006 -> output_screening
# EC2 instance (i-090245bb4449aeda9, "ImageAnalyser-OutputScreening" box)
# private port 8005. Same rationale as ssm-tunnel-rag-service-loop.ps1: the
# security group only opens 8005 to VPC-internal traffic (172.31.0.0/16, for
# langgraph_agent to reach it server-side) -- SSM port-forwarding connects via
# the SSM agent, not through that ingress rule, so it reaches the service
# regardless, without needing to loosen the security group at all.
#
# n8n's "Output Screening (Direct Microservice Call)" node depends on this
# tunnel being up (points at http://host.docker.internal:18006/v1/screen).

$env:Path += ";C:\Program Files\Amazon\SessionManagerPlugin\bin"
$awsExe = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
$logFile = "C:\Users\yitzhama\projects\final_project_safesignal\scripts\ssm-tunnel-output-screening.log"

function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

Write-Log "=== ssm-tunnel-output-screening-loop.ps1 started ==="

while ($true) {
    Write-Log "Starting SSM port-forward session..."
    try {
        & $awsExe ssm start-session `
            --region us-east-1 `
            --target i-090245bb4449aeda9 `
            --document-name AWS-StartPortForwardingSession `
            --parameters '{\"portNumber\":[\"8005\"],\"localPortNumber\":[\"18006\"]}' `
            2>&1 | Out-File -FilePath $logFile -Append -Encoding utf8
    } catch {
        Write-Log "Exception: $_"
    }
    Write-Log "Session ended (exit code $LASTEXITCODE) -- restarting in 3s"
    Start-Sleep -Seconds 3
}
