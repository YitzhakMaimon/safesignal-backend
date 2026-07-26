# Self-restarting SSM port-forward tunnel: localhost:18002 -> rag_service
# EC2 instance (i-06f56419c7dab43ac) private port 8001. Same rationale as
# ssm-tunnel-loop.ps1 (langgraph_agent): this instance's public IP
# (34.203.210.242 as of 2026-07-23) gets blocked unpredictably by the
# corporate proxy (Zscaler) -- confirmed today with a consistent HTTP 403
# from both this host and n8n's own container. Routing through SSM instead
# of a direct connection to the public IP sidesteps that entirely.
#
# n8n's "Context Retrieval (RAG - LangChain + Vector DB)" node depends on
# this tunnel being up (points at http://host.docker.internal:18002/api/v1/rag-context).

$env:Path += ";C:\Program Files\Amazon\SessionManagerPlugin\bin"
$awsExe = "C:\Program Files\Amazon\AWSCLIV2\aws.exe"
$logFile = "C:\Users\yitzhama\projects\final_project_safesignal\scripts\ssm-tunnel-rag-service.log"

function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

Write-Log "=== ssm-tunnel-rag-service-loop.ps1 started ==="

while ($true) {
    Write-Log "Starting SSM port-forward session..."
    try {
        & $awsExe ssm start-session `
            --region us-east-1 `
            --target i-06f56419c7dab43ac `
            --document-name AWS-StartPortForwardingSession `
            --parameters '{\"portNumber\":[\"8001\"],\"localPortNumber\":[\"18002\"]}' `
            2>&1 | Out-File -FilePath $logFile -Append -Encoding utf8
    } catch {
        Write-Log "Exception: $_"
    }
    Write-Log "Session ended (exit code $LASTEXITCODE) -- restarting in 3s"
    Start-Sleep -Seconds 3
}
