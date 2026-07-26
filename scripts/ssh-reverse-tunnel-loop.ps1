# Self-restarting SSH reverse tunnel: EC2 langgraph_agent instance's own
# port 8000 -> back to this machine's localhost:8000 (safesignal.py).
#
# Why this exists: agent_graph.py's output_screening_node / alert / review-queue
# calls need to reach safesignal.py (which only runs on this Windows machine,
# not on EC2) for ALERT_WEBHOOK_URL and REVIEW_QUEUE_URL. langgraph_agent runs
# in a bridge-networked container on EC2, so it can't just call "localhost" --
# it calls http://host.docker.internal:8000/..., which Docker maps to the EC2
# host's own network via --add-host host.docker.internal:host-gateway (set on
# the container). This script is what makes the EC2 HOST's port 8000 actually
# go anywhere: an SSH -R reverse forward, bound to 0.0.0.0 on the EC2 side
# (requires GatewayPorts yes in that instance's /etc/ssh/sshd_config -- already
# set -- so the docker bridge gateway can reach it, not just the EC2 host's own
# loopback) tunneled back to this machine's localhost:8000.
#
# Direct SSH (port 22) is refused on this network (same corporate/Zscoler
# issue documented in ssm-tunnel-loop.ps1) -- so this routes plain SSH through
# an SSM session instead (AWS-StartSSHSession), which uses the same HTTPS-based
# transport that already works reliably here. Auth is still the real SSH key
# (safesignal-shared-key.pem); SSM only carries the bytes.
#
# Runs in an infinite loop so it comes back up whenever the session ends for
# any reason (SSM idle timeout, network blip, instance reboot), same pattern
# as ssm-tunnel-loop.ps1.

$env:Path += ";C:\Program Files\Amazon\SessionManagerPlugin\bin"
$sshExe = "C:\Program Files\Git\usr\bin\ssh.exe"
$sshConfig = "C:\Users\yitzhama\projects\final_project_safesignal\scripts\ssh-reverse-tunnel-config"
$logFile = "C:\Users\yitzhama\projects\final_project_safesignal\scripts\ssh-reverse-tunnel.log"

function Write-Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts $msg" | Out-File -FilePath $logFile -Append -Encoding utf8
}

Write-Log "=== ssh-reverse-tunnel-loop.ps1 started ==="

while ($true) {
    Write-Log "Starting SSH reverse-tunnel session..."
    try {
        & $sshExe -F $sshConfig -o ExitOnForwardFailure=yes -N -R 0.0.0.0:8000:localhost:8000 langgraph-ec2 `
            2>&1 | Out-File -FilePath $logFile -Append -Encoding utf8
    } catch {
        Write-Log "Exception: $_"
    }
    Write-Log "Session ended (exit code $LASTEXITCODE) -- restarting in 3s"
    Start-Sleep -Seconds 3
}
