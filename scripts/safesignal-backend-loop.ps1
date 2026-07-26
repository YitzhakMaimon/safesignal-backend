# Runs the real SafeSignal backend (safesignal.py) with the environment it
# needs. Registered as a Scheduled Task (see install-ssm-tunnels-as-services.ps1)
# rather than left as an ad-hoc `nohup python safesignal.py &` process --
# the latter doesn't survive logoff/reboot/crashes. Task Scheduler's own
# restart-on-failure settings (see the task registration) handle recovery,
# so this script doesn't need its own retry loop like the SSM tunnel
# scripts do.
#
# PYTHONIOENCODING=utf-8 is required -- without it, Hebrew text handling
# breaks (documented issue in this project). Model loading (HeBERT + RAG
# index) takes 1-2 minutes on every fresh start -- callers (tunnels, the
# Telegram bridge) will see connection errors during that window, which is
# expected, not a fault.

$ProjectRoot = "C:\Users\yitzhama\projects\final_project_safesignal"
$LogFile = "$ProjectRoot\scripts\safesignal-backend-service.log"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $ProjectRoot

& "$ProjectRoot\.venv\Scripts\python.exe" "$ProjectRoot\safesignal.py" *>> $LogFile
