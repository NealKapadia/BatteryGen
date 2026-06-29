# Start the on-demand MolForge GPU VM for a session. Run gpu_stop.ps1 when done!
# Billing (~$0.50/hr for a T4) only accrues while the VM is running.
param([string]$rg = "molvae-rg", [string]$vm = "molforge-gpu")
Write-Host "Starting GPU VM '$vm' ..."
az vm start -g $rg -n $vm
$ip = az vm show -d -g $rg -n $vm --query publicIps -o tsv
Write-Host ""
Write-Host "GPU VM is up at IP: $ip"
Write-Host "The app (docker --restart unless-stopped) comes back automatically in ~1-2 min."
Write-Host "Open:  http://$ip:8000   (or your https domain if Caddy is configured)"
Write-Host ""
Write-Host "*** When finished, run:  .\gpu_stop.ps1   to stop billing. ***"
