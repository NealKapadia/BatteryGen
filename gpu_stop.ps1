# Stop (deallocate) the MolForge GPU VM so compute billing stops. Disk persists.
param([string]$rg = "molvae-rg", [string]$vm = "molforge-gpu")
Write-Host "Deallocating GPU VM '$vm' ..."
az vm deallocate -g $rg -n $vm
Write-Host "Done. No further compute charges (the OS disk is kept, ~a few \$/mo)."
