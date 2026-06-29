# Stage the MINIMAL artifact bundle the demo image needs, into molvae/artifacts/.
# Run from anywhere; paths are resolved relative to this script.
# After this, build with Dockerfile.cpu (see DEPLOY.md "Path B — quick demo").
$ErrorActionPreference = "Stop"
$molvae = $PSScriptRoot
$art    = Join-Path $molvae "..\molvae_artifacts" | Resolve-Path
$stage  = Join-Path $molvae "artifacts"

New-Item -ItemType Directory -Force -Path (Join-Path $stage "processed")   | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stage "checkpoints") | Out-Null

# generator essentials
Copy-Item (Join-Path $art "processed\vocab.json")             (Join-Path $stage "processed\")   -Force
Copy-Item (Join-Path $art "processed\descriptor_stats.json")  (Join-Path $stage "processed\")   -Force
Copy-Item (Join-Path $art "processed\meta.json")              (Join-Path $stage "processed\")   -Force
Copy-Item (Join-Path $art "checkpoints\best.pt")              (Join-Path $stage "checkpoints\") -Force
# optional: electrolyte conductivity/coordination readout (small, keep it)
$elec = Join-Path $art "checkpoints\electrolyte_model.pt"
if (Test-Path $elec) { Copy-Item $elec (Join-Path $stage "checkpoints\") -Force }

# NOTE: molport.sqlite (922 MB) is intentionally NOT staged — the "in-catalog/novelty"
# badge is optional and the app runs fine without it. Add membership\ here if you want it.
$size = "{0:N0} MB" -f ((Get-ChildItem $stage -Recurse | Measure-Object Length -Sum).Sum / 1MB)
Write-Host "Staged minimal artifacts -> $stage  ($size)"
Write-Host "Next: az acr build ... -f Dockerfile.cpu  (see DEPLOY.md)"
