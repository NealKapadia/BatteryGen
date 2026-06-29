# Deploying the molvae designer to the web on Azure

End-to-end: build the container, run it on an Azure **GPU** VM (or CPU), point a
**domain** at it, and get **automatic HTTPS**. Two paths:

| Path | When | Cost (rough) |
|---|---|---|
| **A. GPU VM** (recommended) | full speed, electrolyte model, fast latent search | ~$0.5–1.5/hr (NC4as_T4) |
| **B. CPU container** (Azure Container Apps) | cheap, fine for single-molecule generation | ~$0.05–0.2/hr, scale-to-zero |

The app needs only these **artifacts** (not the 2 GB shards): `vocab.json`,
`descriptor_stats.json`, `meta.json`, `checkpoints/best.pt` (or `latest.pt`/`user.pt`),
`membership/` (novelty badge — optional), and `electrolyte_model.pt`. Bundle them as
`./artifacts/` next to the Dockerfile.

```
artifacts/
  processed/{vocab.json, descriptor_stats.json, meta.json}
  checkpoints/best.pt
  checkpoints/electrolyte_model.pt
  membership/{molport.sqlite, molport.bloom}   # optional (novelty / in-catalog)
```

---

## ⭐ RECOMMENDED for a first public demo — CPU on Azure Container Apps (cheap, HTTPS, scale-to-zero)

This is the fastest way to get a shareable `https://…` URL for the **generative designer**
(prompt/sliders → molecule + 3D + electrolyte readout + LLM explanation). No GPU, no domain,
no local Docker needed. Generation is a few seconds on CPU. ~a few $/month at light use
(scales to zero when idle). The CE/RAG tools stay CLI for now (Phase 2).

```powershell
cd "C:\Users\nkapa\Downloads\Molport_Full_Database\All Stock Compounds\molvae"

# 0) prereqs (once)
winget install Microsoft.AzureCLI ; az login
az account set --subscription "<your-subscription>"

# 1) stage the minimal artifacts the image bakes in (~0.5 GB: best.pt + vocab + stats)
powershell -ExecutionPolicy Bypass -File .\deploy_prepare.ps1

# 2) resource group + container registry, then build IN THE CLOUD from Dockerfile.cpu
az group create -n molvae-rg -l eastus
az acr create -g molvae-rg -n molvaeacr --sku Basic
az acr build -r molvaeacr -t molvae:cpu -f Dockerfile.cpu .

# 3) Container Apps env + the app (keys as secrets, scale-to-zero)
az containerapp env create -g molvae-rg -n molvae-env -l eastus
az containerapp create -g molvae-rg -n molvae-app --environment molvae-env `
  --image molvaeacr.azurecr.io/molvae:cpu --registry-server molvaeacr.azurecr.io `
  --target-port 8000 --ingress external --cpu 2 --memory 4Gi `
  --min-replicas 0 --max-replicas 2 `
  --secrets foundrykey=$env:FOUNDRY_API_KEY azurekey=$env:AZURE_OPENAI_KEY `
  --env-vars MOLVAE_DEVICE=cpu `
             FOUNDRY_API_KEY=secretref:foundrykey AZURE_OPENAI_KEY=secretref:azurekey

# 4) get the public URL
az containerapp show -g molvae-rg -n molvae-app --query properties.configuration.ingress.fqdn -o tsv
```

Open `https://<that-fqdn>` and share it. First request after idle has a ~20–40 s cold start
(model load); set `--min-replicas 1` (~$15–30/mo) to keep it warm for a live demo.
Update later: re-run steps 1–2 then `az containerapp update -g molvae-rg -n molvae-app --image molvaeacr.azurecr.io/molvae:cpu`.

> Memory: `best.pt` is ~0.5 GB and torch-CPU needs headroom — use `--memory 4Gi` (8Gi if you
> later bake the 922 MB Molport index for the in-catalog badge).

---

## ⚡ On-demand GPU for fast sessions (best when you use it ~1–2×/month)

Generation is minutes on CPU, ~10–20 s on a T4 GPU. For occasional use, run a GPU VM only
during a session (~$0.50/hr while on, **$0 when deallocated**). One-time setup, then flip on/off
with the included scripts.

**One-time setup**
```powershell
cd "C:\Users\nkapa\Downloads\Molport_Full_Database\All Stock Compounds\molvae"
$ACR="molvaenk42707"
powershell -ExecutionPolicy Bypass -File .\deploy_prepare.ps1            # stage artifacts
az acr build -r $ACR -t molvae:gpu -f Dockerfile.gpu .                   # build GPU image
# create the GPU VM + open port 8000 (see Path A1/A2 below for driver+docker+nvidia toolkit)
az vm create -g molvae-rg -n molforge-gpu --image Ubuntu2204 --size Standard_NC4as_T4_v3 `
  --admin-username azureuser --generate-ssh-keys --public-ip-sku Standard --os-disk-size-gb 64
az vm open-port -g molvae-rg -n molforge-gpu --port 8000 --priority 900
# SSH in, install NVIDIA driver + Docker + nvidia-container-toolkit (Path A2), then run the app
# so it auto-restarts on every VM start:
#   az acr login -n molvaenk42707
#   docker run -d --restart unless-stopped --gpus all -p 8000:8000 `
#     -e FOUNDRY_API_KEY=... -e AZURE_OPENAI_KEY=... molvaenk42707.azurecr.io/molvae:gpu
```

**Every session**
```powershell
.\gpu_start.ps1     # az vm start + prints the IP; app is up in ~1-2 min
# ... use http://<ip>:8000 ...
.\gpu_stop.ps1      # az vm deallocate — stops compute billing
```
(In-person demo on your own machine? You have an RTX 3060 — just run
`MOLVAE_DEVICE=cuda uvicorn server:app` locally for instant, free generation.)

## 📦 Python library (let others run the model on their own CPU/GPU)

`molforge.py` is a standalone API over the pretrained generator — no web app, no Azure keys.
Share the `molvae/` code + the artifacts bundle (`processed/` + `checkpoints/best.pt`; host
`best.pt` on e.g. Hugging Face since it's ~0.5 GB).
```python
from molforge import MolForge
mf = MolForge(device="cpu")                       # or "cuda"
mf.generate(10)                                   # 10 valid, novel SMILES
mf.generate(5, spec={"MolWt":300,"QED":0.8})      # property-targeted
mf.encode("CCO"); mf.decode(z)                    # latent round-trip
mf.predict_properties("OCCN(CCO)CCO")             # the VAE's property head
```
CLI: `python molforge.py --n 20 --device cuda --out molecules.csv`

---

## 0. Prerequisites (once, on your laptop)

```powershell
winget install Microsoft.AzureCLI        # or https://aka.ms/installazurecli
az login
az account set --subscription "<your-subscription>"
# Docker Desktop installed and running (for building the image)
```

---

## Path A — GPU VM (recommended)

### A1. Create a GPU VM

T4 GPUs (NCas_T4_v3 series) are the cheapest and plenty for this model. Pick a region
that has them (e.g. `eastus`, `westus2`).

```powershell
az group create -n molvae-rg -l eastus

az vm create -g molvae-rg -n molvae-gpu `
  --image Ubuntu2204 `
  --size Standard_NC4as_T4_v3 `
  --admin-username azureuser `
  --generate-ssh-keys `
  --public-ip-sku Standard `
  --os-disk-size-gb 128

# open web ports (80/443) and SSH (22)
az vm open-port -g molvae-rg -n molvae-gpu --port 80  --priority 900
az vm open-port -g molvae-rg -n molvae-gpu --port 443 --priority 901
az vm show -d -g molvae-rg -n molvae-gpu --query publicIps -o tsv   # note the IP
```

> If `az vm create` says the size isn't available, try another region or
> `Standard_NC6s_v3` (V100) / `Standard_NC8as_T4_v3`. Check quota:
> `az vm list-usage -l eastus -o table | findstr NC`. New accounts often need a
> quota increase for GPU cores — request it in Portal → *Quotas*.

### A2. Install GPU driver + Docker + NVIDIA container toolkit (on the VM)

```bash
ssh azureuser@<PUBLIC_IP>

# NVIDIA driver
sudo apt-get update && sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot                       # reconnect after ~30s, then verify:
nvidia-smi

# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# NVIDIA Container Toolkit (lets Docker see the GPU)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # should print the GPU
```

### A3. Ship the code + artifacts to the VM

From your laptop (the `molvae/` folder and the `artifacts/` bundle you assembled):

```powershell
# code
scp -r "C:\Users\nkapa\Downloads\Molport_Full_Database\All Stock Compounds\molvae" azureuser@<IP>:~/molvae
# artifacts (model + vocab + stats + membership + electrolyte model)
scp -r .\artifacts azureuser@<IP>:~/molvae/artifacts
```
(For big artifacts, `azcopy` or an Azure Storage account is faster than scp.)

### A4. Configure secrets + domain, then launch

On the VM:
```bash
cd ~/molvae
cp .env.example .env && nano .env          # paste your FOUNDRY_*/AZURE_OPENAI_* keys
nano Caddyfile                              # replace electrolyte.example.com with your domain
docker compose up -d --build                # builds image, starts app + Caddy(HTTPS)
docker compose logs -f molvae               # watch it load the model
curl -s localhost:8000/health               # {"ok":true,"device":"cuda","electrolyte":true}
```

That's it for the server. Now point a domain at `<PUBLIC_IP>` (next section) and Caddy
auto-issues HTTPS within ~30s of DNS resolving.

---

## Buying a domain + DNS

**Option 1 — buy through Azure (simplest, auto-integrates):**
```powershell
# App Service Domains (managed registrar). ~ $12/yr for .com
az appservice domain create -g molvae-rg --hostname electrolyte-ai.com `
  --contact-info @contact.json --accept-terms
```
Then add an **A record** → your VM's public IP (Portal → that domain's DNS zone →
*Record sets* → add `@` and `www` A-records pointing to `<PUBLIC_IP>`).

**Option 2 — any registrar (Namecheap, Cloudflare, Google Domains):** buy the domain,
then in the registrar's DNS panel add:
```
Type  Host   Value            TTL
A     @      <PUBLIC_IP>      3600
A     www    <PUBLIC_IP>      3600
```

**Make the VM IP static** (so it doesn't change on reboot):
```powershell
az network public-ip update -g molvae-rg -n molvae-guPublicIP --allocation-method Static
```
(Find the exact public-ip name with `az network public-ip list -g molvae-rg -o table`.)

**HTTPS:** nothing else to do — Caddy (in docker-compose) provisions and renews a free
Let's Encrypt certificate for the domain in your `Caddyfile`. Just make sure ports 80
and 443 are open (A1) and DNS resolves to the VM.

Visit `https://your-domain.com` 🎉

---

## Path B — CPU, cheap, scale-to-zero (Azure Container Apps)

No GPU; single-molecule generation on CPU takes a few seconds — fine for a public demo.

```powershell
# 1. registry + build in the cloud (no local Docker needed)
az acr create -g molvae-rg -n molvaeacr --sku Basic
az acr build -r molvaeacr -t molvae:cpu `
  --build-arg none .                         # edit Dockerfile base to python:3.11-slim first

# 2. upload artifacts to a storage share and mount it (see Azure Files docs), or bake a
#    small model into the image. Then deploy:
az containerapp env create -g molvae-rg -n molvae-env -l eastus
az containerapp create -g molvae-rg -n molvae-app --environment molvae-env `
  --image molvaeacr.azurecr.io/molvae:cpu --target-port 8000 --ingress external `
  --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 3 `
  --secrets foundrykey=$env:FOUNDRY_API_KEY azurekey=$env:AZURE_OPENAI_KEY `
  --env-vars MOLVAE_DEVICE=cpu MOLVAE_ART_DIR=/artifacts `
             FOUNDRY_API_KEY=secretref:foundrykey AZURE_OPENAI_KEY=secretref:azurekey
az containerapp show -g molvae-rg -n molvae-app --query properties.configuration.ingress.fqdn -o tsv
```
Container Apps gives you a free `*.azurecontainerapps.io` HTTPS URL; add a **custom
domain** in Portal → the app → *Custom domains* (it manages the cert for you). Set
`--min-replicas 0` to scale to zero (pay nothing when idle).

---

## Secrets the right way (production)

Don't bake keys into the image. Either use a `.env` file on the VM (Path A) or **Azure
Key Vault**:
```powershell
az keyvault create -g molvae-rg -n molvae-kv
az keyvault secret set --vault-name molvae-kv -n FOUNDRY-API-KEY --value "<key>"
# grant the VM's managed identity 'get' on secrets, then fetch at boot:
az vm identity assign -g molvae-rg -n molvae-gpu
```

---

## Costs, autoshutdown, scaling

- **Auto-shutdown the GPU VM nightly** (saves the most money):
  ```powershell
  az vm auto-shutdown -g molvae-rg -n molvae-gpu --time 0200
  ```
- **Deallocate when unused** (stop billing for compute): `az vm deallocate -g molvae-rg -n molvae-gpu`; restart with `az vm start`.
- **Scale**: one container = one model in GPU memory. For more traffic, add VM replicas
  behind **Azure Front Door** or an Application Gateway / Load Balancer. Container Apps
  (Path B) autoscales by HTTP concurrency.
- **Rough monthly**: NC4as_T4_v3 ~ $380/mo if always-on; ~ $60/mo if shut down 16 h/day;
  Container Apps CPU scale-to-zero ~ a few $/mo for light use.

## Monitoring + logs

```bash
docker compose logs -f molvae          # app logs
docker stats                           # CPU/GPU mem
nvidia-smi -l 2                        # GPU utilization
```
For production telemetry, enable **Azure Monitor** on the VM and ship container logs to
Log Analytics.

## Updating the app

```bash
cd ~/molvae && git pull       # or scp the changed files
docker compose up -d --build  # rebuild + restart with zero data loss (artifacts are mounted)
```
To roll out a newly trained model, just replace `./artifacts/checkpoints/best.pt` and
`docker compose restart molvae`.

## Troubleshooting

- `device: cpu` in `/health` though you wanted GPU → the toolkit step (A2) failed; re-run
  `docker run --rm --gpus all nvidia/cuda:...-base nvidia-smi`.
- Caddy not issuing a cert → DNS not resolving yet (wait), or ports 80/443 closed (A1),
  or the domain in `Caddyfile` doesn't match.
- LLM explanations blank → keys not in `.env`/env; check `docker compose exec molvae env | grep FOUNDRY`.
- OOM on a small GPU → set `MOLVAE_DEVICE=cuda` but lower the population in `design.py`,
  or use a T4/16 GB+ VM.
