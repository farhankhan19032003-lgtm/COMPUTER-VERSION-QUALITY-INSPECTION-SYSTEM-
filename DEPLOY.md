# Deploying VisionInspect on Render

Render builds the `Dockerfile` from a Git repository and runs it as a web service. Everything needed is in this repo:
`render.yaml` (the service), `Dockerfile` (the image, models included), `.dockerignore`.

## Before you start

| You need | Why |
|---|---|
| A **private** GitHub (or GitLab) repository | Render deploys from Git. Keep it private: the repo holds ~135 MB of trained models, some trained on non-commercial datasets (HAM10000 is CC BY-NC). |
| A Render account and a **paid plan, Standard (2 GB RAM)** | The app uses about 730 MB once every model has been used. The 512 MB plans are killed for running out of memory. |
| Your Supabase project | The schema applied (`supabase/schema.sql`), the storage bucket created, and Auth set up. See `AUTH_SETUP.md`. |

## 1. Put the code on GitHub

`.env`, `data/`, `reports/`, your local tool config (`.claude/`, `.mcp.json`) and old model backups are already ignored.
The trained models are **included** on purpose.

```bash
git init            # already done once; harmless to repeat
git add -A
git status          # check: no .env, no data/, about 110 files
git commit -m "VisionInspect final version"
git branch -M main
git remote add origin https://github.com/<you>/<private-repo>.git
git push -u origin main
```

GitHub rejects files over 100 MB; the largest here is 42.7 MB, so a plain push works (no Git LFS needed).

## 2. Create the service

1. Render dashboard, **New**, **Blueprint**, choose the repository. Render reads `render.yaml`.
2. It will ask for the values marked `sync: false`. Fill them in (copy from your local `.env`):
   - `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `SUPABASE_JWT_SECRET`, `SUPABASE_BUCKET`
   - `CORS_ORIGINS`: for now `*`; you will tighten it in step 4.
3. Apply. The first build downloads PyTorch and takes 10 to 20 minutes.

Fixed in `render.yaml`: `AUTH_REQUIRED=true`, `ALLOW_ANONYMOUS_PATIENTS=false`, `DEVICE=cpu`, `MAX_UPLOAD_MB=10`,
health check `/health`, `autoDeploy: false` (you press Deploy yourself, so a push cannot break a live site).

## 3. Check it works

Replace the address with your Render URL.

```bash
curl https://<service>.onrender.com/health      # {"status":"ok"}
curl https://<service>.onrender.com/ready       # "store":"supabase", "detector":"yolo", "anomaly_banks":["phone"]
```

Open the URL in a browser, sign in, upload a photo on the Inspect tab. `/ready` must say `store: supabase`; if it says
`memory`, the Supabase variables are missing and data would be lost on every restart.

## 4. Lock it down

- Set `CORS_ORIGINS` to your real address (for example `https://visioninspect.onrender.com`), not `*`, then redeploy.
- In Supabase, **Authentication, URL Configuration**: set *Site URL* and add the Render address to *Redirect URLs*.
- The API docs page (`/docs`) is public. That is fine for a demo; for production put the service behind a login or
  remove the route.
- There is no rate limiting in the app. Render's free tier of protection is basic; add a limit (a proxy or a Render
  firewall rule) before real users upload photos.

## Staying up on a small server

On a shared host the app can be restarted (visitors see "502 Bad Gateway" for about half a minute) if many photos are analysed
at the same moment or PyTorch starts more CPU threads than the plan has. Two settings control this; the defaults are
safe, change them only if you have more CPU:

| Variable | Default | Meaning |
|---|---|---|
| `TORCH_THREADS` | `0` (automatic) | CPU threads PyTorch and OpenCV may use. Automatic reads the container's CPU quota: 1 thread on a 1-CPU plan. |
| `MAX_CONCURRENT_INFERENCE` | `0` (automatic) | Photos analysed at the same time; the rest wait. Automatic: 1 on a 1-CPU plan, otherwise 2. |
| `INFERENCE_WAIT_SECONDS` | `45` | How long a photo waits for a free slot before the server answers 503 "busy" (so the proxy never times out with a 502). |

`/ready` now also tells you what actually loaded (no login needed):
`"product_classifier": true`, `"screening_models": ["chest_xray","malaria_smear","retina_dr","skin_lesion"]`,
`"torch_threads"`, `"max_concurrent_inference"`. If `product_classifier` is `false`, auto-detect has fallen back to YOLO only.

Every model is also loaded once in the background right after start-up (`WARMUP=true`, set in the Dockerfile), so the first
photo is not slow. `/ready` shows the result: `"warmup": {"status": "running"}` for the first seconds, then
`{"status": "ok", "stages": {...}, "seconds": ..., "rss_mb": ...}`, or `{"status": "error", "error": "..."}` with the reason.
While the models load (a few minutes after every restart on a small plan) uploads get a quick `503` with a readable message instead of a 502.
`"cpu_quota"` is the CPUs your plan really gives the container; `"cpu_matmul_ms"` is a speed test (a laptop core takes about 15 to 30 ms; hundreds mean a very slow host); `"one_normal_photo_s"` is how long a normal photo takes once everything is loaded.
`"memory_mb"` is the memory the app uses right now: if it gets close to your plan's limit, move to a bigger plan.

If the service still restarts, open the service's **Events** tab (reason such as "out of memory" or "failed health check")
and **Metrics, Memory**; that tells you whether to move to a bigger plan.

## Using it on phones and tablets

Open the Render address in the phone's browser and sign in. The dashboard is responsive, and the camera buttons use the
rear camera on phones (the pulse check uses the front camera). Browsers only allow the camera on **HTTPS**, which Render
provides; it will not work over plain `http://`. Grant the camera permission when the browser asks.

## What to know

- **First request after a restart is slow** (about 10 to 20 s): models load on first use.
- **The disk is temporary.** Nothing is saved on the server itself; inspections and patients go to Supabase. Restarts and
  redeploys lose only the in-memory counters.
- **Cost.** Standard is billed per month while the service exists; suspend or delete it when you do not need it.
- **Updating models.** Retrain, copy the new files into `models/`, commit, push, press Deploy. Each new 42 MB model
  version grows the repository history.
- **What this deploy is.** Real photos currently give REVIEW, not FAIL, until you build per-product banks from your rig photos
  (`scripts/build_memory_bank.py`, see README "Project status"). The health models are research-grade, shown as wellness
  screening only. Do not market them as a medical device.

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| Build fails while downloading packages | Retry the deploy; PyTorch's index is large. |
| Service restarts again and again, log says `Killed` | Out of memory: the plan has less than 2 GB. |
| `/ready` says `"store":"memory"` | A Supabase variable is empty or wrong. |
| Login works but the dashboard calls fail in the browser | `CORS_ORIGINS` does not match the site address. |
| 401 on every request | `SUPABASE_JWT_SECRET` is wrong (use the project's JWT secret, not the anon key). |
