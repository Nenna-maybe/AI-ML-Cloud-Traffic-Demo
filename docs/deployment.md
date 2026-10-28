# Deployment Guide – Streamlit Community Cloud

> The app is **not deployed** until you complete these steps yourself and test the public URL.

## Before you start

- A GitHub account and the project pushed to a GitHub repository (see README → "Git and GitHub").
- The repository must contain `app.py`, `requirements.txt` and `packages.txt` at its root.
- Do **not** push `.env`, `.streamlit/secrets.toml`, `*.db` files or model weights (`.gitignore` already blocks them).

## Steps

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. **Repository:** `<your-username>/AI-ML-Cloud-Traffic-Demo`
4. **Branch:** `main`
5. **Main file path:** `app.py`
6. Open **Advanced settings**:
   - **Python version:** choose **3.12** (recommended).
   - **Secrets:** optional. Paste values from `.streamlit/secrets.toml.example` only if you want an admin PIN or S3. The app works without any secrets.
7. Click **Deploy**. The first build installs PyTorch (CPU) and Ultralytics and can take several minutes.
8. Open **Manage app → Logs** (bottom-right) to watch the build. If it fails, read the last error lines.
9. When the app opens, run the smoke test below.

## Smoke-test checklist (after deployment)

- [ ] Dashboard loads with zero counts (fresh database).
- [ ] Cloud & Settings → **Load SAMPLE data**, turn on the sidebar toggle → dashboard charts appear.
- [ ] Image Detection → upload a JPG → annotated image, counts and congestion gauge appear. (The first run downloads YOLO weights – wait for the spinner.)
- [ ] Download the annotated image.
- [ ] Upload a non-image file renamed to `.jpg` → a friendly error appears.
- [ ] Video Analysis → upload a short MP4 (≤ 20 s recommended on free hosting) → progress bar, per-frame chart, sample frames.
- [ ] Incidents → report an incident; try a 2-character location (should be rejected); edit and resolve it.
- [ ] Priority Engine → select a detection and an incident → priority, explanation and action appear.
- [ ] History → open the priority assessment → **Verify: recompute** shows "Reproduced exactly".
- [ ] Analytics → all three CSV exports download.

## Limitations of the hosted environment (important)

- **Storage is temporary.** Streamlit Community Cloud containers restart (after inactivity, redeploys or updates). The SQLite database, uploaded images and annotated outputs are **lost** on restart. For persistent data you would need S3 for files (supported, optional) and a managed database such as Amazon RDS (proposed, not implemented).
- **CPU only and limited memory** (about 1 GB on the free tier, subject to change). Keep images small and videos short. Video processing works but is slow; if the app runs out of memory, reduce `MAX_VIDEO_FRAMES` in secrets (e.g. `MAX_VIDEO_FRAMES = "8"`).
- **Model download:** weights (~6 MB) download from GitHub on first use after every restart.
- **Public app:** anyone with the link can use it. Set `ADMIN_PIN` in secrets to protect incident editing. This PIN is a simple shared secret, not user accounts.

## Troubleshooting

| Problem in logs | Fix |
|---|---|
| `ImportError: libGL.so.1` | Make sure `packages.txt` (with `libgl1` and `libglib2.0-0`) is in the repo root, then **Reboot app**. |
| Build runs out of space / takes forever downloading `nvidia-*` packages | Check that the first line of `requirements.txt` is the CPU index line (`--extra-index-url https://download.pytorch.org/whl/cpu`). |
| `Could not load YOLO weights` | Temporary network issue; reload. Or commit the `.pt` file deliberately (remove `*.pt` from `.gitignore`) – only if you accept the extra 6 MB. |
| App restarts while processing a video | Out of memory: use a shorter video or fewer frames. |

## Optional: enabling S3 on the hosted app

1. Create the bucket and IAM user as described in README → "Optional AWS S3 setup".
2. In **Manage app → Settings → Secrets** add `USE_S3`, `S3_BUCKET`, `AWS_DEFAULT_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.
3. Save → the app restarts. Cloud & Settings → **Check storage status** should say "Connected".
