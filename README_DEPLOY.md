Render deployment guide for foci-bot

This document explains how to deploy the foci-bot Streamlit app to Render.com and the recommended build/start commands.

1) Prepare the repository

- runtime.txt (added) pins the Python runtime to: python-3.11.13
- start.sh (added) contains a small wrapper to start Streamlit using the PORT provided by Render.
- requirements.txt (updated) pins key dependencies required for a reproducible build.

2) Create a new Web Service on Render

- Sign in to https://render.com and go to "New" -> "Web Service".
- Connect your GitHub account and select the repository: czuni09/foci-bot.
- Choose the branch: main.

3) Environment & Runtime

- Render will detect Python from runtime.txt (python-3.11.13). If needed, set the Environment to "Python" in the service settings.

4) Build Command

Use a robust build command that upgrades pip/tools and then installs requirements:

pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

This ensures any wheel/build tools are current and then installs pinned dependencies.

5) Start Command

Use the start script committed to the repo:

bash start.sh

This runs:

streamlit run streamlit_app.py --server.port $PORT --server.headless true

Render sets the PORT environment variable automatically. The start script falls back to port 8501 if PORT is not set.

6) Health Check & Service Settings

- Optionally set a health check path (e.g., "/") if your app serves a root route.
- Choose an instance type appropriate for your load. For lightweight testing, the free or starter instance may suffice.

7) Environment Variables

- If your app requires any secrets (API keys, database URLs), configure them under the "Environment" -> "Environment Variables" section in the Render dashboard. Never commit secrets to the repository.

8) Automatic Deploys

- Enable "Auto Deploy" to deploy on each push to the main branch.
- You can also trigger manual deploys from the Render dashboard.

9) Troubleshooting

- If dependencies fail to install, check the build logs in Render. The build command above should surface detailed pip errors.
- If Streamlit fails to start, confirm the start command in the service settings is exactly: bash start.sh
- If you need a different Python minor/patch version, update runtime.txt and redeploy.

10) Example Render settings recap

- Branch: main
- Build Command: pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
- Start Command: bash start.sh
- Runtime: Python (detected from runtime.txt)
- Environment Variables: add any required secrets (API keys, DB URIs)

11) Notes

- The repository includes pinned versions for core networking packages (aiohttp, yarl, multidict, frozenlist) to avoid binary compatibility issues.
- If you add or change Python packages, update requirements.txt and push to main to trigger a new deploy.

If you want, I can also add recommended health check endpoints or a Render cron job example for periodic tasks.