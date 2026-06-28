# How to Test the Android App End-to-End

## Prerequisites

- Node.js installed
- Android phone with USB debugging enabled (Settings → About Phone → tap Build Number 7x → Developer Options → USB Debugging)
- ngrok installed (`npm install -g ngrok` or download from ngrok.com)
- MongoDB running locally
- Cloudflare R2 bucket created (named `test` or whatever you set in `.env`)

---

## 1. Set up `backend/.env`

Create `backend/.env` (it is gitignored — never commit it):

```
PORT=5000
MONGO_URI=mongodb://localhost:27017/roadguard
JWT_SECRET=roadguard_dev_secret_2026
INVITE_CODE=ROADGUARD-2026
R2_ACCOUNT_ID=<your cloudflare account id>
R2_ACCESS_KEY_ID=<your R2 access key id>
R2_SECRET_ACCESS_KEY=<your R2 secret access key>
R2_BUCKET=test
R2_ENDPOINT=https://<your_account_id>.r2.cloudflarestorage.com
R2_PRESIGN_EXPIRY=900
```

- Get R2 credentials from: Cloudflare Dashboard → R2 → Manage R2 API Tokens → Create API Token (Object Read & Write)
- Your Account ID is on the Cloudflare home dashboard right sidebar
- **Never paste secrets in chat or commit them to git**

---

## 2. Start the server

```bash
cd backend
npm install       # only needed first time or after pulling new changes
node server.js
```

You should see: `Server running on port 5000`

---

## 3. Start ngrok

In a separate terminal:

```bash
ngrok http 5000
```

Copy the `https://` URL it gives you (e.g. `https://abc123.ngrok-free.app`).  
Leave this terminal open — closing it kills the tunnel.

> **Note:** The ngrok URL changes every time you restart ngrok (free tier). You'll need to update the app settings each time.

---

## 4. Build and install the Android app

Make sure your phone is connected via USB and USB debugging is enabled:

```bash
# From the repo root
cd application
./gradlew assembleDebug
```

Then install:
```
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Or on Windows, use the full adb path:
```
%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
```

---

## 5. Configure the server URL in the app

1. Open the app on your phone
2. On the login screen, tap **Server Settings**
3. Enter: `https://<your-ngrok-url>.ngrok-free.app/api` (include `/api` at the end)
4. Tap **Save**

> Do this every time you restart ngrok since the URL changes.

---

## 6. Run order (every session)

1. Start MongoDB (if not running as a service)
2. `node server.js`
3. `ngrok http 5000`
4. Update the URL in app Settings with the new ngrok URL
5. Register / Login in the app

---

## 7. Testing a drive upload

1. Login to the app
2. Record a drive
3. On the post-drive screen, tap **Yes** to upload
4. The app uploads the video + sensor files directly to Cloudflare R2
5. The server marks the drive as `queued`

> The drive is **not processed automatically**. To run the CV pipeline (violation detection, speed estimation), you need to run `worker.py` separately on a machine with the Python/GPU environment set up.

---

## Notes

- **Video stays on phone** until upload succeeds. If you tap No or the upload fails, the video is auto-deleted after 24 hours by a sweep that runs when HomeActivity opens.
- **worker.py** is the bridge between the server and the CV pipeline. Without it, drives sit as `queued` in the database indefinitely.
- The server and phone **do not need to be on the same WiFi** as long as ngrok is running.
