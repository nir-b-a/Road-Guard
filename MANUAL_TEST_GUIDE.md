# Manual end-to-end test: phone app -> backend -> DB -> police website

Use this when you want to manually verify a real phone can record a drive, upload it,
and have it show up correctly in the system. No emulator involved.

## 0. Prerequisites (one-time per machine)

- Node.js installed
- MongoDB running locally (`mongod`, or however you normally start it) and reachable at
  the `MONGO_URI` in `backend/.env`
- Android Studio + Android SDK installed
- A real Android phone with a USB cable
- `adb` available (comes with Android SDK platform-tools, usually at
  `%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe`)

## 1. Start the backend

```
cd backend
node server.js
```

Confirm in the console it logs `MongoDB Connected: ...` and `Server running on port 5000`.
(Optional) Open MongoDB Compass and connect to the same URI to watch the `drives` and
`violations` collections live.

You do NOT need to start the police website (`frontend/roadguard-authority`) for the
upload/save part of the test -- only for actually viewing the result afterward.

## 2. Connect the phone (USB method -- recommended, avoids Wi-Fi/router problems)

1. On the phone: **Settings -> About phone -> tap "Build number" 7x** to unlock Developer
   Options (skip if already enabled).
2. **Settings -> Developer options -> enable "USB debugging."**
3. Plug the phone into the PC via USB cable.
4. Approve the "Allow USB debugging?" popup on the phone.
5. Verify the PC sees it:
   ```
   adb devices
   ```
   should list your device as `device` (not `unauthorized`/`offline`).
6. Forward the phone's localhost:5000 to the PC's port 5000 over the cable:
   ```
   adb reverse tcp:5000 tcp:5000
   ```
   Re-run this any time the cable is unplugged and replugged -- the forward doesn't
   survive a disconnect.

## 3. Point the app at the right address

Because of the `adb reverse` tunnel, the app should call `127.0.0.1`, not a LAN IP.

In these 4 files, set:
```kotlin
private val BASE_URL = "http://127.0.0.1:5000/api"
```
- `application/app/src/main/java/com/example/roadgaurd/ui/SplashActivity.kt`
- `application/app/src/main/java/com/example/roadgaurd/ui/HomeActivity.kt`
- `application/app/src/main/java/com/example/roadgaurd/ui/PostDriveActivity.kt`
- `application/app/src/main/java/com/example/roadgaurd/ui/NotificationsActivity.kt`


## 4. Make sure login isn't using the fake dev token

In `SplashActivity.kt`, check:
```kotlin
const val BYPASS_LOGIN = true   // <-- must be false for a real test
```
Set it to `false`. With it `true`, the app stores a fake `"dev-offline-token"` that the
backend will always reject with `401`.

If the phone previously had the app installed with `BYPASS_LOGIN = true`, the fake
token is already cached on-device. A normal reinstall keeps app data, so do a clean
reinstall:
```
adb uninstall com.example.roadgaurd
```
(`adb install` again after building, see next step.)

## 5. Build and install the app

Either:
- **Android Studio**: open the project, hit Run, select the connected physical device.

Or from the command line:
```
cd application
./gradlew.bat assembleDebug
adb install app\build\outputs\apk\debug\app-debug.apk
```
APK ends up at `application/app/build/outputs/apk/debug/app-debug.apk`.

(First time on a fresh machine: if the build complains about missing SDK location,
create `application/local.properties` with:
```
sdk.dir=C:\\Users\\<you>\\AppData\\Local\\Android\\Sdk
```
adjusted to wherever the Android SDK is installed on that machine.)

## 6. Run the test

1. Open the app on the phone.
2. Register or log in with a real account (role: driver).
3. Record a short video (a stationary/room video is fine -- speed=0 and static GPS do
   not block anything; nothing in the upload or detection code validates GPS/speed).
4. Stop -> "Yes" to upload.
5. Confirm success ("Drive uploaded!" toast, no 401/network error).

## 7. Verify it landed correctly

- **Backend console**: should log the `POST /driver/upload` request.
- **MongoDB** (`drives` collection): a new doc with `status: 'pending'`, a `files.video`
  path, and the sensor file paths (`files.gps`, `files.gyro`, etc.).
- **Disk**: files under
  `backend/uploads/sessions/<sessionId>/` -- the video plus `frames.csv`, `gps.csv`,
  `gravity.csv`, `gyro.csv`, `linacc.csv`, `intrinsics.json`, `tags.json`.

## 8. (Optional) Simulate a violation hit, to test the website display

The brain (`main.py` / `tools/pipeline/roadguard.py`) does not currently auto-run after
upload and does not POST results back to the backend -- that integration isn't built
yet. To still test backend -> website communication, POST directly to the internal
endpoint the brain would call:

```
curl -X POST http://localhost:5000/api/internal/violation \
  -H "Content-Type: application/json" \
  -d "{\"driveId\": \"<the drive's _id from Mongo>\", \"videoClipPath\": \"uploads/sessions/<sessionId>/<videofile>.mp4\", \"carId\": \"123-45-678\", \"calculatedSpeed\": 87, \"lat\": 32.0625, \"lon\": 34.8349}"
```

This creates a real `Violation` doc and flips the `Drive` to `processed`. Then start
the website (`cd frontend/roadguard-authority && npm run dev`), log in with an
authority account, and check the Violations page -- the clip should actually play back
(it's the real uploaded video).

