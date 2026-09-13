package com.example.roadgaurd.upload

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.example.roadgaurd.AppConfig
import com.example.roadgaurd.storage.SessionStore
import com.example.roadgaurd.ui.PostDriveActivity
import okhttp3.Call
import okhttp3.MediaType
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okio.Buffer
import okio.BufferedSink
import okio.source
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread

/**
 * Uploads ONE recorded drive in the background, as a foreground service.
 *
 * Why a service: a drive is 100 MB–4 GB, so the upload takes minutes. On a plain thread inside
 * PostDriveActivity it could die as soon as the user left the app and Android reclaimed the
 * process. A foreground service shows an ongoing "Uploading drive… X%" notification, which is also
 * what tells Android to keep the process — and its network access — alive; a partial wake lock
 * keeps the CPU running with the screen off. The user can leave the app at any time; while
 * PostDriveActivity is visible it mirrors the progress through [UploadProgress].
 *
 * The upload itself is the same 3-step direct-to-R2 flow as before:
 *   1. POST /driver/upload/init     -> presigned PUT URLs (one per file)
 *   2. PUT each file straight to R2
 *   3. POST /driver/upload/complete -> server validates the objects and queues the drive
 * On success the session is marked uploaded (SessionStore sweeps it after the grace period). On any
 * failure the folder is KEPT: tapping the "upload failed" notification reopens the upload screen,
 * and the next launch's "Unfinished Upload" prompt offers it too. One upload at a time.
 */
class UploadService : Service() {

    companion object {
        private const val TAG = "UploadService"

        private const val EXTRA_SESSION_ID = "session_id"
        private const val EXTRA_SESSION_DIR = "session_dir"
        private const val EXTRA_VIDEO_PATH = "video_path"

        private const val CHANNEL_PROGRESS = "drive_upload_progress"   // silent, ongoing
        private const val CHANNEL_RESULT = "drive_upload_result"       // alerts once when done
        private const val NOTIF_PROGRESS = 4101
        private const val NOTIF_RESULT = 4102

        // Android 15 allows a dataSync foreground service 6 h a day; never hold the CPU longer.
        private const val WAKE_LOCK_TIMEOUT_MS = 6L * 60 * 60 * 1000

        // The 6 fixed-name sensor/calibration files that, with the video, form the "7-file"
        // contract the backend validates.
        private val DATA_FILES = listOf(
            "frames.csv", "gyro.csv", "gravity.csv", "gps.csv", "linacc.csv", "intrinsics.json"
        )

        /**
         * Start uploading a session in the background. Call it from a visible screen (Android only
         * lets an app in the foreground start a foreground service). Ignored while another upload
         * runs — callers check [UploadProgress.activeSessionDir] first to tell the user.
         */
        fun start(context: Context, sessionId: String, sessionDir: String, videoPath: String) {
            if (UploadProgress.activeSessionDir != null) return
            UploadProgress.activeSessionDir = File(sessionDir).absolutePath
            UploadProgress.post(UploadProgress.State.Uploading(sessionId, 0))
            val intent = Intent(context, UploadService::class.java)
                .putExtra(EXTRA_SESSION_ID, sessionId)
                .putExtra(EXTRA_SESSION_DIR, sessionDir)
                .putExtra(EXTRA_VIDEO_PATH, videoPath)
            try {
                ContextCompat.startForegroundService(context, intent)
            } catch (e: RuntimeException) {       // Android refused a foreground service right now
                Log.w(TAG, "Could not start the upload service: ${e.message}")
                UploadProgress.activeSessionDir = null
                UploadProgress.post(UploadProgress.State.Failed(sessionId, "could not start the upload"))
            }
        }
    }

    /** Thrown on HTTP 401 so the result can say "log in again" instead of a raw error. */
    private class SessionExpiredException : IOException("session expired")

    private val mainHandler = Handler(Looper.getMainLooper())
    private var uploadThread: Thread? = null              // main thread only
    private var wakeLock: PowerManager.WakeLock? = null   // main thread only
    @Volatile private var cancelled = false
    @Volatile private var activeCall: Call? = null
    @Volatile private var lastProgress: Notification? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createChannels()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Every startForegroundService() must be answered by startForeground() within seconds —
        // even a request that is then ignored — or Android kills the app.
        startInForeground(lastProgress ?: progressNotification(0, "Starting…", indeterminate = true))
        if (uploadThread != null) return START_NOT_STICKY          // one upload at a time

        val sessionId = intent?.getStringExtra(EXTRA_SESSION_ID)
        val sessionDir = intent?.getStringExtra(EXTRA_SESSION_DIR)
        val videoPath = intent?.getStringExtra(EXTRA_VIDEO_PATH)
        if (sessionId == null || sessionDir == null || videoPath == null) {
            UploadProgress.activeSessionDir = null
            stopInForeground()
            return START_NOT_STICKY
        }

        cancelled = false
        UploadProgress.activeSessionDir = File(sessionDir).absolutePath
        acquireWakeLock()
        uploadThread = thread(name = "drive-upload") {
            val result = try {
                runUpload(sessionId, File(sessionDir), File(videoPath))
                SessionStore.markUploaded(File(sessionDir))
                UploadProgress.State.Succeeded(sessionId)
            } catch (e: SessionExpiredException) {
                // As the upload screen always did: the stored login is no longer valid.
                getSharedPreferences("roadguard", MODE_PRIVATE).edit().clear().apply()
                UploadProgress.State.SessionExpired(sessionId)
            } catch (e: Exception) {
                Log.w(TAG, "Upload failed: ${e.message}")
                UploadProgress.State.Failed(sessionId, e.message ?: e.javaClass.simpleName)
            }
            showResult(result, sessionId, sessionDir, videoPath)
            mainHandler.post { finishUpload(result) }
        }
        // Not restarted by Android if killed: the kept session folder is the retry path.
        return START_NOT_STICKY
    }

    /** Main thread: publish the result, leave the foreground and stop. */
    private fun finishUpload(result: UploadProgress.State) {
        uploadThread = null
        lastProgress = null
        UploadProgress.activeSessionDir = null
        UploadProgress.post(result)
        releaseWakeLock()
        stopInForeground()
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        // Android 15+: the daily allowance for dataSync foreground services is used up, and the
        // service must stop within seconds. The upload fails like a network error would.
        Log.w(TAG, "Foreground-service time limit reached — cancelling the upload")
        cancelUpload()
        stopInForeground()
    }

    override fun onDestroy() {
        cancelUpload()          // no-op unless the service is destroyed mid-upload
        releaseWakeLock()
        super.onDestroy()
    }

    private fun cancelUpload() {
        cancelled = true
        activeCall?.cancel()
    }

    private fun stopInForeground() {
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    /**
     * The 3-step upload. Reports progress to [UploadProgress] and the notification as bytes stream
     * (the video dominates). Throws on any failure, [SessionExpiredException] on a 401.
     */
    private fun runUpload(sessionId: String, dir: File, videoFile: File) {
        val token = getSharedPreferences("roadguard", MODE_PRIVATE).getString("token", "") ?: ""
        val baseUrl = AppConfig.getBaseUrl(this)
        // More patient than OkHttp's 10 s defaults: in the background, with the screen off, a
        // network hand-over (Wi-Fi <-> mobile) can stall a request for a while — and a timeout
        // restarts the whole upload.
        val client = OkHttpClient.Builder()
            .connectTimeout(20, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build()

        // 1. init -> presigned PUT URLs
        val initBody = JSONObject()
            .put("sessionId", sessionId)
            .put("videoName", videoFile.name)
            .put("videoSize", videoFile.length())
            .toString()
        val initReq = Request.Builder()
            .url("$baseUrl/driver/upload/init")
            .addHeader("Authorization", "Bearer $token")
            .addHeader("ngrok-skip-browser-warning", "true")
            .post(initBody.toRequestBody("application/json".toMediaType()))
            .build()
        val uploads = execute(client, initReq).use { resp ->
            val text = resp.body?.string() ?: ""
            if (resp.code == 401) throw SessionExpiredException()
            if (!resp.isSuccessful) throw IOException("init failed (${resp.code}): $text")
            JSONObject(text).getJSONObject("data").getJSONObject("uploads")
        }

        // Build the concrete upload list (video + the sensor files that exist and were
        // presigned) so the progress spans ALL bytes, not one file.
        val plan = mutableListOf<Triple<String, File, String>>()
        plan.add(Triple(videoFile.name, videoFile, "video/mp4"))
        for (name in DATA_FILES) {
            val f = File(dir, name)
            if (f.exists() && uploads.has(name)) {
                val mime = if (name.endsWith(".json")) "application/json" else "text/csv"
                plan.add(Triple(name, f, mime))
            }
        }
        val totalBytes = plan.sumOf { it.second.length() }
        val sentBytes = AtomicLong(0)
        var lastPct = -1

        // 2. PUT every file directly to R2 using its presigned URL.
        for ((name, f, mime) in plan) {
            putFile(client, uploads.getString(name), f, mime) { chunk ->
                val sent = sentBytes.addAndGet(chunk)
                val pct = if (totalBytes > 0) ((sent * 100) / totalBytes).toInt().coerceIn(0, 100) else 100
                if (pct != lastPct) {
                    lastPct = pct
                    UploadProgress.post(UploadProgress.State.Uploading(sessionId, pct))
                    updateProgress(progressNotification(pct, "Uploading… $pct%"))
                }
            }
        }

        UploadProgress.post(UploadProgress.State.Finalizing(sessionId))
        updateProgress(progressNotification(100, "Finalizing…", indeterminate = true))

        // 3. complete -> server HEAD-validates the 7 objects and queues the drive.
        val completeReq = Request.Builder()
            .url("$baseUrl/driver/upload/complete")
            .addHeader("Authorization", "Bearer $token")
            .addHeader("ngrok-skip-browser-warning", "true")
            .post(JSONObject().put("sessionId", sessionId).toString().toRequestBody("application/json".toMediaType()))
            .build()
        execute(client, completeReq).use { resp ->
            val text = resp.body?.string() ?: ""
            if (resp.code == 401) throw SessionExpiredException()
            if (!resp.isSuccessful) throw IOException("complete failed (${resp.code}): $text")
        }
    }

    /** Run one request; [cancelUpload] aborts it, and every later one. */
    private fun execute(client: OkHttpClient, request: Request): Response {
        if (cancelled) throw IOException("upload cancelled")
        val call = client.newCall(request)
        activeCall = call
        if (cancelled) call.cancel()
        try {
            return call.execute()
        } finally {
            activeCall = null
        }
    }

    /**
     * Upload one file with a single HTTP PUT to its presigned R2 URL. [onChunk] is invoked with the
     * number of bytes flushed each time a chunk is written.
     */
    private fun putFile(client: OkHttpClient, url: String, file: File, mime: String, onChunk: (Long) -> Unit) {
        val req = Request.Builder()
            .url(url)
            .put(ProgressRequestBody(file, mime.toMediaType(), onChunk))
            .build()
        execute(client, req).use { resp ->
            if (!resp.isSuccessful) throw IOException("PUT ${file.name} failed (${resp.code})")
        }
    }

    /**
     * A streaming [RequestBody] over a file that reports upload progress. It keeps a known
     * Content-Length (so OkHttp streams instead of buffering) and calls [onChunk] after each
     * 64 KB block reaches the socket — identical wire bytes to a plain file body, just observed.
     */
    private class ProgressRequestBody(
        private val file: File,
        private val mime: MediaType,
        private val onChunk: (Long) -> Unit
    ) : RequestBody() {
        override fun contentType(): MediaType = mime
        override fun contentLength(): Long = file.length()
        override fun writeTo(sink: BufferedSink) {
            file.source().use { source ->
                val buf = Buffer()
                val segment = 64L * 1024
                while (true) {
                    val read = source.read(buf, segment)
                    if (read == -1L) break
                    sink.write(buf, read)
                    onChunk(read)
                }
            }
        }
    }

    // ---- wake lock ---------------------------------------------------------------------------

    private fun acquireWakeLock() {
        val power = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "RoadGuard:driveUpload").apply {
            setReferenceCounted(false)
            acquire(WAKE_LOCK_TIMEOUT_MS)
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    // ---- notifications -----------------------------------------------------------------------

    private fun createChannels() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(
            CHANNEL_PROGRESS, "Drive upload progress", NotificationManager.IMPORTANCE_LOW))
        manager.createNotificationChannel(NotificationChannel(
            CHANNEL_RESULT, "Drive upload result", NotificationManager.IMPORTANCE_DEFAULT))
    }

    private fun progressNotification(pct: Int, text: String, indeterminate: Boolean = false): Notification =
        NotificationCompat.Builder(this, CHANNEL_PROGRESS)
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentTitle("Uploading drive")
            .setContentText(text)
            .setProgress(100, pct, indeterminate)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setForegroundServiceBehavior(NotificationCompat.FOREGROUND_SERVICE_IMMEDIATE)
            .setContentIntent(openAppIntent())
            .build()

    private fun startInForeground(notification: Notification) {
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC else 0
        ServiceCompat.startForeground(this, NOTIF_PROGRESS, notification, type)
    }

    private fun updateProgress(notification: Notification) {
        lastProgress = notification
        notify(NOTIF_PROGRESS, notification)
    }

    private fun showResult(result: UploadProgress.State, sessionId: String, sessionDir: String, videoPath: String) {
        val (title, text) = when (result) {
            is UploadProgress.State.Succeeded ->
                "Drive uploaded ✓" to "You just made the road safer — thank you for your contribution!"
            is UploadProgress.State.SessionExpired ->
                "Drive upload failed" to "Your login expired. Open RoadGuard, log in again and upload the drive when asked."
            is UploadProgress.State.Failed ->
                "Drive upload failed" to "The drive is still saved on this phone — tap to try again. (${result.message})"
            else -> return
        }
        // A plain failure reopens that drive's upload screen; everything else just opens the app.
        val tap = if (result is UploadProgress.State.Failed) retryIntent(sessionId, sessionDir, videoPath) else openAppIntent()
        val icon = if (result is UploadProgress.State.Succeeded)
            android.R.drawable.stat_sys_upload_done else android.R.drawable.stat_notify_error
        notify(NOTIF_RESULT, NotificationCompat.Builder(this, CHANNEL_RESULT)
            .setSmallIcon(icon)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setContentIntent(tap)
            .setAutoCancel(true)
            .build())
    }

    /** Bring the app back the way its launcher icon does. */
    private fun openAppIntent(): PendingIntent? {
        val launch = packageManager.getLaunchIntentForPackage(packageName) ?: return null
        launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED)
        return PendingIntent.getActivity(this, 0, launch,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
    }

    private fun retryIntent(sessionId: String, sessionDir: String, videoPath: String): PendingIntent {
        val retry = Intent(this, PostDriveActivity::class.java)
            .putExtra("session_id", sessionId)
            .putExtra("session_dir", sessionDir)
            .putExtra("video_path", videoPath)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        return PendingIntent.getActivity(this, 1, retry,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
    }

    /** notify() that skips quietly when notifications are not allowed (the upload still runs). */
    private fun notify(id: Int, notification: Notification) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) return
        try {
            NotificationManagerCompat.from(this).notify(id, notification)
        } catch (e: SecurityException) {
            Log.w(TAG, "Notification skipped: ${e.message}")
        }
    }
}
