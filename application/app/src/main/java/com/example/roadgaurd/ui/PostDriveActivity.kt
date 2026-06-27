package com.example.roadgaurd.ui

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.roadgaurd.AppConfig
import com.example.roadgaurd.R
import com.example.roadgaurd.storage.SessionStore
import okhttp3.MediaType
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okio.Buffer
import okio.BufferedSink
import okio.source
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread

class PostDriveActivity : AppCompatActivity() {

    private var sessionId: String? = null
    private var sessionDirPath: String? = null
    private var videoPath: String? = null
    private val BASE_URL get() = AppConfig.getBaseUrl(this)

    // Client-side sanity check before uploading (fast feedback). The server re-validates
    // authoritatively at /upload/init + /complete — keep these in sync with the server's
    // VIDEO_MIN_BYTES / VIDEO_MAX_BYTES.
    private val MIN_VIDEO_BYTES = 100L * 1024 * 1024       // 100 MB
    private val MAX_VIDEO_BYTES = 4L * 1024 * 1024 * 1024  // 4 GB

    // The 6 fixed-name sensor/calibration files that, with the video, form the "7-file"
    // contract the backend validates. (tags.json was removed — no longer captured/sent.)
    private val DATA_FILES = listOf(
        "frames.csv", "gyro.csv", "gravity.csv", "gps.csv", "linacc.csv", "intrinsics.json"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_post_drive)

        sessionId = intent.getStringExtra("session_id") ?: "session_${System.currentTimeMillis()}"
        sessionDirPath = intent.getStringExtra("session_dir")
        videoPath = intent.getStringExtra("video_path")

        findViewById<Button>(R.id.btnNo).setOnClickListener { goHome() }
        findViewById<Button>(R.id.btnYes).setOnClickListener { uploadDrive() }
    }

    /**
     * Direct-to-storage upload in three steps so the big video never flows through our
     * server:
     *   1. POST /driver/upload/init      -> presigned PUT URLs (one per file)
     *   2. PUT each file straight to Cloudflare R2
     *   3. POST /driver/upload/complete  -> server validates the objects and queues the drive
     * On success the session folder is deleted (it now lives in R2). On any failure the
     * folder is kept so the launch-time sweep can reclaim it later.
     */
    private fun uploadDrive() {
        val dir = sessionDirPath?.let { File(it) }
        if (dir == null || !dir.exists()) {
            Toast.makeText(this, "Session folder missing", Toast.LENGTH_LONG).show()
            return
        }
        val videoFile = videoPath?.let { File(it) }
        if (videoFile == null || !videoFile.exists()) {
            Toast.makeText(this, "Video file missing", Toast.LENGTH_LONG).show()
            return
        }

        // Pre-check the video size so we don't waste a long upload the server would reject.
        val size = videoFile.length()
        if (size < MIN_VIDEO_BYTES || size > MAX_VIDEO_BYTES) {
            val mb = size / (1024 * 1024)
            val minMb = MIN_VIDEO_BYTES / (1024 * 1024)
            val maxGb = MAX_VIDEO_BYTES / (1024L * 1024 * 1024)
            Toast.makeText(this,
                "Video is $mb MB — must be between $minMb MB and $maxGb GB. Not uploaded.",
                Toast.LENGTH_LONG).show()
            return
        }

        val btnYes = findViewById<Button>(R.id.btnYes)
        val progressBar = findViewById<ProgressBar>(R.id.pbUpload)
        val tvProgress = findViewById<TextView>(R.id.tvProgress)
        btnYes.isEnabled = false
        Toast.makeText(this, "Uploading drive…", Toast.LENGTH_SHORT).show()

        val token = getSharedPreferences("roadguard", MODE_PRIVATE).getString("token", "") ?: ""
        val sid = sessionId ?: ""
        val client = OkHttpClient()

        // Network must not run on the UI thread; these calls are sequential & dependent.
        thread {
            try {
                // 1. init -> presigned PUT URLs
                val initBody = JSONObject()
                    .put("sessionId", sid)
                    .put("videoName", videoFile.name)
                    .put("videoSize", videoFile.length())
                    .toString()
                val initReq = Request.Builder()
                    .url("$BASE_URL/driver/upload/init")
                    .addHeader("Authorization", "Bearer $token")
                    .addHeader("ngrok-skip-browser-warning", "true")
                    .post(initBody.toRequestBody("application/json".toMediaType()))
                    .build()
                val uploads = client.newCall(initReq).execute().use { resp ->
                    val text = resp.body?.string() ?: ""
                    if (!resp.isSuccessful) throw IOException("init failed (${resp.code}): $text")
                    JSONObject(text).getJSONObject("data").getJSONObject("uploads")
                }

                // Build the concrete upload list (video + the sensor files that exist and
                // were presigned) so we can show a single bar across ALL bytes, not per file.
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
                runOnUiThread {
                    progressBar.progress = 0
                    progressBar.visibility = View.VISIBLE
                    tvProgress.text = "Uploading… 0%"
                    tvProgress.visibility = View.VISIBLE
                }

                // 2. PUT every file directly to R2 using its presigned URL, reporting bytes
                // as they stream so the bar reflects the whole upload (video dominates).
                for ((name, f, mime) in plan) {
                    putFile(client, uploads.getString(name), f, mime) { chunk ->
                        val sent = sentBytes.addAndGet(chunk)
                        val pct = if (totalBytes > 0) ((sent * 100) / totalBytes).toInt().coerceIn(0, 100) else 100
                        if (pct != lastPct) {
                            lastPct = pct
                            runOnUiThread {
                                progressBar.progress = pct
                                tvProgress.text = "Uploading… $pct%"
                            }
                        }
                    }
                }

                runOnUiThread { tvProgress.text = "Finalizing…" }

                // 3. complete -> server HEAD-validates the 7 objects and queues the drive.
                val completeReq = Request.Builder()
                    .url("$BASE_URL/driver/upload/complete")
                    .addHeader("Authorization", "Bearer $token")
                    .addHeader("ngrok-skip-browser-warning", "true")
                    .post(JSONObject().put("sessionId", sid).toString().toRequestBody("application/json".toMediaType()))
                    .build()
                client.newCall(completeReq).execute().use { resp ->
                    val text = resp.body?.string() ?: ""
                    if (!resp.isSuccessful) throw IOException("complete failed (${resp.code}): $text")
                }

                runOnUiThread {
                    progressBar.visibility = View.GONE
                    tvProgress.visibility = View.GONE
                    SessionStore.deleteSession(dir)   // it's in R2 now — reclaim device storage
                    Toast.makeText(this, "Drive uploaded!", Toast.LENGTH_LONG).show()
                    goHome()
                }
            } catch (e: Exception) {
                Log.w("PostDrive", "Upload failed: ${e.message}")
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    tvProgress.visibility = View.GONE
                    btnYes.isEnabled = true
                    // Folder is deliberately KEPT on failure/rejection — the 24h launch-time
                    // sweep (SessionStore) reclaims it later; nothing is deleted here.
                    Toast.makeText(this,
                        "Upload failed: ${e.message}. Saved on device — will retry/clean up later.",
                        Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    /**
     * Upload one file with a single HTTP PUT to its presigned R2 URL. [onChunk] is invoked
     * with the number of bytes flushed each time a chunk is written, so the caller can drive
     * a progress bar spanning the whole multi-file upload.
     */
    private fun putFile(client: OkHttpClient, url: String, file: File, mime: String, onChunk: (Long) -> Unit) {
        val req = Request.Builder()
            .url(url)
            .put(ProgressRequestBody(file, mime.toMediaType(), onChunk))
            .build()
        client.newCall(req).execute().use { resp ->
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

    private fun goHome() {
        startActivity(Intent(this, HomeActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
        })
        finish()
    }
}
