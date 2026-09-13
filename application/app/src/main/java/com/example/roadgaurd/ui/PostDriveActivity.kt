package com.example.roadgaurd.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.roadgaurd.AppConfig
import com.example.roadgaurd.R
import com.example.roadgaurd.storage.SessionStore
import com.example.roadgaurd.upload.UploadProgress
import com.example.roadgaurd.upload.UploadService
import java.io.File

class PostDriveActivity : AppCompatActivity() {

    companion object {
        private const val STATE_WATCHING_UPLOAD = "watching_upload"
    }

    private var sessionId: String? = null
    private var sessionDirPath: String? = null
    private var videoPath: String? = null

    // Client-side sanity check before uploading (fast feedback). The server re-validates
    // authoritatively at /upload/init + /complete — keep these in sync with the server's
    // VIDEO_MIN_BYTES / VIDEO_MAX_BYTES.
    private val MIN_VIDEO_BYTES = 5L * 1024 * 1024         // 5 MB
    private val MAX_VIDEO_BYTES = 4L * 1024 * 1024 * 1024  // 4 GB

    // The 6 fixed-name sensor/calibration files that, with the video, form the "7-file"
    // contract the backend validates. (tags.json was removed — no longer captured/sent.)
    private val DATA_FILES = listOf(
        "frames.csv", "gyro.csv", "gravity.csv", "gps.csv", "linacc.csv", "intrinsics.json"
    )

    // True once THIS screen started (or re-attached to) the background upload of its session.
    // Only then is the upload's result shown here — never a stale result of an earlier attempt.
    private var watchingUpload = false

    private val uploadListener: (UploadProgress.State) -> Unit = { renderUpload(it) }

    // Android 13+ asks before an app may post notifications (the upload's progress and result).
    // The upload itself runs whatever the answer.
    private val notificationPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { startBackgroundUpload() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_post_drive)

        sessionId = intent.getStringExtra("session_id") ?: "session_${System.currentTimeMillis()}"
        sessionDirPath = intent.getStringExtra("session_dir")
        videoPath = intent.getStringExtra("video_path")
        watchingUpload = savedInstanceState?.getBoolean(STATE_WATCHING_UPLOAD) ?: false

        val btnNo = findViewById<Button>(R.id.btnNo)
        val btnYes = findViewById<Button>(R.id.btnYes)

        if (AppConfig.OFFLINE_MODE) {
            // Nothing to submit to — the recording is already complete on disk (RecordingActivity
            // writes the video + CSVs before launching this screen). Turn the question into a
            // receipt showing where the data landed, with a single way out. No skipped.flag is
            // written: that flag only means "don't nag me to upload", and offline it would be
            // misleading (the retention sweep it feeds is disabled anyway).
            findViewById<TextView>(R.id.tvPostTitle).text = "Drive saved on this phone"
            findViewById<TextView>(R.id.tvPostSubtitle).text =
                "Offline mode — nothing was uploaded. Footage and sensor data are in:\n\n" +
                "Android/data/$packageName/files/sessions/${sessionId ?: ""}"
            btnYes.visibility = View.GONE
            btnNo.text = "Done"
            // Drop the "No" X icon — this is a confirmation now, not a refusal.
            (btnNo as? com.google.android.material.button.MaterialButton)?.icon = null
            btnNo.setOnClickListener { goHome() }
            return
        }

        btnNo.setOnClickListener {
            // While the upload runs this button reads "Close": leaving must not mark the drive
            // skipped, or a failed upload would never be offered again.
            if (!watchingUpload) sessionDirPath?.let { SessionStore.markSkipped(File(it)) }
            goHome()
        }
        btnYes.setOnClickListener { uploadDrive() }
    }

    override fun onStart() {
        super.onStart()
        if (!AppConfig.OFFLINE_MODE) UploadProgress.observe(uploadListener)
    }

    override fun onStop() {
        UploadProgress.remove(uploadListener)
        super.onStop()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putBoolean(STATE_WATCHING_UPLOAD, watchingUpload)
    }

    /**
     * Check the session, then hand it to [UploadService], which uploads it in the background
     * (direct to Cloudflare R2 — see the service). The user may leave this screen, or the app, at
     * any time; while this screen is visible it mirrors the upload's progress and result.
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

        // Verify all sensor/calibration files are present and non-empty before starting.
        val missingFiles = DATA_FILES.filter { name ->
            val f = File(dir, name)
            !f.exists() || f.length() == 0L
        }
        if (missingFiles.isNotEmpty()) {
            Toast.makeText(this,
                "Session data incomplete — missing: ${missingFiles.joinToString()}",
                Toast.LENGTH_LONG).show()
            return
        }

        val busyWith = UploadProgress.activeSessionDir
        if (busyWith != null && busyWith != dir.absolutePath) {
            Toast.makeText(this, "Another drive is still uploading — try again when it finishes.",
                Toast.LENGTH_LONG).show()
            return
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)  // -> startBackgroundUpload()
            return
        }
        startBackgroundUpload()
    }

    private fun startBackgroundUpload() {
        val sid = sessionId ?: return
        val dir = sessionDirPath ?: return
        val video = videoPath ?: return
        watchingUpload = true
        showProgress("Uploading… 0%", 0)
        UploadService.start(this, sid, dir, video)
    }

    /** Mirror the background upload of THIS session (main thread, only while visible). */
    private fun renderUpload(state: UploadProgress.State) {
        if (state.sessionId != sessionId) return
        when (state) {
            is UploadProgress.State.Uploading -> {
                watchingUpload = true
                showProgress("Uploading… ${state.percent}%", state.percent)
            }
            is UploadProgress.State.Finalizing -> {
                watchingUpload = true
                showProgress("Finalizing…", 100)
            }
            is UploadProgress.State.Succeeded -> if (finishWatching()) {
                findViewById<Button>(R.id.btnYes).isEnabled = false      // it is on the server now
                com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
                    .setTitle("Drive Uploaded Successfully!")
                    .setMessage("You just made the road safer — thank you for your contribution!")
                    .setPositiveButton("Awesome!") { _, _ -> goHome() }
                    .setCancelable(false)
                    .show()
            }
            is UploadProgress.State.Failed -> if (finishWatching()) {
                // Folder is deliberately KEPT on failure/rejection — the next launch offers it
                // again, and the 24h sweep (SessionStore) reclaims it later.
                Toast.makeText(this,
                    "Upload failed: ${state.message}. Saved on device — will retry/clean up later.",
                    Toast.LENGTH_LONG).show()
            }
            is UploadProgress.State.SessionExpired -> if (finishWatching()) handleSessionExpired()
        }
    }

    /** An upload ended: restore the screen. -> false if this screen was not watching it (stale). */
    private fun finishWatching(): Boolean {
        if (!watchingUpload) return false
        watchingUpload = false
        findViewById<ProgressBar>(R.id.pbUpload).visibility = View.GONE
        findViewById<TextView>(R.id.tvProgress).visibility = View.GONE
        findViewById<Button>(R.id.btnYes).isEnabled = true
        findViewById<Button>(R.id.btnNo).text = "No"
        return true
    }

    private fun showProgress(text: String, percent: Int) {
        findViewById<Button>(R.id.btnYes).isEnabled = false
        findViewById<Button>(R.id.btnNo).text = "Close"
        findViewById<ProgressBar>(R.id.pbUpload).apply {
            progress = percent
            visibility = View.VISIBLE
        }
        findViewById<TextView>(R.id.tvProgress).apply {
            this.text = "$text\nYou can leave the app — the upload continues in the background."
            visibility = View.VISIBLE
        }
    }

    private fun handleSessionExpired() {
        getSharedPreferences("roadguard", MODE_PRIVATE).edit().clear().apply()
        Toast.makeText(this, "Session expired. Please log in again.", Toast.LENGTH_LONG).show()
        startActivity(Intent(this, SplashActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        })
        finish()
    }

    private fun goHome() {
        startActivity(Intent(this, HomeActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
        })
        finish()
    }
}
