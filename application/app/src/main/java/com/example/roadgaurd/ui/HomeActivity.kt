package com.example.roadgaurd.ui

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.example.roadgaurd.AppConfig
import com.example.roadgaurd.R
import com.example.roadgaurd.storage.SessionStore
import com.example.roadgaurd.upload.UploadProgress

class HomeActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)

        // Reclaim un-uploaded sessions left on the device beyond the retention window,
        // then offer to retry any that failed due to a connection drop. Both are upload
        // bookkeeping: offline there is nothing to upload and nothing may be reclaimed,
        // so the recordings simply stay (SessionStore.sweepStaleSessions no-ops too).
        if (!AppConfig.OFFLINE_MODE) {
            SessionStore.sweepStaleSessions(this)
            checkPendingUploads()
        }

        findViewById<android.widget.Button>(R.id.btnStartDrive).setOnClickListener {
            startActivity(Intent(this, RecordingActivity::class.java))
        }

        findViewById<android.widget.Button>(R.id.btnSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        val btnLogout = findViewById<android.widget.Button>(R.id.btnLogout)
        btnLogout.setOnClickListener {
            val prefs = getSharedPreferences("roadguard", MODE_PRIVATE)
            prefs.edit().clear().apply()

            val intent = Intent(this, SplashActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            startActivity(intent)
            finish()
        }

        // Offline there is no account, so hide Logout rather than leave a button that can
        // only fail (it would just bounce off Splash and land straight back here).
        if (AppConfig.OFFLINE_MODE) {
            btnLogout.visibility = View.GONE
        }
    }

    private fun checkPendingUploads() {
        val pending = SessionStore.findPendingSessions(this)
            .filterNot { (_, dir, _) -> UploadProgress.isUploading(dir) }   // uploading in the background right now
        if (pending.isEmpty()) return
        val (sessionId, sessionDir, videoFile) = pending.first()
        com.google.android.material.dialog.MaterialAlertDialogBuilder(this)
            .setTitle("Unfinished Upload")
            .setMessage("A drive recording wasn't uploaded due to a connection issue. Upload it now?")
            .setPositiveButton("Upload") { _, _ ->
                startActivity(Intent(this, PostDriveActivity::class.java).apply {
                    putExtra("session_id", sessionId)
                    putExtra("session_dir", sessionDir.absolutePath)
                    putExtra("video_path", videoFile.absolutePath)
                })
            }
            .setNegativeButton("Skip") { _, _ -> SessionStore.markSkipped(sessionDir) }
            .show()
    }
}
