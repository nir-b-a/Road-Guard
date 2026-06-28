package com.example.roadgaurd.ui

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.roadgaurd.R
import com.example.roadgaurd.storage.SessionStore
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.Response
import java.io.File
import java.io.IOException

class PostDriveActivity : AppCompatActivity() {

    private var sessionId: String? = null
    private var sessionDirPath: String? = null
    private var videoPath: String? = null
    private var tagsJson: String = "[]"
    private val BASE_URL = "http://10.0.2.2:5000/api"

    // Every file the collector writes into the per-session folder. These are the parts
    // the server WOULD receive once it is back up.
    private val DATA_FILES = listOf(
        "frames.csv", "gyro.csv", "gravity.csv", "gps.csv", "linacc.csv", "intrinsics.json", "tags.json"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_post_drive)

        sessionId = intent.getStringExtra("session_id") ?: "session_${System.currentTimeMillis()}"
        sessionDirPath = intent.getStringExtra("session_dir")
        videoPath = intent.getStringExtra("video_path")
        tagsJson = intent.getStringExtra("tags_json") ?: "[]"

        findViewById<Button>(R.id.btnNo).setOnClickListener { goHome() }
        findViewById<Button>(R.id.btnYes).setOnClickListener { uploadDrive() }
    }

    /**
     * Builds the FULL multipart upload — the video plus every captured data file plus the
     * tags — and sends it to the backend. On a successful response the session folder is
     * deleted from the device (the server now has the data). On any failure the folder is
     * kept so nothing is lost; a later launch-time sweep reclaims it once it is older than
     * SessionStore.SESSION_RETENTION_HOURS.
     */
    private fun uploadDrive() {
        val dir = sessionDirPath?.let { File(it) }
        if (dir == null || !dir.exists()) {
            Toast.makeText(this, "Session folder missing", Toast.LENGTH_LONG).show()
            return
        }

        val builder = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("sessionId", sessionId ?: "")
            .addFormDataPart("tags", tagsJson)

        // video
        videoPath?.let { vp ->
            val vf = File(vp)
            if (vf.exists()) {
                builder.addFormDataPart("video", vf.name, vf.asRequestBody("video/mp4".toMediaType()))
            }
        }

        // all captured sensor/frame/intrinsics files
        for (name in DATA_FILES) {
            val f = File(dir, name)
            if (f.exists()) {
                val mime = if (name.endsWith(".json")) "application/json" else "text/csv"
                builder.addFormDataPart(name, name, f.asRequestBody(mime.toMediaType()))
            }
        }

        val requestBody = builder.build()
        val prefs = getSharedPreferences("roadguard", MODE_PRIVATE)
        val token = prefs.getString("token", "") ?: ""
        val request = Request.Builder()
            .url("$BASE_URL/driver/upload")
            .addHeader("Authorization", "Bearer $token")
            .post(requestBody)
            .build()

        // Block a second submit while this one is in flight.
        val btnYes = findViewById<Button>(R.id.btnYes)
        btnYes.isEnabled = false
        Toast.makeText(this, "Uploading drive…", Toast.LENGTH_SHORT).show()

        OkHttpClient().newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.w("PostDrive", "Upload failed: ${e.message}")
                runOnUiThread {
                    btnYes.isEnabled = true
                    // Keep the session on disk; the launch-time sweep removes it after X hours.
                    Toast.makeText(this@PostDriveActivity,
                        "Upload failed: ${e.message}. Saved on device — will retry/clean up later.",
                        Toast.LENGTH_LONG).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val ok = response.use { it.isSuccessful }
                runOnUiThread {
                    if (ok) {
                        // Server has the full session now — reclaim the device storage.
                        SessionStore.deleteSession(dir)
                        Toast.makeText(this@PostDriveActivity, "Drive uploaded!", Toast.LENGTH_LONG).show()
                        goHome()
                    } else {
                        btnYes.isEnabled = true
                        Toast.makeText(this@PostDriveActivity,
                            "Server error: ${response.code}. Saved on device — will clean up later.",
                            Toast.LENGTH_LONG).show()
                    }
                }
            }
        })
    }

    private fun goHome() {
        startActivity(Intent(this, HomeActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
        })
        finish()
    }
}
