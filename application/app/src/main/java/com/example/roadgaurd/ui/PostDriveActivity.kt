package com.example.roadgaurd.ui

import android.app.ProgressDialog
import android.content.Intent
import android.database.Cursor
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.roadgaurd.R
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONArray
import java.io.File
import java.io.IOException

class PostDriveActivity : AppCompatActivity() {

    private var recordingName: String? = null
    private var sessionId: String? = null
    private var tagsJson: String = "[]"
    private var speedJson: String = "[]"
    private val BASE_URL = "http://10.0.2.2:5000/api"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_post_drive)

        recordingName = intent.getStringExtra("recording_name")
        sessionId = intent.getStringExtra("session_id") ?: "session_${System.currentTimeMillis()}"
        tagsJson = intent.getStringExtra("tags_json") ?: "[]"
        speedJson = intent.getStringExtra("speed_json") ?: "[]"

        findViewById<Button>(R.id.btnNo).setOnClickListener { goHome() }
        findViewById<Button>(R.id.btnYes).setOnClickListener { uploadDrive() }
    }

    private fun generateSpeedReport(speedJson: String, fps: Int = 30) {
        try {
            val samples = JSONArray(speedJson)
            if (samples.length() == 0) {
                Log.w("PostDrive", "No speed samples — skipping report")
                return
            }

            val lastTimestampMs = samples.getJSONObject(samples.length() - 1).getLong("timestampMs")
            val totalFrames = ((lastTimestampMs / 1000.0) * fps).toInt()
            val msPerFrame = 1000.0 / fps

            val sb = StringBuilder()
            sb.appendLine("RoadGuard Speed Report")
            sb.appendLine("Session: $sessionId")
            sb.appendLine("FPS: $fps  |  Total Frames: $totalFrames")
            sb.appendLine("=".repeat(65))
            sb.appendLine(String.format("%-12s | %-15s | %-12s | %s", "Frame", "Timestamp", "Speed", "Location"))
            sb.appendLine("-".repeat(65))

            for (frameIndex in 0..totalFrames) {
                val frameTimeMs = (frameIndex * msPerFrame).toLong()

                var lower = samples.getJSONObject(0)
                var upper = samples.getJSONObject(0)
                for (i in 0 until samples.length()) {
                    val s = samples.getJSONObject(i)
                    if (s.getLong("timestampMs") <= frameTimeMs) lower = s
                    if (s.getLong("timestampMs") >= frameTimeMs) { upper = s; break }
                }

                val lowerT = lower.getLong("timestampMs")
                val upperT = upper.getLong("timestampMs")
                val lowerSpeed = lower.getDouble("speedKmh").toFloat()
                val upperSpeed = upper.getDouble("speedKmh").toFloat()
                val lowerLat = lower.getDouble("lat")
                val lowerLon = lower.getDouble("lon")

                val interpolatedSpeed = if (upperT == lowerT) lowerSpeed
                else lowerSpeed + (upperSpeed - lowerSpeed) * ((frameTimeMs - lowerT).toFloat() / (upperT - lowerT))

                val totalSec = frameTimeMs / 1000
                val ms = frameTimeMs % 1000
                val h = totalSec / 3600
                val m = (totalSec % 3600) / 60
                val s = totalSec % 60
                val timestamp = String.format("%02d:%02d:%02d.%03d", h, m, s, ms)

                sb.appendLine(String.format(
                    "%-12d | %-15s | %6.1f km/h | lat=%.4f lon=%.4f",
                    frameIndex, timestamp, interpolatedSpeed, lowerLat, lowerLon
                ))
            }

            val outFile = File(getExternalFilesDir(null), "speed_report_${sessionId}.txt")
            outFile.writeText(sb.toString())
            Log.i("PostDrive", "Speed report saved: ${outFile.absolutePath}")

        } catch (e: Exception) {
            Log.e("PostDrive", "Speed report generation failed: ${e.message}")
        }
    }

    @Suppress("DEPRECATION")
    private fun uploadDrive() {
        val name = recordingName ?: run {
            Toast.makeText(this, "No recording found", Toast.LENGTH_SHORT).show()
            return
        }
        val file = getFileFromMediaStore(name) ?: run {
            Toast.makeText(this, "Could not find video file", Toast.LENGTH_SHORT).show()
            return
        }

        generateSpeedReport(speedJson)

        val prefs = getSharedPreferences("roadguard", MODE_PRIVATE)
        val token = prefs.getString("token", "") ?: ""

        val dialog = ProgressDialog(this).apply {
            setMessage("Uploading drive...")
            isIndeterminate = true
            setCancelable(false)
            show()
        }

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("sessionId", sessionId!!)
            .addFormDataPart("tags", tagsJson)
            .addFormDataPart("video", file.name, file.asRequestBody("video/mp4".toMediaType()))
            .build()

        val request = Request.Builder()
            .url("$BASE_URL/driver/upload")
            .addHeader("Authorization", "Bearer $token")
            .post(requestBody)
            .build()

        OkHttpClient().newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    dialog.dismiss()
                    Toast.makeText(this@PostDriveActivity, "Upload failed: ${e.message}", Toast.LENGTH_LONG).show()
                    Log.e("PostDrive", "Upload error: ${e.message}")
                }
            }
            override fun onResponse(call: Call, response: Response) {
                runOnUiThread {
                    dialog.dismiss()
                    if (response.isSuccessful) {
                        Toast.makeText(this@PostDriveActivity, "Drive uploaded!", Toast.LENGTH_LONG).show()
                        goHome()
                    } else {
                        Toast.makeText(this@PostDriveActivity, "Server error: ${response.code}", Toast.LENGTH_LONG).show()
                        Log.e("PostDrive", response.body?.string() ?: "")
                    }
                }
            }
        })
    }

    private fun getFileFromMediaStore(displayName: String): File? {
        val projection = arrayOf(MediaStore.Video.Media.DATA)
        val selection = "${MediaStore.Video.Media.DISPLAY_NAME} = ?"
        val cursor: Cursor? = contentResolver.query(
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI, projection, selection, arrayOf(displayName), null)
        cursor?.use {
            if (it.moveToFirst()) {
                val path = it.getString(it.getColumnIndexOrThrow(MediaStore.Video.Media.DATA))
                return File(path)
            }
        }
        return null
    }

    private fun goHome() {
        startActivity(Intent(this, HomeActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP
        })
        finish()
    }
}
