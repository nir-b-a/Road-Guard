package com.example.roadgaurd.ui

import android.app.ProgressDialog
import android.content.Intent
import android.database.Cursor
import android.net.Uri
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
import java.io.File
import java.io.IOException

class PostDriveActivity : AppCompatActivity() {

    private var recordingName: String? = null
    private var sessionId: String? = null
    private var tagsJson: String = "[]"
    private val BASE_URL = "http://10.0.2.2:5000/api"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_post_drive)

        recordingName = intent.getStringExtra("recording_name")
        sessionId = intent.getStringExtra("session_id") ?: "session_${System.currentTimeMillis()}"
        tagsJson = intent.getStringExtra("tags_json") ?: "[]"

        findViewById<Button>(R.id.btnNo).setOnClickListener { goHome() }
        findViewById<Button>(R.id.btnYes).setOnClickListener { uploadDrive() }
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
