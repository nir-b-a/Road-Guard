package com.example.roadgaurd.ui

import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.os.Bundle
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.example.roadgaurd.R
import com.example.roadgaurd.model.RecordingSession
import com.google.android.gms.location.*

class RecordingActivity : AppCompatActivity() {

    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private lateinit var session: RecordingSession
    private var savedRecordingName: String? = null
    private lateinit var fusedLocationClient: FusedLocationProviderClient
    private var lastLocation: Location? = null
    private var recordingStartTime: Long = 0

    private val permissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.ACCESS_FINE_LOCATION
    )

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        if (results.all { it.value }) { startCamera(); startLocationUpdates() }
        else { Toast.makeText(this, "Permissions required.", Toast.LENGTH_LONG).show(); finish() }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_recording)

        session = RecordingSession("session_${System.currentTimeMillis()}")
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        if (allPermissionsGranted()) { startCamera(); startLocationUpdates() }
        else permissionLauncher.launch(permissions)

        findViewById<Button>(R.id.btnTag).setOnClickListener {
            val loc = lastLocation
            if (loc != null) {
                session.tagEvent(lat = loc.latitude, lon = loc.longitude)
                Toast.makeText(this, "Tag saved! (${session.getTags().size} total)", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Waiting for GPS...", Toast.LENGTH_SHORT).show()
            }
        }

        findViewById<Button>(R.id.btnStop).setOnClickListener {
            stopRecordingAndProceed()
        }
    }

    private fun startLocationUpdates() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) return

        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000).build()
        fusedLocationClient.requestLocationUpdates(request, object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val location = result.lastLocation ?: return
                lastLocation = location
                if (recordingStartTime > 0) {
                    val speedKmh = location.speed * 3.6f
                    val elapsedMs = System.currentTimeMillis() - recordingStartTime
                    session.addSpeedSample(elapsedMs, speedKmh, location.latitude, location.longitude)
                }
            }
        }, Looper.getMainLooper())
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(findViewById<PreviewView>(R.id.previewView).surfaceProvider)
            }
            val recorder = Recorder.Builder().setQualitySelector(QualitySelector.from(Quality.HD)).build()
            videoCapture = VideoCapture.withOutput(recorder)
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, videoCapture!!)
                beginRecording()
            } catch (e: Exception) { Log.e("RecordingActivity", "Camera bind failed: ${e.message}") }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun beginRecording() {
        val fileName = "roadguard_${System.currentTimeMillis()}.mp4"
        savedRecordingName = fileName
        val contentValues = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
        }
        val outputOptions = MediaStoreOutputOptions
            .Builder(contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            .setContentValues(contentValues).build()

        recording = videoCapture!!.output.prepareRecording(this, outputOptions)
            .apply {
                if (ContextCompat.checkSelfPermission(this@RecordingActivity, Manifest.permission.RECORD_AUDIO)
                    == PackageManager.PERMISSION_GRANTED) withAudioEnabled()
            }
            .start(ContextCompat.getMainExecutor(this)) { event ->
                when (event) {
                    is VideoRecordEvent.Start -> runOnUiThread {
                        recordingStartTime = System.currentTimeMillis()
                        findViewById<TextView>(R.id.tvRecIndicator).visibility = View.VISIBLE
                    }
                    is VideoRecordEvent.Finalize -> {
                        if (event.hasError()) Log.e("RecordingActivity", "Error: ${event.error}")
                    }
                }
            }
    }

    private fun stopRecordingAndProceed() {
        recording?.stop()
        recording = null
        startActivity(Intent(this, PostDriveActivity::class.java).apply {
            putExtra("recording_name", savedRecordingName)
            putExtra("session_id", session.getSessionId())
            putExtra("tags_json", session.getTagsAsJson())
            putExtra("speed_json", session.getSpeedSamplesAsJson())
        })
        finish()
    }

    override fun onDestroy() { super.onDestroy(); recording?.stop() }

    private fun allPermissionsGranted() = permissions.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }
}
