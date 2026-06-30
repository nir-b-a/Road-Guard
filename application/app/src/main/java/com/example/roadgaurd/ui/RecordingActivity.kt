package com.example.roadgaurd.ui

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.os.Build
import android.os.Bundle
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.util.Rational
import android.view.Surface
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.example.roadgaurd.R
import com.example.roadgaurd.model.RecordingSession
import com.google.android.gms.location.*
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class RecordingActivity : AppCompatActivity() {

    private companion object {
        const val TAG = "RecordingActivity"
        const val VIDEO_WIDTH = 1920
        const val VIDEO_HEIGHT = 1080
        const val TARGET_FPS = 30
        // Video encoding bitrate. The device default for FHD is ~17 Mbps (124 MB/min).
        // 12 Mbps is ~30% smaller (~87 MB/min) with near-original quality. Lower it to
        // 10_000_000 / 8_000_000 for smaller files (slightly softer, fine for detection).
        const val VIDEO_BITRATE = 12_000_000

        // Calibrated intrinsics for the 1920x1080 stream, measured by
        // tools/calibrate_intrinsics.py on a checkerboard clip recorded with THIS app.
        // When all four are set, they OVERRIDE the metadata estimate and remove the
        // full-sensor-width assumption (gold standard). Leave null until calibrated.
        val CALIB_FX: Float? = null
        val CALIB_FY: Float? = null
        val CALIB_CX: Float? = null
        val CALIB_CY: Float? = null

        // On-device lens-distortion correction (API 28+). When true we ask the camera HAL
        // to rectify its own frames (HIGH_QUALITY if the device supports it, else FAST), so
        // the recorded video is ~pinhole edge-to-edge and the Python pipeline needs no
        // undistort. Set false to record raw (and undistort offline instead). HIGH_QUALITY
        // MAY cost a little throughput on some devices; flip to FAST-only by setting this
        // false-ish if you ever see the 30 fps pin slip (FAST is guaranteed not to slow it).
        const val DISTORTION_CORRECTION_PREFER_HIGH_QUALITY = true
    }

    private var videoCapture: VideoCapture<Recorder>? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var recording: Recording? = null

    private lateinit var session: RecordingSession
    private lateinit var sessionDir: File
    private var savedVideoPath: String? = null

    private lateinit var fusedLocationClient: FusedLocationProviderClient

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var sensorManager: SensorManager
    private var gyroSensor: Sensor? = null
    private var gravitySensor: Sensor? = null
    private var linAccSensor: Sensor? = null

    // Stream logging is gated on the actual recording window so frames.csv index 0
    // lines up with the first encoded video frame.
    @Volatile private var isCapturing = false

    // Mapping of the camera frame presentation timestamp onto the shared
    // elapsedRealtimeNanos (BOOTTIME) clock used by every other stream.
    private var frameTsIsRealtime = true
    private var frameTsOffsetNs = 0L

    // Resolved once from the camera's supported modes; applied to the capture request in
    // startCamera(). null -> unsupported (older API / device), leave the capture default.
    private var distortionCorrectionMode: Int? = null

    private val permissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.ACCESS_FINE_LOCATION
    )

    private val sensorListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            if (!isCapturing) return
            // event.timestamp is ns on elapsedRealtimeNanos (BOOTTIME) on modern devices.
            // Logged raw — no sign correction here (resolved in Python).
            when (event.sensor.type) {
                Sensor.TYPE_GYROSCOPE ->
                    session.addGyro(event.timestamp, event.values[0], event.values[1], event.values[2])
                Sensor.TYPE_GRAVITY, Sensor.TYPE_ACCELEROMETER ->
                    session.addGravity(event.timestamp, event.values[0], event.values[1], event.values[2])
                Sensor.TYPE_LINEAR_ACCELERATION ->
                    session.addLinAcc(event.timestamp, event.values[0], event.values[1], event.values[2])
            }
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

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
        sessionDir = File(getExternalFilesDir(null), "sessions/${session.getSessionId()}").apply { mkdirs() }

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        // Prefer TYPE_GRAVITY (already low-pass fused); fall back to raw accelerometer.
        gravitySensor = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        // Gravity-removed device-frame accel -> linacc.csv, feeds the accel+GPS speed fusion.
        linAccSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        if (gyroSensor == null) Log.w(TAG, "No gyroscope present — yaw reconstruction unavailable")
        if (gravitySensor == null) Log.w(TAG, "No gravity/accelerometer present")
        if (linAccSensor == null) Log.w(TAG, "No linear-acceleration sensor — speed will be GPS-only")

        configureCameraClockAndIntrinsics()

        if (allPermissionsGranted()) { startCamera(); startLocationUpdates() }
        else permissionLauncher.launch(permissions)

        findViewById<Button>(R.id.btnStop).setOnClickListener { stopRecordingAndProceed() }
    }

    /**
     * Reads the back camera's metadata once: (1) its SENSOR_INFO_TIMESTAMP_SOURCE so we
     * can map frame timestamps onto elapsedRealtimeNanos, and (2) focal length + physical
     * sensor size to derive intrinsics for the 1920x1080 output.
     */
    private fun configureCameraClockAndIntrinsics() {
        try {
            val cm = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val id = cm.cameraIdList.firstOrNull {
                cm.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) ==
                    CameraCharacteristics.LENS_FACING_BACK
            } ?: cm.cameraIdList.firstOrNull() ?: return
            val ch = cm.getCameraCharacteristics(id)

            // ── clock source for frame timestamps ──
            val src = ch.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
            frameTsIsRealtime = src == CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
            if (!frameTsIsRealtime) {
                // Camera PTS is on CLOCK_MONOTONIC (== System.nanoTime). Convert to BOOTTIME
                // with a one-time offset (the two clocks differ only by a constant while awake).
                val t0 = System.nanoTime()
                val r = SystemClock.elapsedRealtimeNanos()
                val t1 = System.nanoTime()
                frameTsOffsetNs = r - (t0 + t1) / 2
            }
            Log.i(TAG, "frame ts source=${if (frameTsIsRealtime) "REALTIME" else "UNKNOWN (+$frameTsOffsetNs ns)"}")

            // ── on-device lens-distortion correction mode (applied in startCamera) ──
            distortionCorrectionMode = resolveDistortionCorrectionMode(ch)

            // ── intrinsics for 1920x1080 (priority: checkerboard CALIB_* >
            //    device LENS_INTRINSIC_CALIBRATION > focal/sensor-width estimate) ──
            val cfx = CALIB_FX; val cfy = CALIB_FY; val ccx = CALIB_CX; val ccy = CALIB_CY
            if (cfx != null && cfy != null && ccx != null && ccy != null) {
                // Gold standard: measured by tools/calibrate_intrinsics.py for this exact
                // 1080p stream — no full-sensor-width assumption.
                session.setIntrinsics(cfx, cfy, ccx, ccy)
                Log.i(TAG, "intrinsics: using CALIBRATED fx=$cfx fy=$cfy cx=$ccx cy=$ccy")
            } else if (!setIntrinsicsFromLensCalibration(ch)) {
                // Last-resort ESTIMATE from focal length + physical sensor size.
                val focal = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
                val sensorSize = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                if (focal != null && focal > 0f && sensorSize != null && sensorSize.width > 0f) {
                    // Assumes the 16:9 video uses the full sensor width with square pixels
                    // (fx == fy) — the assumption calibration removes.
                    val fx = focal / sensorSize.width * VIDEO_WIDTH
                    session.setIntrinsics(fx, fx, VIDEO_WIDTH / 2f, VIDEO_HEIGHT / 2f)
                    Log.i(TAG, "intrinsics (ESTIMATE) fx=fy=$fx cx=${VIDEO_WIDTH / 2f} cy=${VIDEO_HEIGHT / 2f}")
                } else {
                    Log.w(TAG, "intrinsics metadata unavailable; pipeline will fall back to FOV")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "camera clock/intrinsics config failed: ${e.message}")
        }
    }

    /**
     * Device factory intrinsics from CameraCharacteristics.LENS_INTRINSIC_CALIBRATION
     * ([fx, fy, cx, cy, skew] in pixels of the PRE-CORRECTION active array), rescaled to
     * the 1920x1080 output. Assumes the 16:9 video uses the full sensor width, center-
     * cropped in height, then scaled (matches the ViewPort config) with square pixels.
     * Returns true when valid intrinsics were applied; false (null / zeros / no array
     * size) so the caller falls back to the focal/sensor-width estimate.
     */
    private fun setIntrinsicsFromLensCalibration(ch: CameraCharacteristics): Boolean {
        val calib = ch.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION) ?: return false
        if (calib.size < 4) return false
        val fxA = calib[0]; val fyA = calib[1]; val cxA = calib[2]; val cyA = calib[3]
        if (fxA <= 0f || fyA <= 0f) return false   // present but un-calibrated (zeros)

        val arr = ch.get(CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE)
            ?: ch.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            ?: return false
        val wA = arr.width().toFloat(); val hA = arr.height().toFloat()
        if (wA <= 0f || hA <= 0f) return false

        // full-width, center-cropped-to-16:9, scaled to 1920x1080; square pixels => sx == sy.
        val sx = VIDEO_WIDTH / wA
        val cropTop = (hA - wA * VIDEO_HEIGHT / VIDEO_WIDTH) / 2f
        val fxOut = fxA * sx
        val fyOut = fyA * sx
        val cxOut = cxA * sx
        val cyOut = (cyA - cropTop) * sx
        session.setIntrinsics(fxOut, fyOut, cxOut, cyOut)
        Log.i(TAG, "intrinsics: LENS_INTRINSIC_CALIBRATION scaled (array ${arr.width()}x${arr.height()}) " +
            "-> fx=$fxOut fy=$fyOut cx=$cxOut cy=$cyOut")
        return true
    }

    /**
     * Picks the best on-device lens-distortion-correction mode the camera supports so the
     * HAL rectifies its own frames: HIGH_QUALITY (preferred) > FAST > none. API 28+ only;
     * returns null when unavailable so the caller leaves the capture default untouched.
     *
     * NOTE: enabling this changes the effective intrinsics — the recorded stream becomes
     * (near) pinhole, so the checkerboard CALIB_* values measured with correction OFF no
     * longer apply. Recalibrate with this ON (the new report's k1..k3 should collapse to
     * ~0, which is also how you VERIFY it works), and do NOT also run the Python
     * --undistort on clips recorded this way (that double-corrects).
     */
    private fun resolveDistortionCorrectionMode(ch: CameraCharacteristics): Int? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) return null
        val modes = ch.get(CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES) ?: return null
        val has = { m: Int -> modes.any { it == m } }
        val hq = CameraMetadata.DISTORTION_CORRECTION_MODE_HIGH_QUALITY
        val fast = CameraMetadata.DISTORTION_CORRECTION_MODE_FAST
        val chosen = when {
            DISTORTION_CORRECTION_PREFER_HIGH_QUALITY && has(hq) -> hq
            has(fast) -> fast
            has(hq) -> hq
            else -> null
        }
        Log.i(TAG, "distortion correction: ${distortionModeName(chosen)} " +
            "(available=${modes.joinToString { distortionModeName(it) }})")
        return chosen
    }

    private fun distortionModeName(mode: Int?): String = when (mode) {
        null -> "none/unsupported"
        CameraMetadata.DISTORTION_CORRECTION_MODE_OFF -> "OFF"
        CameraMetadata.DISTORTION_CORRECTION_MODE_FAST -> "FAST"
        CameraMetadata.DISTORTION_CORRECTION_MODE_HIGH_QUALITY -> "HIGH_QUALITY"
        else -> "mode$mode"
    }

    private fun startLocationUpdates() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) return

        // Request ~5 Hz; the device delivers what it can. Python interpolates to frames.
        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 200)
            .setMinUpdateIntervalMillis(200)
            .build()
        fusedLocationClient.requestLocationUpdates(request, object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val location = result.lastLocation ?: return
                if (isCapturing) {
                    session.addGps(
                        timestampNs = location.elapsedRealtimeNanos, // already on the shared clock
                        lat = location.latitude,
                        lon = location.longitude,
                        speedMps = location.speed,    // raw m/s — no km/h conversion
                        bearingDeg = location.bearing,
                        accuracyM = location.accuracy
                    )
                }
            }
        }, Looper.getMainLooper())
    }

    @androidx.annotation.OptIn(markerClass = [ExperimentalCamera2Interop::class])
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // 16:9 on every use case + a 16:9 ViewPort (at bind, below) pushes CameraX toward
            // a full-sensor-width readout (cropping height only) -> widest FOV, like the stock
            // camera app. This also makes intrinsics.json's full-width fx assumption valid.
            val ratio16x9 = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .build()

            val previewBuilder = Preview.Builder().setResolutionSelector(ratio16x9)
            // Pin the capture session to 30 fps via the AE target range.
            val previewExtender = Camera2Interop.Extender(previewBuilder)
                .setCaptureRequestOption(
                    CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, Range(TARGET_FPS, TARGET_FPS)
                )
            // Turn on the HAL's own lens-distortion correction. Set here (on preview) because
            // CameraX merges every use case's Camera2Interop options into the ONE repeating
            // request shared by the whole session — same mechanism the 30 fps pin rides — so
            // it also covers the recorded video and the ImageAnalysis frames.
            distortionCorrectionMode?.let { mode ->
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    previewExtender.setCaptureRequestOption(
                        CaptureRequest.DISTORTION_CORRECTION_MODE, mode
                    )
                }
            }
            val preview = previewBuilder.build().also {
                it.setSurfaceProvider(findViewById<PreviewView>(R.id.previewView).surfaceProvider)
            }

            // 1080p video at a capped bitrate (smaller files, near-original quality).
            val recorder = Recorder.Builder()
                .setQualitySelector(
                    QualitySelector.from(Quality.FHD, FallbackStrategy.higherQualityOrLowerThan(Quality.FHD))
                )
                .setTargetVideoEncodingBitRate(VIDEO_BITRATE)
                .build()
            videoCapture = VideoCapture.withOutput(recorder)

            // ImageAnalysis exists only to emit one timestamp per camera frame -> frames.csv.
            // KEEP_ONLY_LATEST + a trivial analyzer (read ts, close) means it keeps up at 30 fps
            // so the logged frames track the recorded frames closely. 16:9 so it shares the
            // video's wide FOV sensor mode instead of dragging in a 4:3 (zoomed) crop.
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(ratio16x9)
                .build()
            analysis.setAnalyzer(cameraExecutor) { image ->
                if (isCapturing) {
                    val ts = if (frameTsIsRealtime) image.imageInfo.timestamp
                             else image.imageInfo.timestamp + frameTsOffsetNs
                    session.addFrame(ts)
                }
                image.close()
            }
            imageAnalysis = analysis

            try {
                cameraProvider.unbindAll()
                // A 16:9 ViewPort with FILL_CENTER keeps the full sensor width (crops height),
                // so all three use cases share the widest common FOV (zoom ratio stays 1.0).
                val rotation = findViewById<PreviewView>(R.id.previewView).display?.rotation
                    ?: Surface.ROTATION_0
                val viewPort = ViewPort.Builder(Rational(VIDEO_WIDTH, VIDEO_HEIGHT), rotation)
                    .setScaleType(ViewPort.FILL_CENTER)
                    .build()
                val group = UseCaseGroup.Builder()
                    .addUseCase(preview)
                    .addUseCase(videoCapture!!)
                    .addUseCase(analysis)
                    .setViewPort(viewPort)
                    .build()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, group)
                beginRecording()
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed: ${e.message}")
                Toast.makeText(this, "Camera bind failed: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun beginRecording() {
        // Record straight into the per-session folder so the video sits next to its CSVs.
        val videoFile = File(sessionDir, "roadguard_${System.currentTimeMillis()}.mp4")
        savedVideoPath = videoFile.absolutePath
        val outputOptions = FileOutputOptions.Builder(videoFile).build()

        recording = videoCapture!!.output.prepareRecording(this, outputOptions)
            .apply {
                if (ContextCompat.checkSelfPermission(this@RecordingActivity, Manifest.permission.RECORD_AUDIO)
                    == PackageManager.PERMISSION_GRANTED) withAudioEnabled()
            }
            .start(ContextCompat.getMainExecutor(this)) { event ->
                when (event) {
                    is VideoRecordEvent.Start -> runOnUiThread {
                        // Keep screen + CPU awake for the whole recording so the camera
                        // frames and GPS don't stall mid-drive (released on stop/destroy).
                        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                        startSensors()
                        isCapturing = true   // open the logging window
                        findViewById<TextView>(R.id.tvRecIndicator).visibility = View.VISIBLE
                    }
                    is VideoRecordEvent.Finalize ->
                        if (event.hasError()) Log.e(TAG, "Recording error: ${event.error}")
                }
            }
    }

    private fun startSensors() {
        // Wrapped so a sensor-rate SecurityException (e.g. missing HIGH_SAMPLING_RATE_SENSORS
        // on some OEM) degrades to fewer samples instead of crashing the recording.
        gyroSensor?.let { register(it, SensorManager.SENSOR_DELAY_FASTEST) }
        gravitySensor?.let { register(it, SensorManager.SENSOR_DELAY_GAME) } // ~50 Hz
        linAccSensor?.let { register(it, SensorManager.SENSOR_DELAY_GAME) }  // ~50 Hz
    }

    private fun register(sensor: Sensor, delayUs: Int) {
        try {
            sensorManager.registerListener(sensorListener, sensor, delayUs)
        } catch (e: SecurityException) {
            // Retry at a permission-free rate (<=200 Hz) so we still get some data.
            Log.w(TAG, "high-rate registration for ${sensor.stringType} denied: ${e.message}; retrying at 100 Hz")
            try {
                sensorManager.registerListener(sensorListener, sensor, 10_000) // 100 Hz
            } catch (e2: Exception) {
                Log.e(TAG, "sensor ${sensor.stringType} registration failed: ${e2.message}")
            }
        }
    }

    private fun stopRecordingAndProceed() {
        isCapturing = false                         // close the logging window first
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        sensorManager.unregisterListener(sensorListener)
        recording?.stop()
        recording = null

        // Buffers are complete now; write the per-session files (frames/gyro/gravity/gps/intrinsics).
        try {
            session.writeSessionFiles(sessionDir)
            Log.i(TAG, "Wrote ${session.frameCount()} frames + sensor CSVs to ${sessionDir.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed writing session files: ${e.message}")
        }

        startActivity(Intent(this, PostDriveActivity::class.java).apply {
            putExtra("session_id", session.getSessionId())
            putExtra("session_dir", sessionDir.absolutePath)
            putExtra("video_path", savedVideoPath)
        })
        finish()
    }

    override fun onDestroy() {
        super.onDestroy()
        isCapturing = false
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        sensorManager.unregisterListener(sensorListener)
        recording?.stop()
        cameraExecutor.shutdown()
    }

    private fun allPermissionsGranted() = permissions.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }
}
