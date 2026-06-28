package com.example.roadgaurd.model

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Raw sensor/frame samples for one recording session.
 *
 * Every timestamp is nanoseconds on the SAME monotonic clock
 * (SystemClock.elapsedRealtimeNanos / camera SENSOR_TIMESTAMP mapped onto it).
 * Values are stored RAW — no sign correction, no unit conversion — exactly as
 * ANDROID_DATA_SPEC.md requires; Python does all fusion/integration.
 */
data class FrameSample(val frame: Int, val timestampNs: Long)
data class GyroSample(val timestampNs: Long, val gx: Float, val gy: Float, val gz: Float)
data class GravitySample(val timestampNs: Long, val grx: Float, val gry: Float, val grz: Float)
data class GpsSample(
    val timestampNs: Long,
    val lat: Double,
    val lon: Double,
    val speedMps: Float,
    val bearingDeg: Float,
    val accuracyM: Float
)
data class LinAccSample(val timestampNs: Long, val ax: Float, val ay: Float, val az: Float)

class RecordingSession(private val sessionId: String) {

    // Single lock guards every buffer. Samples arrive from several threads:
    // sensors (sensor thread), GPS (main looper), frames (camera analyzer thread).
    private val lock = Any()

    private val frames = mutableListOf<FrameSample>()
    private val gyro = mutableListOf<GyroSample>()
    private val gravity = mutableListOf<GravitySample>()
    private val gps = mutableListOf<GpsSample>()
    private val linacc = mutableListOf<LinAccSample>()
    private val tags = mutableListOf<TagEvent>()

    // Camera intrinsics in pixels for the recorded resolution; null until resolved.
    private var fx: Float? = null
    private var fy: Float? = null
    private var cx: Float? = null
    private var cy: Float? = null

    fun getSessionId(): String = sessionId

    // ── ingest (multi-threaded) ───────────────────────────────────────────────

    /** Frame index is assigned here so it is always 0-based and contiguous, matching
     *  the frame index main.py uses when it decodes the video. */
    fun addFrame(timestampNs: Long) = synchronized(lock) {
        frames.add(FrameSample(frames.size, timestampNs))
    }

    fun addGyro(timestampNs: Long, gx: Float, gy: Float, gz: Float) = synchronized(lock) {
        gyro.add(GyroSample(timestampNs, gx, gy, gz))
    }

    fun addGravity(timestampNs: Long, grx: Float, gry: Float, grz: Float) = synchronized(lock) {
        gravity.add(GravitySample(timestampNs, grx, gry, grz))
    }

    fun addGps(
        timestampNs: Long, lat: Double, lon: Double,
        speedMps: Float, bearingDeg: Float, accuracyM: Float
    ) = synchronized(lock) {
        gps.add(GpsSample(timestampNs, lat, lon, speedMps, bearingDeg, accuracyM))
    }

    fun addLinAcc(timestampNs: Long, ax: Float, ay: Float, az: Float) = synchronized(lock) {
        linacc.add(LinAccSample(timestampNs, ax, ay, az))
    }

    fun setIntrinsics(fx: Float, fy: Float, cx: Float, cy: Float) = synchronized(lock) {
        this.fx = fx; this.fy = fy; this.cx = cx; this.cy = cy
    }

    fun frameCount(): Int = synchronized(lock) { frames.size }

    // ── tags (legacy feature, NOT part of the speed spec) ─────────────────────

    fun tagEvent(lat: Double, lon: Double) = synchronized(lock) {
        tags.add(TagEvent(System.currentTimeMillis(), lat, lon, sessionId))
    }

    fun getTags(): List<TagEvent> = synchronized(lock) { tags.toList() }

    fun getTagsAsJson(): String = synchronized(lock) { tagsJsonLocked() }

    private fun tagsJsonLocked(): String {
        val arr = JSONArray()
        tags.forEach {
            arr.put(JSONObject().apply {
                put("timestamp", it.timestamp); put("lat", it.lat); put("lon", it.lon)
            })
        }
        return arr.toString()
    }

    // ── output ────────────────────────────────────────────────────────────────

    /**
     * Writes the per-session files into [dir] exactly per ANDROID_DATA_SPEC.md:
     * frames.csv, gyro.csv, gravity.csv, gps.csv (always — main.py auto-detects
     * Android mode by their presence), plus intrinsics.json when known and tags.json
     * (extra; not consumed by the pipeline).
     *
     * Floats are written via Kotlin/Java toString(), which is locale-independent and
     * always uses '.', so there is no locale-comma risk and no precision loss.
     */
    fun writeSessionFiles(dir: File) = synchronized(lock) {
        if (!dir.exists()) dir.mkdirs()

        val framesSb = StringBuilder("frame,timestamp_ns\n")
        frames.forEach { framesSb.append(it.frame).append(',').append(it.timestampNs).append('\n') }
        File(dir, "frames.csv").writeText(framesSb.toString())

        val gyroSb = StringBuilder("timestamp_ns,gx,gy,gz\n")
        gyro.forEach {
            gyroSb.append(it.timestampNs).append(',')
                .append(it.gx).append(',').append(it.gy).append(',').append(it.gz).append('\n')
        }
        File(dir, "gyro.csv").writeText(gyroSb.toString())

        val gravSb = StringBuilder("timestamp_ns,grx,gry,grz\n")
        gravity.forEach {
            gravSb.append(it.timestampNs).append(',')
                .append(it.grx).append(',').append(it.gry).append(',').append(it.grz).append('\n')
        }
        File(dir, "gravity.csv").writeText(gravSb.toString())

        val gpsSb = StringBuilder("timestamp_ns,lat,lon,speed_mps,bearing_deg,accuracy_m\n")
        gps.forEach {
            gpsSb.append(it.timestampNs).append(',')
                .append(it.lat).append(',').append(it.lon).append(',')
                .append(it.speedMps).append(',').append(it.bearingDeg).append(',')
                .append(it.accuracyM).append('\n')
        }
        File(dir, "gps.csv").writeText(gpsSb.toString())

        val linSb = StringBuilder("timestamp_ns,ax,ay,az\n")
        linacc.forEach {
            linSb.append(it.timestampNs).append(',')
                .append(it.ax).append(',').append(it.ay).append(',').append(it.az).append('\n')
        }
        File(dir, "linacc.csv").writeText(linSb.toString())

        val lfx = fx; val lfy = fy; val lcx = cx; val lcy = cy
        if (lfx != null && lfy != null && lcx != null && lcy != null) {
            val json = JSONObject().apply {
                put("fx", lfx.toDouble()); put("fy", lfy.toDouble())
                put("cx", lcx.toDouble()); put("cy", lcy.toDouble())
            }
            File(dir, "intrinsics.json").writeText(json.toString())
        } else {
            Log.w("RecordingSession", "intrinsics unknown — not writing intrinsics.json (pipeline falls back to FOV)")
        }

        File(dir, "tags.json").writeText(tagsJsonLocked())
    }
}
