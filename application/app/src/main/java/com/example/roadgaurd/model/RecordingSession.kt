package com.example.roadgaurd.model

import org.json.JSONArray
import org.json.JSONObject

data class SpeedSample(
    val timestampMs: Long,
    val speedKmh: Float,
    val lat: Double,
    val lon: Double
)

class RecordingSession(private val sessionId: String) {

    private val tags = mutableListOf<TagEvent>()
    private val speedSamples = mutableListOf<SpeedSample>()

    fun tagEvent(lat: Double, lon: Double) {
        tags.add(TagEvent(System.currentTimeMillis(), lat, lon, sessionId))
    }

    fun getTags(): List<TagEvent> = tags.toList()

    fun getSessionId(): String = sessionId

    fun addSpeedSample(timestampMs: Long, speedKmh: Float, lat: Double, lon: Double) {
        speedSamples.add(SpeedSample(timestampMs, speedKmh, lat, lon))
    }

    fun getSpeedSamples(): List<SpeedSample> = speedSamples.toList()

    fun getSpeedSamplesAsJson(): String {
        val arr = JSONArray()
        speedSamples.forEach { sample ->
            arr.put(JSONObject().apply {
                put("timestampMs", sample.timestampMs)
                put("speedKmh", sample.speedKmh)
                put("lat", sample.lat)
                put("lon", sample.lon)
            })
        }
        return arr.toString()
    }

    fun getTagsAsJson(): String {
        val arr = JSONArray()
        tags.forEach { tag ->
            arr.put(JSONObject().apply {
                put("timestamp", tag.timestamp)
                put("lat", tag.lat)
                put("lon", tag.lon)
            })
        }
        return arr.toString()
    }
}
