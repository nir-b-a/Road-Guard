package com.example.roadgaurd.model

import org.json.JSONArray
import org.json.JSONObject

class RecordingSession(private val sessionId: String) {

    private val tags = mutableListOf<TagEvent>()

    fun tagEvent(lat: Double, lon: Double) {
        tags.add(TagEvent(System.currentTimeMillis(), lat, lon, sessionId))
    }

    fun getTags(): List<TagEvent> = tags.toList()

    fun getSessionId(): String = sessionId

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
