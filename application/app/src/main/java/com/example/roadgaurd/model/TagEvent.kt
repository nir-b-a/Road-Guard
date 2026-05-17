package com.example.roadgaurd.model

data class TagEvent(
    val timestamp: Long,
    val lat: Double,
    val lon: Double,
    val sessionId: String
)
