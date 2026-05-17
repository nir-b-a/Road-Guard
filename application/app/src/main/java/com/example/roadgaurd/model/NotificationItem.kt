package com.example.roadgaurd.model

data class NotificationItem(
    val message: String,
    val timestamp: String,
    var isRead: Boolean = false
)
