package com.example.roadgaurd

import android.content.Context

object AppConfig {
    private const val PREF_KEY_URL = "server_url"
    private const val DEFAULT_URL = "http://192.168.1.106:5000/api"

    fun getBaseUrl(context: Context): String {
        val prefs = context.getSharedPreferences("roadguard", Context.MODE_PRIVATE)
        return prefs.getString(PREF_KEY_URL, DEFAULT_URL) ?: DEFAULT_URL
    }

    fun setBaseUrl(context: Context, url: String) {
        context.getSharedPreferences("roadguard", Context.MODE_PRIVATE)
            .edit()
            .putString(PREF_KEY_URL, url)
            .apply()
    }
}
