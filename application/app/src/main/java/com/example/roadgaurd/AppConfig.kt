package com.example.roadgaurd

import android.content.Context

object AppConfig {
    private const val PREF_KEY_URL = "server_url"
    // Change this or override at runtime via Settings if the server runs on a different machine.
    private const val DEFAULT_URL = "https://repacking-tainted-unclasp.ngrok-free.dev/api"

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
