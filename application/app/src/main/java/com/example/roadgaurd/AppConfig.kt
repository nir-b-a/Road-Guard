package com.example.roadgaurd

import android.content.Context

object AppConfig {
    /**
     * OFFLINE MODE — run the whole app with no server at all. One switch; flip and rebuild.
     *
     * true  → login/register is skipped (straight to Home), notifications and logout are
     *         hidden, drives are NEVER uploaded and NEVER auto-deleted: every session stays
     *         in Android/data/com.example.roadgaurd/files/sessions/ until you pull it off
     *         the phone (or delete it yourself). Storage is therefore your responsibility.
     * false → normal online build: login required, upload to R2, 24 h retention sweep.
     *
     * Nothing else in the app reads the network at startup, so this is the only gate needed.
     */
    const val OFFLINE_MODE = false

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
