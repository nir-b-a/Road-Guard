package com.example.roadgaurd.ui

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.roadgaurd.AppConfig
import com.example.roadgaurd.R
import com.example.roadgaurd.storage.SessionStore
import okhttp3.*
import org.json.JSONObject
import java.io.IOException

class HomeActivity : AppCompatActivity() {

    private var unreadCount = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)

        // Reclaim un-uploaded sessions left on the device beyond the retention window.
        SessionStore.sweepStaleSessions(this)

        fetchNotifications()

        findViewById<android.widget.Button>(R.id.btnStartDrive).setOnClickListener {
            startActivity(Intent(this, RecordingActivity::class.java))
        }

        findViewById<android.widget.Button>(R.id.btnNotifications).setOnClickListener {
            unreadCount = 0
            updateBadge()
            startActivity(Intent(this, NotificationsActivity::class.java))
        }

        findViewById<android.widget.Button>(R.id.btnSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        findViewById<android.widget.Button>(R.id.btnLogout).setOnClickListener {
            val prefs = getSharedPreferences("roadguard", MODE_PRIVATE)
            prefs.edit().clear().apply()

            val intent = Intent(this, SplashActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            startActivity(intent)
            finish()
        }
    }

    private fun fetchNotifications() {
        val prefs = getSharedPreferences("roadguard", MODE_PRIVATE)
        val token = prefs.getString("token", null) ?: return
        val userId = prefs.getString("userId", null) ?: return

        val request = Request.Builder()
            .url("${AppConfig.getBaseUrl(this)}/notifications/$userId")
            .addHeader("Authorization", "Bearer $token")
            .addHeader("ngrok-skip-browser-warning", "true")
            .build()

        OkHttpClient().newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {}
            override fun onResponse(call: Call, response: Response) {
                val body = response.body?.string() ?: return
                runOnUiThread {
                    try {
                        val data = JSONObject(body).getJSONArray("data")
                        var unread = 0
                        for (i in 0 until data.length()) {
                            if (!data.getJSONObject(i).getBoolean("isRead")) unread++
                        }
                        unreadCount = unread
                        updateBadge()
                    } catch (_: Exception) {}
                }
            }
        })
    }

    private fun updateBadge() {
        val badge = findViewById<TextView>(R.id.tvBadge)
        badge.text = unreadCount.toString()
        badge.visibility = if (unreadCount > 0) View.VISIBLE else View.GONE
    }
}
