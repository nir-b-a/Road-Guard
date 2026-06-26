package com.example.roadgaurd.ui

import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.roadgaurd.R
import com.example.roadgaurd.adapter.NotificationAdapter
import com.example.roadgaurd.model.NotificationItem
import okhttp3.*
import org.json.JSONObject
import java.io.IOException

class NotificationsActivity : AppCompatActivity() {

    private val BASE_URL = "http://10.100.102.129:5000/api"
    private val notifications = mutableListOf<NotificationItem>()
    private lateinit var adapter: NotificationAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_notifications)

        adapter = NotificationAdapter(notifications)
        findViewById<RecyclerView>(R.id.rvNotifications).apply {
            layoutManager = LinearLayoutManager(this@NotificationsActivity)
            this.adapter = this@NotificationsActivity.adapter
        }

        findViewById<Button>(R.id.btnMarkAllRead).setOnClickListener {
            adapter.markAllAsRead()
            Toast.makeText(this, "All marked as read.", Toast.LENGTH_SHORT).show()
        }

        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = getString(R.string.notifications)

        fetchNotifications()
    }

    private fun fetchNotifications() {
        val prefs = getSharedPreferences("roadguard", MODE_PRIVATE)
        val token = prefs.getString("token", null) ?: return
        val userId = prefs.getString("userId", null) ?: return

        val request = Request.Builder()
            .url("$BASE_URL/notifications/$userId")
            .addHeader("Authorization", "Bearer $token")
            .build()

        OkHttpClient().newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread { Toast.makeText(this@NotificationsActivity, "Failed to load", Toast.LENGTH_SHORT).show() }
            }
            override fun onResponse(call: Call, response: Response) {
                val body = response.body?.string() ?: return
                runOnUiThread {
                    try {
                        val data = JSONObject(body).getJSONArray("data")
                        notifications.clear()
                        for (i in 0 until data.length()) {
                            val obj = data.getJSONObject(i)
                            notifications.add(NotificationItem(
                                message = obj.getString("message"),
                                timestamp = obj.getString("createdAt"),
                                isRead = obj.getBoolean("isRead")
                            ))
                        }
                        adapter.notifyDataSetChanged()
                    } catch (e: Exception) {
                        Toast.makeText(this@NotificationsActivity, "Parse error", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        })
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressedDispatcher.onBackPressed()
        return true
    }
}
