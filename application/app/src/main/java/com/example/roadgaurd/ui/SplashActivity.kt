package com.example.roadgaurd.ui

import android.content.Intent
import android.content.SharedPreferences
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.roadgaurd.AppConfig
import com.example.roadgaurd.R
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException

class SplashActivity : AppCompatActivity() {

    private lateinit var prefs: SharedPreferences
    private var isRegisterMode = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = getSharedPreferences("roadguard", MODE_PRIVATE)

        // Already logged in with a real token -> go straight to Home. A previous DEV build
        // may have stored a fake "dev-offline-token"; ignore and clear it so the login
        // screen shows again instead of silently using a token the server will reject.
        val token = prefs.getString("token", null)
        if (token != null && token != "dev-offline-token") {
            startActivity(Intent(this, HomeActivity::class.java))
            finish()
            return
        }
        prefs.edit().remove("token").remove("userId").remove("name").apply()

        setContentView(R.layout.activity_splash)

        val etName = findViewById<EditText>(R.id.etName)
        val btnLogin = findViewById<Button>(R.id.btnLogin)
        val tvToggle = findViewById<TextView>(R.id.tvToggle)

        findViewById<TextView>(R.id.tvSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        tvToggle.setOnClickListener {
            isRegisterMode = !isRegisterMode
            if (isRegisterMode) {
                etName.visibility = View.VISIBLE
                btnLogin.text = "REGISTER"
                tvToggle.text = "Already have an account? Login"
            } else {
                etName.visibility = View.GONE
                btnLogin.text = "LOGIN"
                tvToggle.text = "Don't have an account? Register"
            }
        }

        btnLogin.setOnClickListener {
            val name = etName.text.toString().trim()
            val email = findViewById<EditText>(R.id.etEmail).text.toString().trim()
            val password = findViewById<EditText>(R.id.etPassword).text.toString().trim()
            if (isRegisterMode && name.isEmpty()) {
                Toast.makeText(this, "Enter your name", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (email.isEmpty() || password.isEmpty()) {
                Toast.makeText(this, "Enter email and password", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (isRegisterMode) register(name, email, password)
            else login(email, password)
        }
    }

    private fun login(email: String, password: String) {
        val json = JSONObject().apply {
            put("email", email)
            put("password", password)
        }
        makeRequest("${AppConfig.getBaseUrl(this)}/auth/login", json)
    }

    private fun register(name: String, email: String, password: String) {
        val json = JSONObject().apply {
            put("name", name)
            put("email", email)
            put("password", password)
            put("role", "driver")
        }
        makeRequest("${AppConfig.getBaseUrl(this)}/auth/register", json)
    }

    private fun makeRequest(url: String, json: JSONObject) {
        val body = json.toString().toRequestBody("application/json".toMediaType())
        val request = Request.Builder().url(url).post(body).build()

        OkHttpClient().newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    Toast.makeText(this@SplashActivity, "Network error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                val resBody = response.body?.string()
                runOnUiThread {
                    try {
                        val json = JSONObject(resBody ?: "")
                        if (json.getBoolean("success")) {
                            val data = json.getJSONObject("data")
                            prefs.edit()
                                .putString("token", data.getString("token"))
                                .putString("userId", data.getString("id"))
                                .putString("name", data.getString("name"))
                                .apply()
                            startActivity(Intent(this@SplashActivity, HomeActivity::class.java))
                            finish()
                        } else {
                            val message = json.optString("message", "Something went wrong")
                            Toast.makeText(this@SplashActivity, message, Toast.LENGTH_SHORT).show()
                        }
                    } catch (e: Exception) {
                        Toast.makeText(this@SplashActivity, "Unexpected error", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        })
    }
}