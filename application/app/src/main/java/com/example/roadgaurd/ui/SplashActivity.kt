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
import com.example.roadgaurd.R
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException

class SplashActivity : AppCompatActivity() {

    private companion object {
        // DEV: while the backend is offline, skip the login screen entirely so the app
        // is testable. Set to false (or delete the block in onCreate) to restore the
        // real login/register flow once the server is up.
        const val BYPASS_LOGIN = true
    }

    private val BASE_URL = "http://10.0.2.2:5000/api"
    private lateinit var prefs: SharedPreferences
    private var isRegisterMode = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = getSharedPreferences("roadguard", MODE_PRIVATE)

        // ── DEV login bypass (server down) ──
        if (BYPASS_LOGIN) {
            if (prefs.getString("token", null) == null) {
                prefs.edit()
                    .putString("token", "dev-offline-token")
                    .putString("userId", "dev-user")
                    .putString("name", "Dev Driver")
                    .apply()
            }
            startActivity(Intent(this, HomeActivity::class.java))
            finish()
            return
        }

        val token = prefs.getString("token", null)
        if (token != null) {
            startActivity(Intent(this, HomeActivity::class.java))
            finish()
            return
        }

        setContentView(R.layout.activity_splash)

        val etName = findViewById<EditText>(R.id.etName)
        val btnLogin = findViewById<Button>(R.id.btnLogin)
        val tvToggle = findViewById<TextView>(R.id.tvToggle)

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
        makeRequest("$BASE_URL/auth/login", json)
    }

    private fun register(name: String, email: String, password: String) {
        val json = JSONObject().apply {
            put("name", name)
            put("email", email)
            put("password", password)
            put("role", "driver")
        }
        makeRequest("$BASE_URL/auth/register", json)
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