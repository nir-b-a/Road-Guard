package com.example.roadgaurd.ui

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.roadgaurd.AppConfig
import com.example.roadgaurd.R

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        val etUrl = findViewById<EditText>(R.id.etServerUrl)
        val btnSave = findViewById<Button>(R.id.btnSave)

        etUrl.setText(AppConfig.getBaseUrl(this))

        btnSave.setOnClickListener {
            val url = etUrl.text.toString().trim()
            if (url.isEmpty()) {
                Toast.makeText(this, "URL cannot be empty", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            AppConfig.setBaseUrl(this, url)
            Toast.makeText(this, "Server URL saved", Toast.LENGTH_SHORT).show()
            finish()
        }
    }
}
