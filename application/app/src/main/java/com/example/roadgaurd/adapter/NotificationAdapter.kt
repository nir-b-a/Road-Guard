package com.example.roadgaurd.adapter

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.cardview.widget.CardView
import androidx.recyclerview.widget.RecyclerView
import com.example.roadgaurd.R
import com.example.roadgaurd.model.NotificationItem

class NotificationAdapter(
    private val items: MutableList<NotificationItem>
) : RecyclerView.Adapter<NotificationAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val icon: ImageView = view.findViewById(R.id.ivIcon)
        val message: TextView = view.findViewById(R.id.tvMessage)
        val timestamp: TextView = view.findViewById(R.id.tvTimestamp)
        val card: CardView = view.findViewById(R.id.cardNotification)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_notification, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = items[position]
        holder.message.text = item.message
        holder.timestamp.text = item.timestamp
        holder.card.alpha = if (item.isRead) 0.5f else 1.0f
    }

    override fun getItemCount(): Int = items.size

    fun markAllAsRead() {
        items.forEach { it.isRead = true }
        notifyDataSetChanged()
    }
}
