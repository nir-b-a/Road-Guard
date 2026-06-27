package com.example.roadgaurd.storage

import android.content.Context
import android.util.Log
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * On-device storage policy for recorded sessions.
 *
 * Each drive is written by RecordingActivity to
 *     getExternalFilesDir(null)/sessions/<sessionId>/
 * A session is deleted as soon as it uploads successfully (PostDriveActivity).
 * Sessions that did NOT upload are kept so the captured data is not lost, then
 * reclaimed by a launch-time sweep once they are older than SESSION_RETENTION_HOURS.
 */
object SessionStore {

    // ─────────────────────────────────────────────────────────────────────────
    // Retention window for UNSENT sessions. A session is deleted the moment it
    // uploads successfully; if it did NOT upload (e.g. the server rejected it, or
    // the phone was offline) it stays on disk so the data isn't lost, and the
    // launch-time sweep (HomeActivity) reclaims it only once it is older than this.
    // ─────────────────────────────────────────────────────────────────────────
    const val SESSION_RETENTION_HOURS = 24L

    private const val TAG = "SessionStore"

    /** Root folder holding every per-session directory. */
    fun sessionsRoot(context: Context): File =
        File(context.getExternalFilesDir(null), "sessions")

    /** Delete one session folder (called after a successful upload). Returns true on success. */
    fun deleteSession(dir: File): Boolean {
        if (!dir.exists()) return true
        val ok = dir.deleteRecursively()
        if (!ok) Log.w(TAG, "Failed to delete session folder ${dir.absolutePath}")
        return ok
    }

    /**
     * Delete UNSENT sessions older than SESSION_RETENTION_HOURS. A folder's age is
     * its last-modified time (set when RecordingActivity wrote the session files).
     * Successfully uploaded sessions are already gone, so anything still on disk is
     * an un-uploaded drive.
     */
    fun sweepStaleSessions(context: Context) {
        val children = sessionsRoot(context).listFiles() ?: return
        val maxAgeMs = TimeUnit.HOURS.toMillis(SESSION_RETENTION_HOURS)
        val now = System.currentTimeMillis()
        for (dir in children) {
            if (!dir.isDirectory) continue
            val ageMs = now - dir.lastModified()
            if (ageMs > maxAgeMs && deleteSession(dir)) {
                Log.i(TAG, "Swept stale session ${dir.name} (age ${ageMs / 3_600_000}h)")
            }
        }
    }
}
