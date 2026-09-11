package com.example.roadgaurd.storage

import android.content.Context
import android.util.Log
import com.example.roadgaurd.AppConfig
import com.example.roadgaurd.upload.UploadProgress
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * On-device storage policy for recorded sessions.
 *
 * Each drive is written by RecordingActivity to
 *     getExternalFilesDir(null)/sessions/<sessionId>/
 * On successful upload, UploadService writes an "uploaded.flag" sentinel and returns
 * immediately — the session stays on device for UPLOAD_GRACE_MINUTES so the user can
 * verify or retry if something looked wrong. The launch-time sweep (HomeActivity) then
 * deletes flagged dirs once the grace period has passed. Un-uploaded sessions are kept
 * for SESSION_RETENTION_HOURS before being swept.
 */
object SessionStore {

    const val SESSION_RETENTION_HOURS = 24L
    const val UPLOAD_GRACE_MINUTES = 2L
    private const val UPLOADED_FLAG = "uploaded.flag"
    private const val SKIPPED_FLAG  = "skipped.flag"
    private const val MIN_VIDEO_BYTES = 5L * 1024 * 1024   // 5 MB — keep in sync with PostDriveActivity

    private const val TAG = "SessionStore"

    /** Root folder holding every per-session directory. */
    fun sessionsRoot(context: Context): File =
        File(context.getExternalFilesDir(null), "sessions")

    /** Write the sentinel that marks this session as successfully uploaded. */
    fun markUploaded(dir: File) {
        try { File(dir, UPLOADED_FLAG).createNewFile() }
        catch (e: Exception) { Log.w(TAG, "Could not write uploaded flag: ${e.message}") }
    }

    /** Write the sentinel that marks this session as deliberately skipped by the user. */
    fun markSkipped(dir: File) {
        try { File(dir, SKIPPED_FLAG).createNewFile() }
        catch (e: Exception) { Log.w(TAG, "Could not write skipped flag: ${e.message}") }
    }

    /** Delete one session folder. Returns true on success. */
    fun deleteSession(dir: File): Boolean {
        if (!dir.exists()) return true
        val ok = dir.deleteRecursively()
        if (!ok) Log.w(TAG, "Failed to delete session folder ${dir.absolutePath}")
        return ok
    }

    /**
     * Return sessions that are on-device but not yet uploaded and haven't expired.
     * Sorted newest-first so the caller can prioritise the most recent one.
     * Each entry is (sessionId, sessionDir, videoFile).
     */
    fun findPendingSessions(context: Context): List<Triple<String, File, File>> {
        val children = sessionsRoot(context).listFiles() ?: return emptyList()
        val now = System.currentTimeMillis()
        val maxAgeMs = TimeUnit.HOURS.toMillis(SESSION_RETENTION_HOURS)
        return children
            .filter { dir ->
                dir.isDirectory &&
                !File(dir, UPLOADED_FLAG).exists() &&
                !File(dir, SKIPPED_FLAG).exists() &&
                (now - dir.lastModified()) < maxAgeMs
            }
            .sortedByDescending { it.lastModified() }
            .mapNotNull { dir ->
                val video = dir.listFiles()?.find { it.name.endsWith(".mp4") } ?: return@mapNotNull null
                if (video.length() < MIN_VIDEO_BYTES) return@mapNotNull null
                Triple(dir.name, dir, video)
            }
    }

    /**
     * Sweep stale sessions at launch:
     * - Uploaded (flag present): delete after UPLOAD_GRACE_MINUTES.
     * - Not uploaded: delete after SESSION_RETENTION_HOURS.
     *
     * Disabled entirely in offline builds: retention exists to reclaim space once a drive is
     * safely on the server, and offline it never is — sweeping would silently destroy the only
     * copy of the data. Offline sessions are kept until they are pulled off the phone by hand.
     */
    fun sweepStaleSessions(context: Context) {
        if (AppConfig.OFFLINE_MODE) {
            Log.i(TAG, "offline mode — retention sweep disabled, sessions kept on device")
            return
        }
        val children = sessionsRoot(context).listFiles() ?: return
        val now = System.currentTimeMillis()
        for (dir in children) {
            if (!dir.isDirectory) continue
            if (UploadProgress.isUploading(dir)) continue   // never delete a folder mid-upload
            val flag = File(dir, UPLOADED_FLAG)
            if (flag.exists()) {
                val graceMs = TimeUnit.MINUTES.toMillis(UPLOAD_GRACE_MINUTES)
                if (now - flag.lastModified() > graceMs && deleteSession(dir))
                    Log.i(TAG, "Swept uploaded session ${dir.name}")
            } else {
                val maxAgeMs = TimeUnit.HOURS.toMillis(SESSION_RETENTION_HOURS)
                val ageMs = now - dir.lastModified()
                if (ageMs > maxAgeMs && deleteSession(dir))
                    Log.i(TAG, "Swept stale session ${dir.name} (age ${ageMs / 3_600_000}h)")
            }
        }
    }
}
