package com.example.roadgaurd.upload

import android.os.Handler
import android.os.Looper
import java.io.File
import java.util.concurrent.CopyOnWriteArraySet

/**
 * In-process status of the drive upload that [UploadService] runs in the background.
 *
 * The service posts every change here; screens observe it: PostDriveActivity mirrors the progress
 * bar and shows the result, HomeActivity and SessionStore leave the session that is uploading
 * right now alone. Listeners are always called on the main thread. Nothing is persisted: if the
 * process dies, so does the upload, and the kept session folder is the retry path (the
 * "Unfinished Upload" prompt at the next launch).
 */
object UploadProgress {

    sealed class State(val sessionId: String) {
        class Uploading(sessionId: String, val percent: Int) : State(sessionId)
        class Finalizing(sessionId: String) : State(sessionId)
        class Succeeded(sessionId: String) : State(sessionId)
        class Failed(sessionId: String, val message: String) : State(sessionId)
        class SessionExpired(sessionId: String) : State(sessionId)
    }

    /** Absolute path of the session folder being uploaded right now, or null when idle. */
    @Volatile var activeSessionDir: String? = null
        internal set

    /** The latest state. Kept after the upload ends, so a screen that comes back sees the result. */
    @Volatile var current: State? = null
        private set

    private val listeners = CopyOnWriteArraySet<(State) -> Unit>()
    private val mainHandler = Handler(Looper.getMainLooper())

    /** True while [dir] is being uploaded: it must be neither offered again nor swept. */
    fun isUploading(dir: File): Boolean = activeSessionDir == dir.absolutePath

    /** Start listening; the latest state (if any) is delivered right away. */
    fun observe(listener: (State) -> Unit) {
        listeners.add(listener)
        mainHandler.post { current?.let { if (listener in listeners) listener(it) } }
    }

    fun remove(listener: (State) -> Unit) {
        listeners.remove(listener)
    }

    internal fun post(state: State) {
        current = state
        mainHandler.post { listeners.forEach { it(state) } }
    }
}
