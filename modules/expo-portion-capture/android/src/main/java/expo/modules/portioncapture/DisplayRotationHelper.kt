package expo.modules.portioncapture

import android.app.Activity
import android.hardware.display.DisplayManager
import com.google.ar.core.Session

/**
 * Feeds display size/rotation changes to the ARCore session so the camera
 * background and hit-testing stay aligned with the screen.
 */
class DisplayRotationHelper(private val activity: Activity) : DisplayManager.DisplayListener {
  private var viewportChanged = false
  private var viewportWidth = 0
  private var viewportHeight = 0
  private val displayManager =
    activity.getSystemService(Activity.DISPLAY_SERVICE) as DisplayManager

  fun onResume() = displayManager.registerDisplayListener(this, null)

  fun onPause() = displayManager.unregisterDisplayListener(this)

  fun onSurfaceChanged(width: Int, height: Int) {
    viewportWidth = width
    viewportHeight = height
    viewportChanged = true
  }

  fun updateSessionIfNeeded(session: Session) {
    if (!viewportChanged) return
    @Suppress("DEPRECATION")
    val rotation = activity.windowManager.defaultDisplay.rotation
    session.setDisplayGeometry(rotation, viewportWidth, viewportHeight)
    viewportChanged = false
  }

  override fun onDisplayAdded(displayId: Int) {}
  override fun onDisplayRemoved(displayId: Int) {}
  override fun onDisplayChanged(displayId: Int) {
    viewportChanged = true
  }
}
