package expo.modules.portioncapture

import android.content.Intent
import com.google.ar.core.ArCoreApk
import expo.modules.kotlin.Promise
import expo.modules.kotlin.exception.Exceptions
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import expo.modules.kotlin.records.Field
import expo.modules.kotlin.records.Record

class LaunchOptions : Record {
  @Field var requireStroke: Boolean = true
  @Field var minStrokeLengthM: Double = 0.10
}

/**
 * Hands the resolved payload from ARCaptureActivity back to the pending JS
 * promise. One capture at a time; delivery is single-shot.
 */
object CaptureBridge {
  private var pending: Promise? = null

  @Synchronized
  fun take(promise: Promise): Boolean {
    if (pending != null) return false
    pending = promise
    return true
  }

  @Synchronized
  fun deliver(payload: Map<String, Any?>?) {
    pending?.resolve(payload)
    pending = null
  }
}

class PortionCaptureModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("ExpoPortionCapture")

    Function("isSupported") {
      val context = appContext.reactContext ?: return@Function false
      val availability = ArCoreApk.getInstance().checkAvailability(context)
      availability.isSupported || availability.isTransient
    }

    AsyncFunction("launch") { options: LaunchOptions, promise: Promise ->
      val activity = appContext.currentActivity ?: throw Exceptions.MissingActivity()
      if (!CaptureBridge.take(promise)) {
        promise.reject("ERR_CAPTURE_IN_PROGRESS", "A capture session is already open", null)
        return@AsyncFunction
      }
      val intent = Intent(activity, ARCaptureActivity::class.java)
        .putExtra(ARCaptureActivity.EXTRA_REQUIRE_STROKE, options.requireStroke)
        .putExtra(ARCaptureActivity.EXTRA_MIN_STROKE_M, options.minStrokeLengthM.toFloat())
      activity.startActivity(intent)
    }
  }
}
