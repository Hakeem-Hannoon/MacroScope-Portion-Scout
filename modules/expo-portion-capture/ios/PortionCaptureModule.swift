import ARKit
import ExpoModulesCore

struct LaunchOptions: Record {
  @Field var requireStroke: Bool = true
  @Field var minStrokeLengthM: Double = 0.10
}

public class PortionCaptureModule: Module {
  public func definition() -> ModuleDefinition {
    Name("ExpoPortionCapture")

    Function("isSupported") { () -> Bool in
      ARWorldTrackingConfiguration.isSupported
    }

    AsyncFunction("launch") { (options: LaunchOptions, promise: Promise) in
      DispatchQueue.main.async {
        guard ARWorldTrackingConfiguration.isSupported else {
          promise.reject("ERR_AR_UNSUPPORTED", "ARKit world tracking is unavailable on this device")
          return
        }
        guard let presenter = Self.topViewController() else {
          promise.reject("ERR_NO_VIEWCONTROLLER", "No view controller available to present from")
          return
        }
        let controller = ARCaptureViewController()
        controller.requireStroke = options.requireStroke
        controller.minStrokeLengthM = Float(options.minStrokeLengthM)
        controller.modalPresentationStyle = .fullScreen
        controller.onComplete = { payload in
          promise.resolve(payload)
        }
        presenter.present(controller, animated: true)
      }
    }
  }

  private static func topViewController() -> UIViewController? {
    let windows = UIApplication.shared.connectedScenes
      .compactMap { $0 as? UIWindowScene }
      .flatMap(\.windows)
    var top = (windows.first { $0.isKeyWindow } ?? windows.first)?.rootViewController
    while let presented = top?.presentedViewController {
      top = presented
    }
    return top
  }
}
