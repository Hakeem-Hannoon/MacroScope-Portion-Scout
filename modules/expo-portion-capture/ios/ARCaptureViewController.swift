import ARKit
import CoreImage
import SceneKit
import UIKit

/// One committed ruler stroke, world-space meters (MATH.md §2.3).
private struct RulerStroke {
  let p1: simd_float3
  let p2: simd_float3
  let kind: String // "horizontal" | "vertical"
  let node: SCNNode
  var lengthM: Float { simd_distance(p1, p2) }
}

/// Full-screen AR capture: the user tap-holds and drags to draw metric ruler
/// strokes (raycast against tracked geometry each frame — MATH.md §2), then
/// takes the photo. Produces the versioned CapturePayload consumed by
/// @ppe/pipeline (schema: packages/pipeline/src/contracts.ts).
final class ARCaptureViewController: UIViewController {
  var requireStroke = true
  var minStrokeLengthM: Float = 0.10
  /// Called exactly once: payload dictionary, or nil when cancelled.
  var onComplete: (([String: Any]?) -> Void)?

  private let sceneView = ARSCNView()
  private let coachingOverlay = ARCoachingOverlayView()
  private let measureLabel = UILabel()
  private let hintLabel = UILabel()
  private let shutterButton = UIButton(type: .system)
  private let cancelButton = UIButton(type: .system)
  private let undoButton = UIButton(type: .system)

  private var strokes: [RulerStroke] = []
  private var activeStart: simd_float3?
  private var activeNode: SCNNode?
  /// The plane the meal sits on: locked by the first successful stroke raycast.
  private var lockedPlaneAnchor: ARPlaneAnchor?
  private var lockedPlaneTransform: simd_float4x4?
  private var planeSource = "estimated"
  private var completed = false

  // MARK: - Lifecycle

  override func viewDidLoad() {
    super.viewDidLoad()
    view.backgroundColor = .black
    layoutViews()

    sceneView.automaticallyUpdatesLighting = true
    sceneView.session.run(Self.makeConfiguration())

    coachingOverlay.session = sceneView.session
    coachingOverlay.goal = .horizontalPlane
    coachingOverlay.activatesAutomatically = true

    let press = UILongPressGestureRecognizer(target: self, action: #selector(handleLongPress(_:)))
    press.minimumPressDuration = 0.15
    press.allowableMovement = .greatestFiniteMagnitude
    sceneView.addGestureRecognizer(press)

    updateControls()
  }

  override func viewWillDisappear(_ animated: Bool) {
    super.viewWillDisappear(animated)
    sceneView.session.pause()
  }

  override var prefersStatusBarHidden: Bool { true }

  private static func makeConfiguration() -> ARWorldTrackingConfiguration {
    let config = ARWorldTrackingConfiguration()
    config.planeDetection = [.horizontal]
    if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
      config.sceneReconstruction = .mesh
    }
    if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
      config.frameSemantics.insert(.smoothedSceneDepth)
    }
    return config
  }

  // MARK: - The ruler gesture (MATH.md §2.3)

  @objc private func handleLongPress(_ gesture: UILongPressGestureRecognizer) {
    let point = gesture.location(in: sceneView)
    switch gesture.state {
    case .began:
      guard let hit = raycast(at: point) else {
        setMeasureText("Point at the table surface")
        return
      }
      rememberPlane(from: hit)
      activeStart = hit.worldTransform.position
    case .changed:
      guard let start = activeStart, let hit = raycast(at: point) else { return }
      let end = hit.worldTransform.position
      redrawActiveStroke(from: start, to: end)
      setMeasureText(Self.format(meters: simd_distance(start, end)))
    case .ended:
      defer {
        activeStart = nil
        activeNode = nil
      }
      guard let start = activeStart, let hit = raycast(at: point) else {
        activeNode?.removeFromParentNode()
        return
      }
      let end = hit.worldTransform.position
      let length = simd_distance(start, end)
      guard length >= 0.005, let node = activeNode else {
        activeNode?.removeFromParentNode()
        setMeasureText("Too short — drag a longer line")
        return
      }
      let direction = simd_normalize(end - start)
      let vertical = abs(simd_dot(direction, planeNormal())) > 0.7
      strokes.append(
        RulerStroke(p1: start, p2: end, kind: vertical ? "vertical" : "horizontal", node: node)
      )
      setMeasureText(Self.format(meters: length))
      updateControls()
    case .cancelled, .failed:
      activeNode?.removeFromParentNode()
      activeStart = nil
      activeNode = nil
    default:
      break
    }
  }

  /// Prefer mapped geometry (LiDAR mesh / detected planes); fall back to the
  /// estimated plane so the ruler works before mapping completes.
  private func raycast(at point: CGPoint) -> ARRaycastResult? {
    let targets: [ARRaycastQuery.Target] = [.existingPlaneGeometry, .estimatedPlane]
    for target in targets {
      if let query = sceneView.raycastQuery(from: point, allowing: target, alignment: .any),
         let result = sceneView.session.raycast(query).first {
        return result
      }
    }
    return nil
  }

  private func rememberPlane(from hit: ARRaycastResult) {
    if let anchor = hit.anchor as? ARPlaneAnchor {
      lockedPlaneAnchor = anchor
      planeSource = "detected_plane"
    } else if lockedPlaneAnchor == nil, lockedPlaneTransform == nil {
      // Raycast surface frame: the Y column is the surface normal.
      lockedPlaneTransform = hit.worldTransform
      planeSource = "estimated"
    }
  }

  private func planeNormal() -> simd_float3 {
    if let anchor = lockedPlaneAnchor {
      return simd_normalize(anchor.transform.columns.1.xyz)
    }
    if let transform = lockedPlaneTransform {
      return simd_normalize(transform.columns.1.xyz)
    }
    return simd_float3(0, 1, 0)
  }

  private func redrawActiveStroke(from start: simd_float3, to end: simd_float3) {
    activeNode?.removeFromParentNode()
    let node = Self.strokeNode(from: start, to: end)
    sceneView.scene.rootNode.addChildNode(node)
    activeNode = node
  }

  /// A thin cylinder between two world points, with endpoint dots.
  private static func strokeNode(from start: simd_float3, to end: simd_float3) -> SCNNode {
    let parent = SCNNode()
    let length = simd_distance(start, end)
    if length > 1e-4 {
      let cylinder = SCNCylinder(radius: 0.0015, height: CGFloat(length))
      cylinder.firstMaterial?.diffuse.contents = UIColor.systemYellow
      cylinder.firstMaterial?.lightingModel = .constant
      let line = SCNNode(geometry: cylinder)
      line.simdPosition = (start + end) / 2
      // SCNCylinder's axis is +Y; rotate it onto the stroke direction.
      line.simdOrientation = simd_quatf(from: simd_float3(0, 1, 0), to: simd_normalize(end - start))
      parent.addChildNode(line)
    }
    for point in [start, end] {
      let dot = SCNNode(geometry: SCNSphere(radius: 0.004))
      dot.geometry?.firstMaterial?.diffuse.contents = UIColor.systemYellow
      dot.geometry?.firstMaterial?.lightingModel = .constant
      dot.simdPosition = point
      parent.addChildNode(dot)
    }
    return parent
  }

  // MARK: - Capture (docs/ARCHITECTURE.md payload contract)

  @objc private func shutterTapped() {
    guard let frame = sceneView.session.currentFrame else { return }
    guard case .normal = frame.camera.trackingState else {
      setMeasureText("Hold steady — tracking is limited")
      return
    }
    do {
      let payload = try buildPayload(frame: frame)
      finish(with: payload)
    } catch {
      setMeasureText("Capture failed: \(error.localizedDescription)")
    }
  }

  @objc private func cancelTapped() {
    finish(with: nil)
  }

  @objc private func undoTapped() {
    guard let last = strokes.popLast() else { return }
    last.node.removeFromParentNode()
    updateControls()
  }

  private func finish(with payload: [String: Any]?) {
    guard !completed else { return }
    completed = true
    sceneView.session.pause()
    dismiss(animated: true) { [onComplete] in
      onComplete?(payload)
    }
  }

  private enum CaptureError: LocalizedError {
    case noPlane
    var errorDescription: String? { "No table surface was detected — draw a ruler stroke first" }
  }

  private func buildPayload(frame: ARFrame) throws -> [String: Any] {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("portion-capture-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

    // 1. The image, at sensor resolution and sensor orientation. All pixel
    //    coordinates in the payload refer to this stored image (MATH.md §9.2).
    let pixelBuffer = frame.capturedImage
    let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
    let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
    let imageURL = try writeImage(pixelBuffer, to: directory)

    // 2. The supporting plane: n·X = d0 (MATH.md §2.2).
    let normal: simd_float3
    var extent: [Float]? = nil
    var planePoint: simd_float3
    if let anchor = lockedPlaneAnchor {
      normal = simd_normalize(anchor.transform.columns.1.xyz)
      planePoint = anchor.transform.columns.3.xyz
      extent = [anchor.planeExtent.width, anchor.planeExtent.height]
      planeSource = sceneView.session.configuration?.sceneReconstruction == .mesh
        ? "lidar_mesh" : "detected_plane"
    } else if let transform = lockedPlaneTransform {
      normal = simd_normalize(transform.columns.1.xyz)
      planePoint = transform.columns.3.xyz
    } else if let firstStroke = strokes.first {
      normal = simd_float3(0, 1, 0)
      planePoint = firstStroke.p1
    } else {
      throw CaptureError.noPlane
    }
    let d0 = simd_dot(normal, planePoint)

    // 3. Depth, when the hardware provides it.
    var depthDict: Any = NSNull()
    if let sceneDepth = frame.smoothedSceneDepth ?? frame.sceneDepth {
      depthDict = try writeDepth(
        sceneDepth,
        to: directory,
        cameraIntrinsics: frame.camera.intrinsics,
        imageSize: (imageWidth, imageHeight)
      )
    }

    let horizontalOK = strokes.contains { $0.kind == "horizontal" && $0.lengthM >= minStrokeLengthM }
    let scaleSource: String = depthDict is NSNull ? (horizontalOK ? "ruler" : "none") : "lidar"

    return [
      "version": 1,
      "image": imageURL.absoluteString,
      "image_size": [imageWidth, imageHeight],
      "intrinsics": Self.rows(of: frame.camera.intrinsics),
      // simd matrices are column-major; the contract is row-major.
      "camera_to_world": Self.rowMajor(frame.camera.transform),
      "plane": [
        "normal": [normal.x, normal.y, normal.z],
        "d0": d0,
        "extent": extent as Any,
      ],
      "strokes": strokes.map { stroke in
        [
          "p1": [stroke.p1.x, stroke.p1.y, stroke.p1.z],
          "p2": [stroke.p2.x, stroke.p2.y, stroke.p2.z],
          "length_m": stroke.lengthM,
          "kind": stroke.kind,
        ]
      },
      "depth": depthDict,
      "tracking": ["state": "normal", "plane_source": planeSource],
      "scale_source": scaleSource,
    ]
  }

  private func writeImage(_ pixelBuffer: CVPixelBuffer, to directory: URL) throws -> URL {
    let image = CIImage(cvPixelBuffer: pixelBuffer)
    let context = CIContext()
    let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    let heicURL = directory.appendingPathComponent("capture.heic")
    do {
      try context.writeHEIFRepresentation(of: image, to: heicURL, format: .RGBA8, colorSpace: colorSpace)
      return heicURL
    } catch {
      // Simulators and older devices without HEIC encoders fall back to JPEG.
      let jpegURL = directory.appendingPathComponent("capture.jpg")
      try context.writeJPEGRepresentation(of: image, to: jpegURL, colorSpace: colorSpace)
      return jpegURL
    }
  }

  /// Serializes the depth + confidence maps as raw binaries, with intrinsics
  /// rescaled from the RGB resolution to the depth resolution (MATH.md §9.1).
  private func writeDepth(
    _ sceneDepth: ARDepthData,
    to directory: URL,
    cameraIntrinsics: simd_float3x3,
    imageSize: (Int, Int)
  ) throws -> [String: Any] {
    let depthMap = sceneDepth.depthMap
    let width = CVPixelBufferGetWidth(depthMap)
    let height = CVPixelBufferGetHeight(depthMap)

    let depthURL = directory.appendingPathComponent("depth.f32")
    try Self.pixelBufferData(depthMap, bytesPerPixel: 4).write(to: depthURL)

    var confidencePath: Any = NSNull()
    if let confidenceMap = sceneDepth.confidenceMap {
      let confidenceURL = directory.appendingPathComponent("depth-confidence.u8")
      try Self.pixelBufferData(confidenceMap, bytesPerPixel: 1).write(to: confidenceURL)
      confidencePath = confidenceURL.absoluteString
    }

    let sx = Float(width) / Float(imageSize.0)
    let sy = Float(height) / Float(imageSize.1)
    var k = cameraIntrinsics
    k[0][0] *= sx // fx  (simd_float3x3 subscripts are [column][row])
    k[1][1] *= sy // fy
    k[2][0] *= sx // cx
    k[2][1] *= sy // cy

    return [
      "map": depthURL.absoluteString,
      "confidence": confidencePath,
      "size": [width, height],
      "intrinsics": Self.rows(of: k),
    ]
  }

  private static func pixelBufferData(_ buffer: CVPixelBuffer, bytesPerPixel: Int) -> Data {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    let base = CVPixelBufferGetBaseAddress(buffer)!
    var data = Data(capacity: width * height * bytesPerPixel)
    let rowLength = width * bytesPerPixel
    for row in 0..<height {
      data.append(
        Data(bytes: base.advanced(by: row * bytesPerRow), count: rowLength)
      )
    }
    return data
  }

  // MARK: - Matrix serialization (row-major contract)

  private static func rowMajor(_ m: simd_float4x4) -> [Float] {
    // simd stores columns; the payload stores rows: element (r, c) = columns[c][r].
    var out: [Float] = []
    out.reserveCapacity(16)
    for r in 0..<4 {
      for c in 0..<4 {
        out.append(m[c][r])
      }
    }
    return out
  }

  private static func rows(of m: simd_float3x3) -> [[Float]] {
    (0..<3).map { r in (0..<3).map { c in m[c][r] } }
  }

  private static func format(meters: Float) -> String {
    meters < 1 ? String(format: "%.1f cm", meters * 100) : String(format: "%.2f m", meters)
  }

  // MARK: - UI chrome

  private func setMeasureText(_ text: String) {
    measureLabel.text = text
  }

  private func updateControls() {
    let hasValidStroke = strokes.contains {
      $0.kind == "horizontal" && $0.lengthM >= minStrokeLengthM
    }
    shutterButton.isEnabled = hasValidStroke || !requireStroke
    shutterButton.alpha = shutterButton.isEnabled ? 1 : 0.4
    undoButton.isHidden = strokes.isEmpty
    hintLabel.text = hasValidStroke
      ? "Add a vertical stroke up the food for better accuracy, or shoot"
      : "Hold and drag along the plate to measure (≥ \(Int(minStrokeLengthM * 100)) cm)"
  }

  private func layoutViews() {
    for subview in [sceneView, coachingOverlay] {
      subview.translatesAutoresizingMaskIntoConstraints = false
      view.addSubview(subview)
      NSLayoutConstraint.activate([
        subview.topAnchor.constraint(equalTo: view.topAnchor),
        subview.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        subview.leadingAnchor.constraint(equalTo: view.leadingAnchor),
        subview.trailingAnchor.constraint(equalTo: view.trailingAnchor),
      ])
    }

    measureLabel.textColor = .white
    measureLabel.font = .monospacedDigitSystemFont(ofSize: 22, weight: .semibold)
    measureLabel.textAlignment = .center

    hintLabel.textColor = .white.withAlphaComponent(0.85)
    hintLabel.font = .systemFont(ofSize: 14)
    hintLabel.textAlignment = .center
    hintLabel.numberOfLines = 2

    shutterButton.setImage(UIImage(systemName: "circle.inset.filled"), for: .normal)
    shutterButton.tintColor = .white
    shutterButton.transform = CGAffineTransform(scaleX: 2.4, y: 2.4)
    shutterButton.addTarget(self, action: #selector(shutterTapped), for: .touchUpInside)

    cancelButton.setTitle("Cancel", for: .normal)
    cancelButton.tintColor = .white
    cancelButton.addTarget(self, action: #selector(cancelTapped), for: .touchUpInside)

    undoButton.setImage(UIImage(systemName: "arrow.uturn.backward"), for: .normal)
    undoButton.tintColor = .white
    undoButton.addTarget(self, action: #selector(undoTapped), for: .touchUpInside)

    for control in [measureLabel, hintLabel, shutterButton, cancelButton, undoButton] {
      control.translatesAutoresizingMaskIntoConstraints = false
      view.addSubview(control)
    }
    let safe = view.safeAreaLayoutGuide
    NSLayoutConstraint.activate([
      measureLabel.topAnchor.constraint(equalTo: safe.topAnchor, constant: 12),
      measureLabel.centerXAnchor.constraint(equalTo: safe.centerXAnchor),
      hintLabel.topAnchor.constraint(equalTo: measureLabel.bottomAnchor, constant: 6),
      hintLabel.leadingAnchor.constraint(equalTo: safe.leadingAnchor, constant: 24),
      hintLabel.trailingAnchor.constraint(equalTo: safe.trailingAnchor, constant: -24),
      shutterButton.centerXAnchor.constraint(equalTo: safe.centerXAnchor),
      shutterButton.bottomAnchor.constraint(equalTo: safe.bottomAnchor, constant: -32),
      cancelButton.leadingAnchor.constraint(equalTo: safe.leadingAnchor, constant: 24),
      cancelButton.centerYAnchor.constraint(equalTo: shutterButton.centerYAnchor),
      undoButton.trailingAnchor.constraint(equalTo: safe.trailingAnchor, constant: -24),
      undoButton.centerYAnchor.constraint(equalTo: shutterButton.centerYAnchor),
    ])
  }
}

private extension simd_float4 {
  var xyz: simd_float3 { simd_float3(x, y, z) }
}

private extension simd_float4x4 {
  var position: simd_float3 { columns.3.xyz }
}
