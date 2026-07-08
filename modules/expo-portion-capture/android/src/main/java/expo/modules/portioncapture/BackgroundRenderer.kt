package expo.modules.portioncapture

import android.opengl.GLES11Ext
import android.opengl.GLES20
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Draws the ARCore camera image as a fullscreen quad (external OES texture).
 * Texture coordinates come from Frame.transformCoordinates2d so the image
 * stays correct across display rotations and aspect crops.
 */
class BackgroundRenderer {
  var textureId = -1
    private set

  private var program = 0
  private var positionAttrib = 0
  private var texCoordAttrib = 0
  private var textureUniform = 0

  private val quadCoords: FloatBuffer = ByteBuffer
    .allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    .put(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)).apply { position(0) } as FloatBuffer

  private val quadTexCoords: FloatBuffer = ByteBuffer
    .allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

  fun createOnGlThread() {
    val textures = IntArray(1)
    GLES20.glGenTextures(1, textures, 0)
    textureId = textures[0]
    GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
    GLES20.glTexParameteri(
      GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR,
    )
    GLES20.glTexParameteri(
      GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR,
    )

    val vertex = compile(
      GLES20.GL_VERTEX_SHADER,
      """
      attribute vec4 a_Position;
      attribute vec2 a_TexCoord;
      varying vec2 v_TexCoord;
      void main() {
        gl_Position = a_Position;
        v_TexCoord = a_TexCoord;
      }
      """,
    )
    val fragment = compile(
      GLES20.GL_FRAGMENT_SHADER,
      """
      #extension GL_OES_EGL_image_external : require
      precision mediump float;
      varying vec2 v_TexCoord;
      uniform samplerExternalOES u_Texture;
      void main() {
        gl_FragColor = texture2D(u_Texture, v_TexCoord);
      }
      """,
    )
    program = GLES20.glCreateProgram()
    GLES20.glAttachShader(program, vertex)
    GLES20.glAttachShader(program, fragment)
    GLES20.glLinkProgram(program)
    positionAttrib = GLES20.glGetAttribLocation(program, "a_Position")
    texCoordAttrib = GLES20.glGetAttribLocation(program, "a_TexCoord")
    textureUniform = GLES20.glGetUniformLocation(program, "u_Texture")
  }

  fun draw(frame: Frame) {
    if (frame.hasDisplayGeometryChanged()) {
      quadCoords.position(0)
      quadTexCoords.position(0)
      frame.transformCoordinates2d(
        Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, quadCoords,
        Coordinates2d.TEXTURE_NORMALIZED, quadTexCoords,
      )
    }
    if (frame.timestamp == 0L) return // no camera image yet

    GLES20.glDisable(GLES20.GL_DEPTH_TEST)
    GLES20.glDepthMask(false)
    GLES20.glUseProgram(program)
    GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
    GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
    GLES20.glUniform1i(textureUniform, 0)

    quadCoords.position(0)
    quadTexCoords.position(0)
    GLES20.glVertexAttribPointer(positionAttrib, 2, GLES20.GL_FLOAT, false, 0, quadCoords)
    GLES20.glVertexAttribPointer(texCoordAttrib, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)
    GLES20.glEnableVertexAttribArray(positionAttrib)
    GLES20.glEnableVertexAttribArray(texCoordAttrib)
    GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
    GLES20.glDisableVertexAttribArray(positionAttrib)
    GLES20.glDisableVertexAttribArray(texCoordAttrib)
    GLES20.glDepthMask(true)
    GLES20.glEnable(GLES20.GL_DEPTH_TEST)
  }

  private fun compile(type: Int, source: String): Int {
    val shader = GLES20.glCreateShader(type)
    GLES20.glShaderSource(shader, source.trimIndent())
    GLES20.glCompileShader(shader)
    val status = IntArray(1)
    GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
    check(status[0] != 0) { "Shader compile failed: ${GLES20.glGetShaderInfoLog(shader)}" }
    return shader
  }
}
