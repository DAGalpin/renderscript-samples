/*
 * Copyright (C) 2026 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.android.example.rsmigration

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.HardwareRenderer
import android.graphics.PixelFormat
import android.graphics.RenderEffect
import android.graphics.RenderNode
import android.graphics.RuntimeShader
import android.hardware.HardwareBuffer
import android.media.ImageReader
import android.os.Build
import androidx.annotation.RequiresApi
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

private const val MAX_BLUR_RADIUS = 25
private const val BLUR_KERNEL_SIZE = 2 * MAX_BLUR_RADIUS + 1

private const val ROTATE_HUE_AGSL = """
    uniform shader inputImage;
    uniform mat3 colorMatrix;

    half4 main(float2 fragCoord) {
        half4 inputPixel = inputImage.eval(fragCoord);
        float3 resultRgb = clamp(colorMatrix * float3(inputPixel.rgb), 0.0, float(inputPixel.a));
        return half4(half3(resultRgb), inputPixel.a);
    }
"""

// AGSL enforces ES2 bounded-loop and constant/induction-variable array indexing rules:
// - The loop bound is fixed at compile time (51 = 2 * 25 + 1) and centered at MAX_RADIUS (25).
// - The uniform kernel array is indexed directly by the loop induction variable `i`.
// - Coordinates are clamped to texel centers [0.5, size - 0.5] to match CLAMP_TO_EDGE behavior.
private const val BLUR_HORIZONTAL_AGSL = """
    const int MAX_RADIUS = 25;

    uniform shader inputImage;
    uniform float maxCoordX;
    uniform float kernel[51];
    uniform int radius;

    half4 main(float2 fragCoord) {
        float4 blurredPixel = float4(0.0);
        for (int i = 0; i < 51; ++i) {
            int r = i - MAX_RADIUS;
            if (r >= -radius && r <= radius) {
                float sampleX = clamp(fragCoord.x + float(r), 0.5, maxCoordX);
                half4 pixel = inputImage.eval(float2(sampleX, fragCoord.y));
                blurredPixel += kernel[i] * float4(pixel);
            }
        }
        return half4(blurredPixel);
    }
"""

private const val BLUR_VERTICAL_AGSL = """
    const int MAX_RADIUS = 25;

    uniform shader inputImage;
    uniform float maxCoordY;
    uniform float kernel[51];
    uniform int radius;

    half4 main(float2 fragCoord) {
        float4 blurredPixel = float4(0.0);
        for (int i = 0; i < 51; ++i) {
            int r = i - MAX_RADIUS;
            if (r >= -radius && r <= radius) {
                float sampleY = clamp(fragCoord.y + float(r), 0.5, maxCoordY);
                half4 pixel = inputImage.eval(float2(fragCoord.x, sampleY));
                blurredPixel += kernel[i] * float4(pixel);
            }
        }
        return half4(blurredPixel);
    }
"""

@RequiresApi(Build.VERSION_CODES.TIRAMISU)
class AGSLImageProcessor : ImageProcessor {
    override val name = "AGSL"

    private val mLock = Any()
    private val mRotateHueShader = RuntimeShader(ROTATE_HUE_AGSL)
    private val mBlurHorizontalShader = RuntimeShader(BLUR_HORIZONTAL_AGSL)
    private val mBlurVerticalShader = RuntimeShader(BLUR_VERTICAL_AGSL)

    // Pre-allocated uniform arrays (column-major 3x3 hue matrix and 51-tap centered Gaussian kernel)
    private val mColorMatrixValues = FloatArray(9)
    private val mBlurKernelValues = FloatArray(BLUR_KERNEL_SIZE)

    private var mParams: Params? = null

    override fun configureInputAndOutput(inputImage: Bitmap, numberOfOutputImages: Int) {
        synchronized(mLock) {
            cleanupLocked()
            val maxCoordX = (inputImage.width - 0.5f).coerceAtLeast(0.5f)
            val maxCoordY = (inputImage.height - 0.5f).coerceAtLeast(0.5f)
            mBlurHorizontalShader.setFloatUniform("maxCoordX", maxCoordX)
            mBlurVerticalShader.setFloatUniform("maxCoordY", maxCoordY)
            mParams = Params(inputImage, numberOfOutputImages)
        }
    }

    private fun applyEffect(params: Params, renderEffect: RenderEffect): Bitmap {
        params.renderNode.setRenderEffect(renderEffect)
        val renderCanvas = params.renderNode.beginRecording()
        renderCanvas.drawBitmap(params.bitmap, 0f, 0f, null)
        params.renderNode.endRecording()
        params.hardwareRenderer.createRenderRequest()
            .setWaitForPresent(true)
            .syncAndDraw()

        val image = params.imageReader.acquireNextImage() ?: throw RuntimeException("No Image")
        val hardwareBuffer = image.hardwareBuffer ?: run {
            image.close()
            throw RuntimeException("No HardwareBuffer")
        }
        val bitmap = Bitmap.wrapHardwareBuffer(hardwareBuffer, null) ?: run {
            hardwareBuffer.close()
            image.close()
            throw RuntimeException("Create Bitmap Failed")
        }
        hardwareBuffer.close()
        image.close()
        return bitmap
    }

    override fun rotateHue(radian: Float, outputIndex: Int): Bitmap {
        synchronized(mLock) {
            val params = mParams ?: throw RuntimeException("Not configured!")

            // Set HUE rotation matrix (RGB->HSV * HUE rotation * HSV->RGB) in column-major order,
            // matching Vulkan, GLSL, and WebGPU ColorMatrix shaders.
            val cosVal = cos(radian.toDouble()).toFloat()
            val sinVal = sin(radian.toDouble()).toFloat()
            // Column 0
            mColorMatrixValues[0] = 0.299f + 0.701f * cosVal + 0.168f * sinVal
            mColorMatrixValues[1] = 0.299f - 0.299f * cosVal - 0.328f * sinVal
            mColorMatrixValues[2] = 0.299f - 0.300f * cosVal + 1.250f * sinVal
            // Column 1
            mColorMatrixValues[3] = 0.587f - 0.587f * cosVal + 0.330f * sinVal
            mColorMatrixValues[4] = 0.587f + 0.413f * cosVal + 0.035f * sinVal
            mColorMatrixValues[5] = 0.587f - 0.588f * cosVal - 1.050f * sinVal
            // Column 2
            mColorMatrixValues[6] = 0.114f - 0.114f * cosVal - 0.497f * sinVal
            mColorMatrixValues[7] = 0.114f - 0.114f * cosVal + 0.292f * sinVal
            mColorMatrixValues[8] = 0.114f + 0.886f * cosVal - 0.203f * sinVal

            mRotateHueShader.setFloatUniform("colorMatrix", mColorMatrixValues)
            val hueEffect = RenderEffect.createRuntimeShaderEffect(mRotateHueShader, "inputImage")
            return applyEffect(params, hueEffect)
        }
    }

    override fun blur(radius: Float, outputIndex: Int): Bitmap {
        if (radius < 1.0f || radius > 25.0f) {
            throw RuntimeException("Invalid radius $radius, must be within [1.0, 25.0]")
        }

        synchronized(mLock) {
            val params = mParams ?: throw RuntimeException("Not configured!")

            // Calculate normalized 1D Gaussian kernel centered at MAX_BLUR_RADIUS (index 25).
            val e = 2.718281828459045f
            val pi = 3.1415926535897932f
            val sigma = 0.4f * radius + 0.6f
            val coeff1 = (1.0 / (sqrt(2.0 * pi) * sigma)).toFloat()
            val coeff2 = -1.0f / (2.0f * sigma * sigma)
            val iRadius = ceil(radius.toDouble()).toInt()

            mBlurKernelValues.fill(0.0f)
            var normalizeFactor = 0.0f
            for (r in -iRadius..iRadius) {
                val value = coeff1 * e.pow(coeff2 * (r * r).toFloat())
                mBlurKernelValues[r + MAX_BLUR_RADIUS] = value
                normalizeFactor += value
            }
            val invNormalize = 1.0f / normalizeFactor
            for (r in -iRadius..iRadius) {
                mBlurKernelValues[r + MAX_BLUR_RADIUS] *= invNormalize
            }

            mBlurHorizontalShader.setFloatUniform("kernel", mBlurKernelValues)
            mBlurHorizontalShader.setIntUniform("radius", iRadius)
            mBlurVerticalShader.setFloatUniform("kernel", mBlurKernelValues)
            mBlurVerticalShader.setIntUniform("radius", iRadius)

            // Chain horizontal then vertical passes via RenderEffect so Skia renders the
            // horizontal pass into an intermediate GPU surface before sampling vertically.
            val horizontalEffect =
                RenderEffect.createRuntimeShaderEffect(mBlurHorizontalShader, "inputImage")
            val verticalEffect =
                RenderEffect.createRuntimeShaderEffect(mBlurVerticalShader, "inputImage")
            val chainedBlurEffect = RenderEffect.createChainEffect(verticalEffect, horizontalEffect)
            return applyEffect(params, chainedBlurEffect)
        }
    }

    override fun cleanup() {
        synchronized(mLock) {
            cleanupLocked()
        }
    }

    private fun cleanupLocked() {
        mParams?.let { params ->
            mParams = null
            params.imageReader.close()
            params.renderNode.discardDisplayList()
            params.hardwareRenderer.destroy()
        }
    }

    private class Params(val bitmap: Bitmap, numberOfOutputImages: Int) {
        @SuppressLint("WrongConstant")
        val imageReader: ImageReader = ImageReader.newInstance(
            bitmap.width,
            bitmap.height,
            PixelFormat.RGBA_8888,
            max(numberOfOutputImages + 1, 3),
            HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE or HardwareBuffer.USAGE_GPU_COLOR_OUTPUT,
        )
        val renderNode = RenderNode("AGSL")
        val hardwareRenderer = HardwareRenderer()

        init {
            hardwareRenderer.setSurface(imageReader.surface)
            hardwareRenderer.setContentRoot(renderNode)
            renderNode.setPosition(0, 0, imageReader.width, imageReader.height)
        }
    }
}
