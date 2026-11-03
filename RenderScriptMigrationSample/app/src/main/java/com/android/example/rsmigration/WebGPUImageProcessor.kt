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
import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.hardware.HardwareBuffer
import android.util.Log
import androidx.core.graphics.createBitmap
import androidx.webgpu.AddressMode
import androidx.webgpu.BackendType
import androidx.webgpu.BufferUsage
import androidx.webgpu.DeviceLostCallback
import androidx.webgpu.DeviceLostReason
import androidx.webgpu.ExperimentalWebGpuApi
import androidx.webgpu.FeatureName
import androidx.webgpu.FilterMode
import androidx.webgpu.GPU
import androidx.webgpu.GPUAdapter
import androidx.webgpu.GPUBindGroup
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUComputePipeline
import androidx.webgpu.GPUComputePipelineDescriptor
import androidx.webgpu.GPUComputeState
import androidx.webgpu.GPUConstantEntry
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPUDeviceDescriptor
import androidx.webgpu.GPUExtent3D
import androidx.webgpu.GPUHardwareBufferTexture
import androidx.webgpu.GPUInstance
import androidx.webgpu.GPUInstanceDescriptor
import androidx.webgpu.GPULimits
import androidx.webgpu.GPUQueue
import androidx.webgpu.GPURequestAdapterOptions
import androidx.webgpu.GPURequestCallback
import androidx.webgpu.GPUSampler
import androidx.webgpu.GPUSamplerDescriptor
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.GPUTexelCopyBufferInfo
import androidx.webgpu.GPUTexelCopyBufferLayout
import androidx.webgpu.GPUTexelCopyTextureInfo
import androidx.webgpu.GPUTexture
import androidx.webgpu.GPUTextureDescriptor
import androidx.webgpu.GPUTextureView
import androidx.webgpu.MapMode
import androidx.webgpu.MipmapFilterMode
import androidx.webgpu.TextureFormat
import androidx.webgpu.TextureUsage
import androidx.webgpu.UncapturedErrorCallback
import androidx.webgpu.WebGpuRuntimeException
import androidx.webgpu.helper.GPUAndroidHardwareBufferUtil
import androidx.webgpu.helper.initLibrary
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.LockSupport
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.runBlocking

private const val TAG = "WebGPUImageProcessor"

// ColorMatrixParams: mat3x3f in WGSL uniform layout uses 3 columns * 16-byte stride = 48 bytes.
private const val COLOR_MATRIX_UNIFORM_BYTES = 3 * 4 * Float.SIZE_BYTES

// BlurParams: array<vec4f, 13> (208 bytes) + radius: i32 + 12 bytes struct alignment = 224 bytes.
private const val BLUR_KERNEL_FLOATS = 52
private const val BLUR_RADIUS_OFFSET_BYTES = BLUR_KERNEL_FLOATS * Float.SIZE_BYTES
private const val BLUR_UNIFORM_BYTES = BLUR_RADIUS_OFFSET_BYTES + 4 * Int.SIZE_BYTES

private const val WEBGPU_BYTES_PER_ROW_ALIGNMENT = 256
private const val SYNC_FENCE_TIMEOUT_MS = 3000L
private const val EVENT_POLL_BACKOFF_NS = 50_000L

@OptIn(ExperimentalWebGpuApi::class)
class WebGPUImageProcessor(context: Context) : ImageProcessor {
    override val name = "WebGPU"

    private val mAssetManager: AssetManager = context.assets
    private val mClosed = AtomicBoolean(false)

    // Dawn is not thread-safe across threads; execute all WebGPU JNI calls on a dedicated thread.
    private val mWebGpuDispatcher =
        Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "WebGPU-ImageProcessor").apply { isDaemon = true }
        }.asCoroutineDispatcher()

    private lateinit var mInstance: GPUInstance
    private lateinit var mAdapter: GPUAdapter
    private lateinit var mDevice: GPUDevice
    private lateinit var mQueue: GPUQueue
    private lateinit var mSampler: GPUSampler

    private var mWorkGroupSize: Int = 8
    private var mSupportsZeroCopyHardwareBuffer: Boolean = false

    // Compute pipelines and uniform buffers mirroring Vulkan ImageProcessor
    private val mRotateHueUniformData: ByteBuffer =
        ByteBuffer.allocateDirect(COLOR_MATRIX_UNIFORM_BYTES).order(ByteOrder.nativeOrder())
    private lateinit var mRotateHueUniformBuffer: GPUBuffer
    private lateinit var mRotateHuePipeline: GPUComputePipeline

    private val mBlurKernelWeights = FloatArray(BLUR_KERNEL_FLOATS)
    private val mBlurUniformData: ByteBuffer =
        ByteBuffer.allocateDirect(BLUR_UNIFORM_BYTES).order(ByteOrder.nativeOrder())
    private lateinit var mBlurUniformBuffer: GPUBuffer
    private lateinit var mBlurHorizontalPipeline: GPUComputePipeline
    private lateinit var mBlurVerticalPipeline: GPUComputePipeline

    // Per-image resources configured in configureInputAndOutput
    private var mImageWidth: Int = 0
    private var mImageHeight: Int = 0
    private var mGroupCountX: Int = 0
    private var mGroupCountY: Int = 0
    private var mImageExtent: GPUExtent3D = GPUExtent3D(width = 1, height = 1)

    private var mInputTexture: GPUTexture? = null
    private var mInputTextureView: GPUTextureView? = null
    private var mTempTexture: GPUTexture? = null
    private var mTempTextureView: GPUTextureView? = null
    private var mStagingOutputTexture: GPUTexture? = null
    private var mStagingOutputTextureView: GPUTextureView? = null
    private var mStagingCopySrc: GPUTexelCopyTextureInfo? = null

    private var mRotateHueBindGroup: GPUBindGroup? = null
    private var mBlurHorizontalBindGroup: GPUBindGroup? = null
    private var mBlurVerticalBindGroup: GPUBindGroup? = null

    // Output resources (zero-copy HardwareBuffer textures when supported, or GPU readback buffer)
    private var mUseHardwareBufferOutputs: Boolean = false
    private var mOutputHardwareBuffers: Array<HardwareBuffer>? = null
    private var mOutputHardwareTextures: Array<GPUHardwareBufferTexture>? = null
    private var mOutputCopyDsts: Array<GPUTexelCopyTextureInfo>? = null
    private var mReadbackBuffer: GPUBuffer? = null
    private var mReadbackBytesPerRow: Int = 0
    private var mUnpaddedReadbackData: ByteBuffer? = null
    private lateinit var mOutputImages: Array<Bitmap>

    init {
        runBlocking(mWebGpuDispatcher) {
            initialize()
        }
    }

    private fun initialize() {
        initLibrary()
        val inlineExecutor = Executor(Runnable::run)

        mInstance = GPU.createInstance(GPUInstanceDescriptor())
        mAdapter = try {
            awaitGpuRequest(mInstance) { executor, callback ->
                mInstance.requestAdapter(
                    executor,
                    GPURequestAdapterOptions(backendType = BackendType.Vulkan),
                    callback,
                )
            }
        } catch (_: Exception) {
            awaitGpuRequest(mInstance) { executor, callback ->
                mInstance.requestAdapter(executor, GPURequestAdapterOptions(), callback)
            }
        }

        val adapterLimits = mAdapter.getLimits()
        mWorkGroupSize = chooseWorkGroupSize(adapterLimits)

        val hasSharedTextureAHB =
            mAdapter.hasFeature(FeatureName.SharedTextureMemoryAHardwareBuffer)
        val hasSharedFenceSyncFD = mAdapter.hasFeature(FeatureName.SharedFenceSyncFD)
        mSupportsZeroCopyHardwareBuffer = hasSharedTextureAHB && hasSharedFenceSyncFD

        val requiredFeatures = when {
            hasSharedTextureAHB && hasSharedFenceSyncFD -> intArrayOf(
                FeatureName.SharedTextureMemoryAHardwareBuffer,
                FeatureName.SharedFenceSyncFD,
            )
            hasSharedTextureAHB -> intArrayOf(
                FeatureName.SharedTextureMemoryAHardwareBuffer,
            )
            hasSharedFenceSyncFD -> intArrayOf(
                FeatureName.SharedFenceSyncFD,
            )
            else -> intArrayOf()
        }

        val requiredLimits = GPULimits(
            maxComputeWorkgroupSizeX = adapterLimits.maxComputeWorkgroupSizeX,
            maxComputeWorkgroupSizeY = adapterLimits.maxComputeWorkgroupSizeY,
            maxComputeInvocationsPerWorkgroup = adapterLimits.maxComputeInvocationsPerWorkgroup,
        )

        // GPUDeviceDescriptor annotates the IntArray parameter itself with @FeatureName.Type.
        @SuppressLint("WrongConstant")
        val deviceDescriptor = GPUDeviceDescriptor(
            deviceLostCallbackExecutor = inlineExecutor,
            uncapturedErrorCallbackExecutor = inlineExecutor,
            label = "RenderScriptMigrationWebGPUDevice",
            requiredFeatures = requiredFeatures,
            requiredLimits = requiredLimits,
            deviceLostCallback = DeviceLostCallback { _, reason, message ->
                if (reason != DeviceLostReason.Destroyed &&
                    reason != DeviceLostReason.CallbackCancelled
                ) {
                    Log.e(
                        TAG,
                        "WebGPU device lost (${DeviceLostReason.toString(reason)}): $message"
                    )
                }
            },
            uncapturedErrorCallback = UncapturedErrorCallback { _, type, message ->
                Log.e(TAG, "Uncaptured WebGPU error ($type): $message")
                throw WebGpuRuntimeException.create(type, message)
            },
        )

        mDevice = awaitGpuRequest(mInstance) { executor, callback ->
            mAdapter.requestDevice(executor, deviceDescriptor, callback)
        }
        mQueue = mDevice.getQueue()

        mSampler = mDevice.createSampler(
            GPUSamplerDescriptor(
                label = "ImageProcessorSampler",
                addressModeU = AddressMode.ClampToEdge,
                addressModeV = AddressMode.ClampToEdge,
                addressModeW = AddressMode.ClampToEdge,
                magFilter = FilterMode.Nearest,
                minFilter = FilterMode.Nearest,
                mipmapFilter = MipmapFilterMode.Nearest,
            )
        )

        mRotateHueUniformBuffer = mDevice.createBuffer(
            GPUBufferDescriptor(
                label = "RotateHueUniformBuffer",
                usage = BufferUsage.Uniform or BufferUsage.CopyDst,
                size = COLOR_MATRIX_UNIFORM_BYTES.toLong(),
            )
        )
        mBlurUniformBuffer = mDevice.createBuffer(
            GPUBufferDescriptor(
                label = "BlurUniformBuffer",
                usage = BufferUsage.Uniform or BufferUsage.CopyDst,
                size = BLUR_UNIFORM_BYTES.toLong(),
            )
        )

        mRotateHuePipeline = createComputePipelineFromAsset(
            assetPath = "shaders/ColorMatrix.wgsl",
            label = "RotateHuePipeline",
        )
        mBlurHorizontalPipeline = createComputePipelineFromAsset(
            assetPath = "shaders/BlurHorizontal.wgsl",
            label = "BlurHorizontalPipeline",
        )
        mBlurVerticalPipeline = createComputePipelineFromAsset(
            assetPath = "shaders/BlurVertical.wgsl",
            label = "BlurVerticalPipeline",
        )
    }

    // Mirrors chooseWorkGroupSize in VulkanContext.cpp.
    private fun chooseWorkGroupSize(limits: GPULimits): Int {
        var size = 64
        size = min(size, limits.maxComputeWorkgroupSizeX)
        size = min(size, limits.maxComputeWorkgroupSizeY)
        size = min(size, sqrt(limits.maxComputeInvocationsPerWorkgroup.toDouble()).toInt())
        size = size and 3.inv()
        return max(size, 4)
    }

    private fun createComputePipelineFromAsset(
        assetPath: String,
        label: String,
    ): GPUComputePipeline {
        val wgslSource = mAssetManager.open(assetPath).bufferedReader().use { it.readText() }
        val workgroupConstants = arrayOf(
            GPUConstantEntry("workgroupSizeX", mWorkGroupSize.toDouble()),
            GPUConstantEntry("workgroupSizeY", mWorkGroupSize.toDouble()),
        )
        return mDevice.createShaderModule(
            GPUShaderModuleDescriptor(
                label = "$label-Shader",
                shaderSourceWGSL = GPUShaderSourceWGSL(wgslSource),
            )
        ).use { shaderModule ->
            mDevice.createComputePipeline(
                GPUComputePipelineDescriptor(
                    label = label,
                    compute = GPUComputeState(
                        module = shaderModule,
                        entryPoint = "main",
                        constants = workgroupConstants,
                    ),
                )
            )
        }
    }

    override fun configureInputAndOutput(inputImage: Bitmap, numberOfOutputImages: Int) {
        if (numberOfOutputImages <= 0) {
            throw RuntimeException("Invalid number of output images: $numberOfOutputImages")
        }
        val argbBitmap = if (inputImage.config == Bitmap.Config.ARGB_8888) {
            inputImage
        } else {
            inputImage.copy(Bitmap.Config.ARGB_8888, false)
        }

        val width = argbBitmap.width
        val height = argbBitmap.height
        val pixelBytes = ByteBuffer.allocateDirect(width * height * Int.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
        argbBitmap.copyPixelsToBuffer(pixelBytes)
        pixelBytes.rewind()

        runBlocking(mWebGpuDispatcher) {
            releaseImageResources()

            mImageWidth = width
            mImageHeight = height
            mGroupCountX = ceilOfDiv(width, mWorkGroupSize)
            mGroupCountY = ceilOfDiv(height, mWorkGroupSize)
            mImageExtent = GPUExtent3D(width = width, height = height)

            // Create input texture and upload input bitmap pixels.
            val inputTexture = mDevice.createTexture(
                GPUTextureDescriptor(
                    label = "InputImage",
                    usage = TextureUsage.TextureBinding or TextureUsage.CopyDst,
                    size = mImageExtent,
                    format = TextureFormat.RGBA8Unorm,
                )
            )
            mQueue.writeTexture(
                destination = GPUTexelCopyTextureInfo(texture = inputTexture),
                data = pixelBytes,
                writeSize = mImageExtent,
                dataLayout = GPUTexelCopyBufferLayout(
                    offset = 0,
                    bytesPerRow = width * Int.SIZE_BYTES,
                    rowsPerImage = height,
                ),
            )
            mInputTexture = inputTexture
            val inputView = inputTexture.createView()
            mInputTextureView = inputView

            // Create intermediate image for two-pass blur (written in pass 1, sampled in pass 2).
            val tempTexture = mDevice.createTexture(
                GPUTextureDescriptor(
                    label = "TempBlurImage",
                    usage = TextureUsage.StorageBinding or TextureUsage.TextureBinding,
                    size = mImageExtent,
                    format = TextureFormat.RGBA8Unorm,
                )
            )
            mTempTexture = tempTexture
            val tempView = tempTexture.createView()
            mTempTextureView = tempView

            // Create staging output image (written by compute shaders, copied to output image).
            val stagingTexture = mDevice.createTexture(
                GPUTextureDescriptor(
                    label = "StagingOutputImage",
                    usage = TextureUsage.StorageBinding or TextureUsage.CopySrc,
                    size = mImageExtent,
                    format = TextureFormat.RGBA8Unorm,
                )
            )
            mStagingOutputTexture = stagingTexture
            val stagingView = stagingTexture.createView()
            mStagingOutputTextureView = stagingView
            mStagingCopySrc = GPUTexelCopyTextureInfo(texture = stagingTexture)

            // Pre-build bind groups for zero-allocation dispatch in rotateHue and blur.
            mRotateHueBindGroup = createStageBindGroup(
                pipeline = mRotateHuePipeline,
                label = "RotateHueBindGroup",
                inputView = inputView,
                outputView = stagingView,
                uniformBuffer = mRotateHueUniformBuffer,
            )
            mBlurHorizontalBindGroup = createStageBindGroup(
                pipeline = mBlurHorizontalPipeline,
                label = "BlurHorizontalBindGroup",
                inputView = inputView,
                outputView = tempView,
                uniformBuffer = mBlurUniformBuffer,
            )
            mBlurVerticalBindGroup = createStageBindGroup(
                pipeline = mBlurVerticalPipeline,
                label = "BlurVerticalBindGroup",
                inputView = tempView,
                outputView = stagingView,
                uniformBuffer = mBlurUniformBuffer,
            )

            // Allocate HardwareBuffer-backed zero-copy outputs when supported (matching Vulkan),
            // with a mapped-buffer readback fallback for environments without AHB sync fences.
            val configuredHardwareOutputs =
                mSupportsZeroCopyHardwareBuffer &&
                    configureHardwareBufferOutputs(width, height, numberOfOutputImages)
            mUseHardwareBufferOutputs = configuredHardwareOutputs
            if (!configuredHardwareOutputs) {
                configureReadbackOutputs(width, height, numberOfOutputImages)
            }
            mInstance.processEvents()
        }
    }

    private fun createStageBindGroup(
        pipeline: GPUComputePipeline,
        label: String,
        inputView: GPUTextureView,
        outputView: GPUTextureView,
        uniformBuffer: GPUBuffer,
    ): GPUBindGroup {
        return pipeline.getBindGroupLayout(0).use { layout ->
            mDevice.createBindGroup(
                GPUBindGroupDescriptor(
                    layout = layout,
                    label = label,
                    entries = arrayOf(
                        GPUBindGroupEntry(binding = 0, sampler = mSampler),
                        GPUBindGroupEntry(binding = 1, textureView = inputView),
                        GPUBindGroupEntry(binding = 2, textureView = outputView),
                        GPUBindGroupEntry(binding = 3, buffer = uniformBuffer),
                    ),
                )
            )
        }
    }

    private fun configureHardwareBufferOutputs(
        width: Int,
        height: Int,
        numberOfOutputImages: Int,
    ): Boolean {
        val usage = HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE or HardwareBuffer.USAGE_GPU_COLOR_OUTPUT
        if (!HardwareBuffer.isSupported(width, height, HardwareBuffer.RGBA_8888, 1, usage)) {
            return false
        }

        val allocatedBuffers = mutableListOf<HardwareBuffer>()
        val importedTextures = mutableListOf<GPUHardwareBufferTexture>()
        val wrappedBitmaps = mutableListOf<Bitmap>()
        return try {
            for (i in 0 until numberOfOutputImages) {
                val hardwareBuffer =
                    HardwareBuffer.create(width, height, HardwareBuffer.RGBA_8888, 1, usage)
                allocatedBuffers.add(hardwareBuffer)

                val hwTexture = GPUAndroidHardwareBufferUtil.createTexture(
                    mDevice,
                    hardwareBuffer,
                    TextureUsage.CopyDst,
                )
                importedTextures.add(hwTexture)

                val outputBitmap = Bitmap.wrapHardwareBuffer(hardwareBuffer, null)
                    ?: throw IllegalStateException("Failed to wrapHardwareBuffer at index $i")
                wrappedBitmaps.add(outputBitmap)
            }
            mOutputHardwareBuffers = allocatedBuffers.toTypedArray()
            mOutputHardwareTextures = importedTextures.toTypedArray()
            mOutputCopyDsts = Array(numberOfOutputImages) { i ->
                GPUTexelCopyTextureInfo(texture = importedTextures[i].texture)
            }
            mOutputImages = wrappedBitmaps.toTypedArray()
            true
        } catch (e: Exception) {
            Log.w(TAG, "Falling back to GPUBuffer readback outputs: ${e.message}")
            importedTextures.forEach { runCatching { it.close() } }
            allocatedBuffers.forEach { runCatching { it.close() } }
            false
        }
    }

    private fun configureReadbackOutputs(
        width: Int,
        height: Int,
        numberOfOutputImages: Int,
    ) {
        val unpaddedBytesPerRow = width * Int.SIZE_BYTES
        val alignedBytesPerRow = roundUpToMultiple(
            unpaddedBytesPerRow,
            WEBGPU_BYTES_PER_ROW_ALIGNMENT,
        )
        mReadbackBytesPerRow = alignedBytesPerRow
        mReadbackBuffer = mDevice.createBuffer(
            GPUBufferDescriptor(
                label = "OutputReadbackBuffer",
                usage = BufferUsage.CopyDst or BufferUsage.MapRead,
                size = alignedBytesPerRow.toLong() * height.toLong(),
            )
        )
        mUnpaddedReadbackData = if (alignedBytesPerRow != unpaddedBytesPerRow) {
            ByteBuffer.allocateDirect(unpaddedBytesPerRow * height).order(ByteOrder.nativeOrder())
        } else {
            null
        }
        mOutputImages = Array(numberOfOutputImages) {
            createBitmap(width, height, Bitmap.Config.ARGB_8888)
        }
    }

    override fun rotateHue(radian: Float, outputIndex: Int): Bitmap {
        // Set HUE rotation matrix (RGB->HSV * HUE rotation * HSV->RGB), column-major with 16-byte
        // column stride matching Vulkan ImageProcessor::rotateHue and ColorMatrix.comp.
        val cosVal = cos(radian.toDouble()).toFloat()
        val sinVal = sin(radian.toDouble()).toFloat()
        mRotateHueUniformData.clear()
        // Column 0
        mRotateHueUniformData.putFloat(0.299f + 0.701f * cosVal + 0.168f * sinVal)
        mRotateHueUniformData.putFloat(0.299f - 0.299f * cosVal - 0.328f * sinVal)
        mRotateHueUniformData.putFloat(0.299f - 0.300f * cosVal + 1.250f * sinVal)
        mRotateHueUniformData.putFloat(0.0f)
        // Column 1
        mRotateHueUniformData.putFloat(0.587f - 0.587f * cosVal + 0.330f * sinVal)
        mRotateHueUniformData.putFloat(0.587f + 0.413f * cosVal + 0.035f * sinVal)
        mRotateHueUniformData.putFloat(0.587f - 0.588f * cosVal - 1.050f * sinVal)
        mRotateHueUniformData.putFloat(0.0f)
        // Column 2
        mRotateHueUniformData.putFloat(0.114f - 0.114f * cosVal - 0.497f * sinVal)
        mRotateHueUniformData.putFloat(0.114f - 0.114f * cosVal + 0.292f * sinVal)
        mRotateHueUniformData.putFloat(0.114f + 0.886f * cosVal - 0.203f * sinVal)
        mRotateHueUniformData.putFloat(0.0f)
        mRotateHueUniformData.rewind()

        runBlocking(mWebGpuDispatcher) {
            mQueue.writeBuffer(mRotateHueUniformBuffer, 0L, mRotateHueUniformData)
            executeAndPresentOutput(outputIndex) { encoder ->
                encoder.beginComputePass().use { pass ->
                    pass.setPipeline(mRotateHuePipeline)
                    pass.setBindGroup(0, checkNotNull(mRotateHueBindGroup))
                    pass.dispatchWorkgroups(mGroupCountX, mGroupCountY, 1)
                    pass.end()
                }
            }
        }
        return mOutputImages[outputIndex]
    }

    override fun blur(radius: Float, outputIndex: Int): Bitmap {
        if (radius < 1.0f || radius > 25.0f) {
            throw RuntimeException("Invalid radius $radius, must be within [1.0, 25.0]")
        }

        // Calculate Gaussian kernel weights matching Vulkan ImageProcessor::blur.
        val e = 2.718281828459045f
        val pi = 3.1415926535897932f
        val sigma = 0.4f * radius + 0.6f
        val coeff1 = (1.0 / (sqrt(2.0 * pi) * sigma)).toFloat()
        val coeff2 = -1.0f / (2.0f * sigma * sigma)
        val iRadius = ceil(radius.toDouble()).toInt()
        var normalizeFactor = 0.0f
        for (r in -iRadius..iRadius) {
            val value = coeff1 * e.pow(coeff2 * (r * r).toFloat())
            mBlurKernelWeights[r + iRadius] = value
            normalizeFactor += value
        }
        normalizeFactor = 1.0f / normalizeFactor

        mBlurUniformData.clear()
        val kernelLength = 2 * iRadius + 1
        for (i in 0 until BLUR_KERNEL_FLOATS) {
            val weight = if (i < kernelLength) mBlurKernelWeights[i] * normalizeFactor else 0.0f
            mBlurUniformData.putFloat(weight)
        }
        mBlurUniformData.putInt(iRadius)
        mBlurUniformData.putInt(0)
        mBlurUniformData.putInt(0)
        mBlurUniformData.putInt(0)
        mBlurUniformData.rewind()

        runBlocking(mWebGpuDispatcher) {
            mQueue.writeBuffer(mBlurUniformBuffer, 0L, mBlurUniformData)
            executeAndPresentOutput(outputIndex) { encoder ->
                // First pass: horizontal Gaussian blur (input -> temp).
                encoder.beginComputePass().use { pass ->
                    pass.setPipeline(mBlurHorizontalPipeline)
                    pass.setBindGroup(0, checkNotNull(mBlurHorizontalBindGroup))
                    pass.dispatchWorkgroups(mGroupCountX, mGroupCountY, 1)
                    pass.end()
                }
                // Second pass: vertical Gaussian blur (temp -> staging output).
                encoder.beginComputePass().use { pass ->
                    pass.setPipeline(mBlurVerticalPipeline)
                    pass.setBindGroup(0, checkNotNull(mBlurVerticalBindGroup))
                    pass.dispatchWorkgroups(mGroupCountX, mGroupCountY, 1)
                    pass.end()
                }
            }
        }
        return mOutputImages[outputIndex]
    }

    private inline fun executeAndPresentOutput(
        outputIndex: Int,
        recordComputePasses: (androidx.webgpu.GPUCommandEncoder) -> Unit,
    ) {
        val stagingSrc = checkNotNull(mStagingCopySrc) { "configureInputAndOutput not called" }
        if (mUseHardwareBufferOutputs) {
            val hwTexture = checkNotNull(mOutputHardwareTextures)[outputIndex]
            val outputDst = checkNotNull(mOutputCopyDsts)[outputIndex]
            hwTexture.beginAccess(null)
            try {
                mDevice.createCommandEncoder().use { encoder ->
                    recordComputePasses(encoder)
                    encoder.copyTextureToTexture(stagingSrc, outputDst, mImageExtent)
                    encoder.finish().use { commandBuffer ->
                        mQueue.submit(arrayOf(commandBuffer))
                    }
                }
            } finally {
                val syncFence = hwTexture.endAccess()
                if (syncFence != null) {
                    syncFence.use { fence ->
                        if (!fence.await(SYNC_FENCE_TIMEOUT_MS)) {
                            throw RuntimeException("Timed out waiting for WebGPU output sync fence")
                        }
                    }
                } else {
                    awaitSubmittedWorkDone()
                }
            }
            mInstance.processEvents()
        } else {
            val readbackBuffer = checkNotNull(mReadbackBuffer)
            val totalReadbackBytes = mReadbackBytesPerRow.toLong() * mImageHeight.toLong()
            mDevice.createCommandEncoder().use { encoder ->
                recordComputePasses(encoder)
                encoder.copyTextureToBuffer(
                    source = stagingSrc,
                    destination = GPUTexelCopyBufferInfo(
                        buffer = readbackBuffer,
                        layout = GPUTexelCopyBufferLayout(
                            offset = 0,
                            bytesPerRow = mReadbackBytesPerRow,
                            rowsPerImage = mImageHeight,
                        ),
                    ),
                    copySize = mImageExtent,
                )
                encoder.finish().use { commandBuffer ->
                    mQueue.submit(arrayOf(commandBuffer))
                }
            }
            awaitGpuRequest(mInstance) { executor, callback ->
                readbackBuffer.mapAsync(MapMode.Read, 0L, totalReadbackBytes, executor, callback)
            }
            try {
                val mappedRange = readbackBuffer.getConstMappedRange(0L, totalReadbackBytes)
                val rowBytes = mImageWidth * Int.SIZE_BYTES
                if (mReadbackBytesPerRow == rowBytes) {
                    mappedRange.rewind()
                    mOutputImages[outputIndex].copyPixelsFromBuffer(mappedRange)
                } else {
                    val unpadded = checkNotNull(mUnpaddedReadbackData)
                    unpadded.clear()
                    val rowSlice = mappedRange.duplicate()
                    for (y in 0 until mImageHeight) {
                        val rowStart = y * mReadbackBytesPerRow
                        rowSlice.limit(rowStart + rowBytes)
                        rowSlice.position(rowStart)
                        unpadded.put(rowSlice)
                    }
                    unpadded.rewind()
                    mOutputImages[outputIndex].copyPixelsFromBuffer(unpadded)
                }
            } finally {
                readbackBuffer.unmap()
            }
        }
    }

    private fun awaitSubmittedWorkDone() {
        awaitGpuRequest(mInstance) { executor, callback ->
            mQueue.onSubmittedWorkDone(executor, callback)
        }
    }

    private fun <T> awaitGpuRequest(
        instance: GPUInstance,
        register: (Executor, GPURequestCallback<T>) -> Unit,
    ): T {
        var completed = false
        var resultValue: T? = null
        var failure: Exception? = null
        register(
            Executor(Runnable::run),
            object : GPURequestCallback<T> {
                override fun onResult(result: T) {
                    resultValue = result
                    completed = true
                }

                override fun onError(exception: Exception) {
                    failure = exception
                    completed = true
                }
            },
        )
        while (!completed) {
            instance.processEvents()
            if (!completed) {
                LockSupport.parkNanos(EVENT_POLL_BACKOFF_NS)
            }
        }
        failure?.let { throw it }
        @Suppress("UNCHECKED_CAST")
        return resultValue as T
    }

    private fun releaseImageResources() {
        mRotateHueBindGroup?.close()
        mRotateHueBindGroup = null
        mBlurHorizontalBindGroup?.close()
        mBlurHorizontalBindGroup = null
        mBlurVerticalBindGroup?.close()
        mBlurVerticalBindGroup = null

        mInputTextureView?.close()
        mInputTextureView = null
        mInputTexture?.destroy()
        mInputTexture?.close()
        mInputTexture = null

        mTempTextureView?.close()
        mTempTextureView = null
        mTempTexture?.destroy()
        mTempTexture?.close()
        mTempTexture = null

        mStagingOutputTextureView?.close()
        mStagingOutputTextureView = null
        mStagingOutputTexture?.destroy()
        mStagingOutputTexture?.close()
        mStagingOutputTexture = null
        mStagingCopySrc = null

        mOutputHardwareTextures?.forEach { it.close() }
        mOutputHardwareTextures = null
        mOutputHardwareBuffers?.forEach { it.close() }
        mOutputHardwareBuffers = null
        mOutputCopyDsts = null

        mReadbackBuffer?.destroy()
        mReadbackBuffer?.close()
        mReadbackBuffer = null
        mUnpaddedReadbackData = null
    }

    override fun cleanup() {
        if (mClosed.getAndSet(true)) return
        runBlocking(mWebGpuDispatcher) {
            releaseImageResources()
            if (::mRotateHuePipeline.isInitialized) mRotateHuePipeline.close()
            if (::mBlurHorizontalPipeline.isInitialized) mBlurHorizontalPipeline.close()
            if (::mBlurVerticalPipeline.isInitialized) mBlurVerticalPipeline.close()
            if (::mRotateHueUniformBuffer.isInitialized) {
                mRotateHueUniformBuffer.destroy()
                mRotateHueUniformBuffer.close()
            }
            if (::mBlurUniformBuffer.isInitialized) {
                mBlurUniformBuffer.destroy()
                mBlurUniformBuffer.close()
            }
            if (::mSampler.isInitialized) mSampler.close()
            if (::mQueue.isInitialized) mQueue.close()
            if (::mAdapter.isInitialized) mAdapter.close()
            if (::mInstance.isInitialized) mInstance.close()
        }
        mWebGpuDispatcher.close()
    }

    private fun ceilOfDiv(lhs: Int, rhs: Int): Int = (lhs + rhs - 1) / rhs

    private fun roundUpToMultiple(value: Int, alignment: Int): Int =
        (value + alignment - 1) and (alignment - 1).inv()
}
