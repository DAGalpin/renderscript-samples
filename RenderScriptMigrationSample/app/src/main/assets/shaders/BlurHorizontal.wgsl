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

override workgroupSizeX: u32 = 8u;
override workgroupSizeY: u32 = 8u;

struct BlurParams {
    // Uniform address space requires 16-byte array element stride.
    // Pack 52 kernel weights into 13 vec4f elements, matching BlurHorizontal.comp.
    kernel: array<vec4f, 13>,
    radius: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
}

@group(0) @binding(0) var inputSampler: sampler;
@group(0) @binding(1) var inputImage: texture_2d<f32>;
@group(0) @binding(2) var outputImage: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: BlurParams;

@compute @workgroup_size(workgroupSizeX, workgroupSizeY, 1u)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(outputImage);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let invDims = 1.0 / vec2f(dims);
    let sampleY = (f32(global_id.y) + 0.5) * invDims.y;
    var blurredRgb = vec3f(0.0, 0.0, 0.0);
    for (var r: i32 = -params.radius; r <= params.radius; r += 1) {
        // Out-of-bounds coordinates clamp automatically via AddressMode.ClampToEdge.
        let sampleX = (f32(i32(global_id.x) + r) + 0.5) * invDims.x;
        let pixel = textureSampleLevel(inputImage, inputSampler, vec2f(sampleX, sampleY), 0.0).rgb;
        let kernelIndex = r + params.radius;
        blurredRgb += params.kernel[kernelIndex / 4][kernelIndex % 4] * pixel;
    }
    textureStore(outputImage, global_id.xy, vec4f(blurredRgb, 1.0));
}
