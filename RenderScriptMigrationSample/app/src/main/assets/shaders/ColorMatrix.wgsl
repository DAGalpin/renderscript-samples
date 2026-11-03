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

struct ColorMatrixParams {
    colorMatrix: mat3x3f,
}

@group(0) @binding(0) var inputSampler: sampler;
@group(0) @binding(1) var inputImage: texture_2d<f32>;
@group(0) @binding(2) var outputImage: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: ColorMatrixParams;

@compute @workgroup_size(workgroupSizeX, workgroupSizeY, 1u)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(outputImage);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    let uv = (vec2f(global_id.xy) + vec2f(0.5)) / vec2f(dims);
    let inputPixel = textureSampleLevel(inputImage, inputSampler, uv, 0.0).rgb;
    let resultPixel = params.colorMatrix * inputPixel;
    textureStore(outputImage, global_id.xy, vec4f(resultPixel, 1.0));
}
