/*
 * Copyright (C) 2023 The Android Open Source Project
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

#include <android/log.h>
#include <jni.h>
#include <GLES3/gl32.h>
#include <EGL/egl.h>

// our JNICallback struct.
static thread_local struct JNICallback {
    JavaVM *jvm;
    jmethodID mid;
    jobject obj;
} jniCallback;

static void GL_APIENTRY openGLMessageCallback(GLenum source, GLenum type, GLuint id,
                                   GLenum severity, GLsizei, const GLchar* message,
                                   const void* userParam)
{
    if (nullptr != userParam) {
        const JNICallback* callback = reinterpret_cast<const JNICallback*>(userParam);
        if (callback->jvm == nullptr || callback->obj == nullptr || callback->mid == nullptr) {
            return;
        }

        void* envPtr = nullptr;
        if (callback->jvm->GetEnv(&envPtr, JNI_VERSION_1_6) != JNI_OK || envPtr == nullptr) {
            return;
        }
        JNIEnv* env = static_cast<JNIEnv*>(envPtr);

        jstring jniMessageString = env->NewStringUTF(message != nullptr ? message : "");
        if (jniMessageString != nullptr) {
            env->CallVoidMethod(callback->obj, callback->mid,
                                static_cast<jint>(source),
                                static_cast<jint>(type),
                                static_cast<jint>(id),
                                static_cast<jint>(severity),
                                jniMessageString);
            env->DeleteLocalRef(jniMessageString);
        }
    }
}

// There's no way to do this in managed code, so here's something to help out those devs that
// want some more debug info.
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_android_example_rsmigration_GLSLImageProcessorKt_EnableDebugLogging(JNIEnv *env,
                                                                             jclass,
                                                                             jobject callback) {
    if (env != nullptr) {
        if (jniCallback.obj != nullptr) {
            env->DeleteGlobalRef(jniCallback.obj);
            jniCallback.obj = nullptr;
            jniCallback.mid = nullptr;
        }

        if (callback == nullptr) {
            glDebugMessageCallback(nullptr, nullptr);
            glDisable(GL_DEBUG_OUTPUT);
            return JNI_TRUE;
        }

        if (env->GetJavaVM(&jniCallback.jvm) != JNI_OK || jniCallback.jvm == nullptr) {
            return JNI_FALSE;
        }

        jclass cls = env->GetObjectClass(callback);
        if (cls == nullptr) {
            return JNI_FALSE;
        }
        jmethodID mid = env->GetMethodID(cls, "onMessage", "(IIIILjava/lang/String;)V");
        env->DeleteLocalRef(cls);
        if (mid == nullptr) {
            return JNI_FALSE;
        }

        jobject globalCallback = env->NewGlobalRef(callback);
        if (globalCallback == nullptr) {
            return JNI_FALSE;
        }

        // enable debug output
        glEnable(GL_DEBUG_OUTPUT);
        // call back on the same thread
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

        jniCallback.mid = mid;
        jniCallback.obj = globalCallback;
        glDebugMessageCallback(openGLMessageCallback, &jniCallback);
        return JNI_TRUE;
    }
    return JNI_FALSE;
}
