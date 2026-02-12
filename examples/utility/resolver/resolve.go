// Package resolver provides helpers to resolve the ONNX Runtime shared library
// when bundled with the app (data/ or lib/<platform>/). Use from examples or
// apps that bundle the runtime.
package resolver

import (
	"os"
	"path/filepath"
	"runtime"
)

func pathExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// BundledLibDir is the directory name under which platform-specific ONNX Runtime
// libraries are stored (e.g. lib/darwin_arm64/libonnxruntime.dylib).
const BundledLibDir = "lib"

// DataDir is the directory (e.g. data/) where ONNX models and optionally the
// runtime are stored. Runtime files may be named e.g. onnxruntime_arm64.dylib.
const DataDir = "data"

// ModelsDir is the directory where the resolver downloads models and the ONNX
// Runtime shared library (e.g. models/silero_vad.onnx, models/onnxruntime_arm64.dylib).
const ModelsDir = "models"

func bundledLibNames() []string {
	switch runtime.GOOS {
	case "darwin":
		return []string{"libonnxruntime.dylib"}
	case "windows":
		return []string{"onnxruntime.dll"}
	default:
		return []string{"libonnxruntime.so.1.23.2", "libonnxruntime.so"}
	}
}

func dataDirLibName() string {
	switch runtime.GOOS {
	case "darwin":
		return "onnxruntime_" + runtime.GOARCH + ".dylib"
	case "windows":
		return "onnxruntime.dll"
	default:
		return "onnxruntime_" + runtime.GOARCH + ".so"
	}
}

func bundledLibPlatform() string {
	return runtime.GOOS + "_" + runtime.GOARCH
}

func candidateBaseDirs() []string {
	cwd, err := os.Getwd()
	if err != nil {
		cwd = "."
	}
	exe, err := os.Executable()
	if err != nil {
		return []string{cwd}
	}
	exeDir := filepath.Dir(exe)
	if exeDir == cwd {
		return []string{cwd}
	}
	return []string{cwd, exeDir}
}

// ResolveONNXRuntimeLib returns the first path that exists. It checks (1) data/
// with platform-specific names (e.g. data/onnxruntime_arm64.dylib), then
// (2) lib/<platform>/ with standard names (e.g. lib/darwin_arm64/libonnxruntime.dylib).
// Returns empty string if no bundled lib is found; caller can use env or system lib.
func ResolveONNXRuntimeLib() string {
	platform := bundledLibPlatform()
	dataName := dataDirLibName()
	for _, base := range candidateBaseDirs() {
		if base == "" {
			continue
		}
		p := filepath.Join(base, DataDir, dataName)
		if pathExists(p) {
			return p
		}
	}
	for _, base := range candidateBaseDirs() {
		if base == "" {
			continue
		}
		for _, name := range bundledLibNames() {
			p := filepath.Join(base, BundledLibDir, platform, name)
			if pathExists(p) {
				return p
			}
		}
	}
	return ""
}
