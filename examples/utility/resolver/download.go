// Download helpers: fetch files from URLs into DataDir when not present.
package resolver

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
)

const (
	// URLs for ONNX Runtime shared libraries (yalue/onnxruntime_go test_data).
	urlONNXRuntimeBase = "https://github.com/yalue/onnxruntime_go/raw/refs/heads/master/test_data"
	// URLs for models.
	urlSileroVAD  = "https://github.com/snakers4/silero-vad/raw/refs/heads/master/src/silero_vad/data/silero_vad.onnx"
	urlSmartTurn  = "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
	sileroVADName = "silero_vad.onnx"
	smartTurnName = "smart-turn-v3.2-cpu.onnx"
)

// onnxRuntimeURL returns the download URL for the current GOOS/GOARCH, or "" if not supported.
func onnxRuntimeURL() string {
	m := map[string]string{
		"windows_amd64": urlONNXRuntimeBase + "/onnxruntime.dll",
		"darwin_amd64":  urlONNXRuntimeBase + "/onnxruntime_amd64.dylib",
		"darwin_arm64":  urlONNXRuntimeBase + "/onnxruntime_arm64.dylib",
		"linux_arm64":   urlONNXRuntimeBase + "/onnxruntime_arm64.so",
	}
	return m[runtime.GOOS+"_"+runtime.GOARCH]
}

// downloadFile fetches url into destDir with filename destName. Skips if file already exists.
// Uses a temp file and rename for atomic write.
func downloadFile(url, destDir, destName string) (path string, err error) {
	path = filepath.Join(destDir, destName)
	if pathExists(path) {
		return path, nil
	}
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return "", fmt.Errorf("mkdir %s: %w", destDir, err)
	}
	resp, err := http.Get(url)
	if err != nil {
		return "", fmt.Errorf("GET %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return "", fmt.Errorf("create %s: %w", tmpPath, err)
	}
	n, err := io.Copy(f, resp.Body)
	_ = f.Close()
	if err != nil {
		_ = os.Remove(tmpPath)
		return "", fmt.Errorf("write %s: %w", tmpPath, err)
	}
	if n == 0 {
		_ = os.Remove(tmpPath)
		return "", fmt.Errorf("empty response from %s", url)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return "", fmt.Errorf("rename to %s: %w", path, err)
	}
	return path, nil
}

// ResolveSileroVAD ensures silero_vad.onnx exists in dir (e.g. models/), downloading from Silero repo if missing.
// Returns the absolute path to the file.
func ResolveSileroVAD(dir string) (string, error) {
	path, err := downloadFile(urlSileroVAD, dir, sileroVADName)
	if err != nil {
		return "", err
	}
	return filepath.Abs(path)
}

// ResolveSmartTurn ensures smart-turn-v3.2-cpu.onnx exists in dir (e.g. models/), downloading from Hugging Face if missing.
// Returns the absolute path to the file.
func ResolveSmartTurn(dir string) (string, error) {
	path, err := downloadFile(urlSmartTurn, dir, smartTurnName)
	if err != nil {
		return "", err
	}
	return filepath.Abs(path)
}

// ResolveONNXRuntimeLibWithDownload ensures the ONNX Runtime shared library exists in dir (e.g. models/) for the
// current platform, downloading from yalue/onnxruntime_go test_data if missing. If this platform has no download
// URL, falls back to ResolveONNXRuntimeLib() (path-only). Returns the path to the library, or "" if not found.
func ResolveONNXRuntimeLibWithDownload(dir string) (string, error) {
	url := onnxRuntimeURL()
	if url == "" {
		// No download URL for this platform; use path-only resolution.
		return ResolveONNXRuntimeLib(), nil
	}
	name := filepath.Base(url)
	path, err := downloadFile(url, dir, name)
	if err != nil {
		return "", err
	}
	return filepath.Abs(path)
}
