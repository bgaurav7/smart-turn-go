// Example that uses the utility to resolve (and download when needed) the ONNX Runtime lib
// and models, then passes them into the SDK.
// Run from repo root: go run ./examples/utility
// Models and ONNX Runtime lib are downloaded into models/ if not present.
package main

import (
	"fmt"
	"os"

	"github.com/cortexswarm/smart-turn-go"
	"github.com/cortexswarm/smart-turn-go/examples/utility/resolver"
)

func main() {
	// Resolve (download when needed) models and ONNX Runtime lib into models/.
	sileroPath, err := resolver.ResolveSileroVAD(resolver.ModelsDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "resolve Silero VAD: %v\n", err)
		os.Exit(1)
	}
	smartTurnPath, err := resolver.ResolveSmartTurn(resolver.ModelsDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "resolve Smart-Turn: %v\n", err)
		os.Exit(1)
	}
	onnxLibPath, err := resolver.ResolveONNXRuntimeLibWithDownload(resolver.ModelsDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "resolve ONNX Runtime lib: %v\n", err)
		os.Exit(1)
	}
	if onnxLibPath == "" {
		fmt.Fprintf(os.Stderr, "ONNX Runtime lib not found for this platform and no download URL; set ONNXRUNTIME_SHARED_LIBRARY_PATH or place lib in models/\n")
		os.Exit(1)
	}

	cfg := smartturn.Config{
		SampleRate:              16000,
		ChunkSize:               512,
		VadThreshold:            0.75,
		VadPreSpeechMs:          200,
		VadStopMs:               800,
		TurnMaxDurationSeconds:  600,
		TurnSegmentEmitMs:       1000,
		TurnThreshold:           0.9,
		TurnTimeoutMs:           1000,
		SileroVADModelPath:      sileroPath,
		SmartTurnModelPath:      smartTurnPath,
		ONNXRuntimeLibPath:      onnxLibPath,
	}
	cb := smartturn.Callbacks{
		OnListeningStarted: func() { fmt.Println("[event] listening started") },
		OnListeningStopped: func() { fmt.Println("[event] listening stopped") },
		OnSpeechStart:      func() { fmt.Println("[event] speech start") },
		OnSpeechEnd:        func() { fmt.Println("[event] speech end") },
		OnTurnPrediction:   func(complete bool, prob float32) { fmt.Printf("[event] turn complete=%v prob=%.3f\n", complete, prob) },
		OnError:            func(err error) { fmt.Printf("[error] %v\n", err) },
	}

	engine, err := smartturn.New(cfg, cb)
	if err != nil {
		fmt.Fprintf(os.Stderr, "New: %v\n", err)
		os.Exit(1)
	}
	defer engine.Close()

	// Minimal run: just start/stop to show SDK accepts lib from config.
	engine.Start()
	engine.Stop()
	fmt.Println("ok (lib path was passed into SDK via Config.ONNXRuntimeLibPath)")
}
