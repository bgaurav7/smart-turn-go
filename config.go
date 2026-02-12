package smartturn

import (
	"errors"
	"os"
)

const (
	RequiredSampleRate = 16000
	RequiredChunkSize  = 512
)

// Config holds SDK configuration. All fields must be set; no silent defaults.
type Config struct {
	SampleRate   int     // must be 16000
	ChunkSize    int     // must be 512
	VadThreshold float32 // speech probability threshold (e.g. 0.5)

	// VAD behaviour and buffering.
	VadPreSpeechMs int     // ms of audio to keep before speech trigger (e.g. 200)
	VadStopMs      int     // ms of trailing silence to end VAD speech (e.g. 800)
	// TurnMaxDurationSeconds is a hard cap per turn in seconds (e.g. 600 for 10 minutes).
	TurnMaxDurationSeconds float32

	// TurnSegmentEmitMs controls how often OnSegmentReady is called while speech is active.
	// For example, 1000 emits 1-second slices; any remaining tail is emitted before OnSpeechEnd.
	TurnSegmentEmitMs int

	// TurnThreshold is the minimum Smart-Turn probability required to treat a
	// segment as a completed turn. When the model's probability is below this
	// threshold (or Smart-Turn fails), OnSpeechEnd is not invoked.
	TurnThreshold float32

	// TurnTimeoutMs is how long (in ms of silence) to wait after a failed turn
	// before forcing OnSpeechEnd. If there is no speech for this period after
	// we skipped OnSpeechEnd, we invoke OnSpeechEnd (timeout).
	TurnTimeoutMs int

	SileroVADModelPath string // path to silero_vad.onnx
	SmartTurnModelPath string // path to smart-turn-v3.2-cpu.onnx
}

// validate checks Config and returns an error on invalid or missing values.
func validateConfig(cfg Config) error {
	if cfg.SampleRate != RequiredSampleRate {
		return errors.New("config: SampleRate must be 16000")
	}
	if cfg.ChunkSize != RequiredChunkSize {
		return errors.New("config: ChunkSize must be 512")
	}
	if cfg.VadThreshold < 0 || cfg.VadThreshold > 1 {
		return errors.New("config: VadThreshold must be in [0, 1]")
	}
	if cfg.VadPreSpeechMs < 0 {
		return errors.New("config: VadPreSpeechMs must be >= 0")
	}
	if cfg.VadStopMs <= 0 {
		return errors.New("config: VadStopMs must be > 0")
	}
	if cfg.TurnMaxDurationSeconds <= 0 {
		return errors.New("config: TurnMaxDurationSeconds must be > 0")
	}
	if cfg.TurnSegmentEmitMs <= 0 {
		return errors.New("config: TurnSegmentEmitMs must be > 0")
	}
	if cfg.TurnThreshold < 0 || cfg.TurnThreshold > 1 {
		return errors.New("config: TurnThreshold must be in [0, 1]")
	}
	if cfg.TurnTimeoutMs <= 0 {
		return errors.New("config: TurnTimeoutMs must be > 0")
	}
	if cfg.SileroVADModelPath == "" {
		return errors.New("config: SileroVADModelPath is required")
	}
	if cfg.SmartTurnModelPath == "" {
		return errors.New("config: SmartTurnModelPath is required")
	}
	if _, err := os.Stat(cfg.SileroVADModelPath); err != nil {
		if os.IsNotExist(err) {
			return errors.New("config: Silero VAD model file not found: " + cfg.SileroVADModelPath)
		}
		return err
	}
	if _, err := os.Stat(cfg.SmartTurnModelPath); err != nil {
		if os.IsNotExist(err) {
			return errors.New("config: Smart-Turn model file not found: " + cfg.SmartTurnModelPath)
		}
		return err
	}
	return nil
}
