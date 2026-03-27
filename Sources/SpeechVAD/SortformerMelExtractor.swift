import Foundation
import Accelerate

/// 128-dim log-mel feature extractor for Sortformer diarization.
///
/// Matches NeMo's audio preprocessor: Hann window (no Povey), no pre-emphasis,
/// nFFT=400, hop=160, 128 mel bins, 16kHz. Uses vDSP for FFT and mel filterbank.
///
/// Key differences from `MelFeatureExtractor` (WeSpeaker):
/// - 128 mel bins (vs 80)
/// - Hann window (vs Povey window)
/// - No pre-emphasis (vs 0.97)
/// - Power spectrum (vs magnitude spectrum)
class SortformerMelExtractor {
    let sampleRate: Int
    let nFFT: Int
    let hopLength: Int
    let nMels: Int

    private let paddedFFT: Int = 512
    private let log2PaddedFFT: vDSP_Length = 9
    private var fftSetup: FFTSetup
    private var window: [Float]
    private var melFilterbank: [Float]  // [nMels, nBins]

    init(config: SortformerConfig = .default) {
        self.sampleRate = config.sampleRate
        self.nFFT = config.nFFT
        self.hopLength = config.hopLength
        self.nMels = config.nMels

        let halfPadded = 512 / 2  // paddedFFT / 2
        let nBins = halfPadded + 1

        // Hann window (NeMo default, no Povey modification)
        window = [Float](repeating: 0, count: config.nFFT)
        for i in 0..<config.nFFT {
            window[i] = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(i) / Float(config.nFFT - 1))
        }

        guard let setup = vDSP_create_fftsetup(log2PaddedFFT, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create vDSP FFT setup")
        }
        fftSetup = setup

        // Pre-allocate reusable buffers for single-frame extraction
        sfPaddedFrame = [Float](repeating: 0, count: 512)
        sfSplitReal = [Float](repeating: 0, count: halfPadded)
        sfSplitImag = [Float](repeating: 0, count: halfPadded)
        sfPowerSpec = [Float](repeating: 0, count: nBins)
        sfMelFrame = [Float](repeating: 0, count: config.nMels)

        melFilterbank = []
        setupMelFilterbank()

        // Transpose filterbank for per-frame dot products
        melFilterbankT = [Float](repeating: 0, count: config.nMels * nBins)
        for m in 0..<config.nMels {
            for b in 0..<nBins {
                melFilterbankT[m * nBins + b] = melFilterbank[m * nBins + b]
            }
        }
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    private func setupMelFilterbank() {
        let fMin: Float = 0.0
        let fMax: Float = Float(sampleRate) / 2.0

        // HTK mel scale
        func hzToMel(_ hz: Float) -> Float {
            2595.0 * log10(1.0 + hz / 700.0)
        }

        func melToHz(_ mel: Float) -> Float {
            700.0 * (pow(10.0, mel / 2595.0) - 1.0)
        }

        let nBins = paddedFFT / 2 + 1  // 257

        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(paddedFFT)
        }

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)

        let nMelPoints = nMels + 2
        var melPoints = [Float](repeating: 0, count: nMelPoints)
        for i in 0..<nMelPoints {
            melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(nMelPoints - 1)
        }

        let filterFreqs = melPoints.map { melToHz($0) }

        var filterDiff = [Float](repeating: 0, count: nMelPoints - 1)
        for i in 0..<(nMelPoints - 1) {
            filterDiff[i] = filterFreqs[i + 1] - filterFreqs[i]
        }

        // Build filterbank [nBins, nMels]
        var filterbank = [Float](repeating: 0, count: nBins * nMels)
        for bin in 0..<nBins {
            let freq = fftFreqs[bin]
            for mel in 0..<nMels {
                let lowFreq = filterFreqs[mel]
                let highFreq = filterFreqs[mel + 2]
                let downSlope = (freq - lowFreq) / filterDiff[mel]
                let upSlope = (highFreq - freq) / filterDiff[mel + 1]
                filterbank[bin * nMels + mel] = max(0.0, min(downSlope, upSlope))
            }
        }

        // Slaney normalization
        for mel in 0..<nMels {
            let enorm = 2.0 / (filterFreqs[mel + 2] - filterFreqs[mel])
            for bin in 0..<nBins {
                filterbank[bin * nMels + mel] *= enorm
            }
        }

        // Transpose to [nMels, nBins]
        var transposed = [Float](repeating: 0, count: nMels * nBins)
        for mel in 0..<nMels {
            for bin in 0..<nBins {
                transposed[mel * nBins + bin] = filterbank[bin * nMels + mel]
            }
        }

        self.melFilterbank = transposed
    }

    // MARK: - Streaming State

    /// Audio buffer for incremental extraction (padded audio tail)
    private var streamBuffer: [Float] = []
    /// Global sample offset: how many samples have been trimmed from streamBuffer
    private var streamBaseOffset: Int = 0
    /// Number of mel frames extracted so far in streaming mode
    private var streamFramesExtracted: Int = 0
    /// Whether the first chunk has been processed (for left reflect padding)
    private var streamStarted: Bool = false

    /// Reusable buffers for single-frame FFT (avoid allocation per frame)
    private var sfPaddedFrame: [Float]
    private var sfSplitReal: [Float]
    private var sfSplitImag: [Float]
    private var sfPowerSpec: [Float]
    private var sfMelFrame: [Float]
    /// Transposed filterbank for dot product: [nMels, nBins]
    private var melFilterbankT: [Float] = []

    /// Reset streaming state for a new audio session.
    func resetStreamingState() {
        streamBuffer = []
        streamBaseOffset = 0
        streamFramesExtracted = 0
        streamStarted = false
    }

    /// Extract mel frames incrementally from new audio samples.
    ///
    /// Maintains internal overlap state so that chunk boundaries produce
    /// the exact same mel values as batch `extract()` on the full audio.
    /// Memory usage is constant (~2 KB for overlap buffer).
    ///
    /// - Parameter newSamples: New PCM Float32 audio samples at 16kHz
    /// - Returns: Flat array of new mel frames `[nNewFrames * nMels]`
    func extractIncremental(newSamples: [Float]) -> [Float] {
        guard !newSamples.isEmpty else { return [] }

        if !streamStarted {
            // First chunk: add reflect padding at the beginning
            let padLen = nFFT / 2
            var pad = [Float](repeating: 0, count: padLen)
            for i in 0..<padLen {
                let srcIdx = min(padLen - i, newSamples.count - 1)
                pad[i] = newSamples[max(0, srcIdx)]
            }
            streamBuffer = pad
            streamStarted = true
        }
        streamBuffer.append(contentsOf: newSamples)

        // Extract all possible new frames
        var newMel = [Float]()
        while true {
            let globalStart = streamFramesExtracted * hopLength
            let localStart = globalStart - streamBaseOffset
            guard localStart >= 0, localStart + nFFT <= streamBuffer.count else { break }

            streamBuffer.withUnsafeBufferPointer { buf in
                let frameMel = computeSingleFrameMel(samples: buf.baseAddress! + localStart)
                newMel.append(contentsOf: frameMel)
            }
            streamFramesExtracted += 1
        }

        // Trim consumed samples — keep only what's needed for the next frame
        let nextGlobal = streamFramesExtracted * hopLength
        let trimCount = nextGlobal - streamBaseOffset
        if trimCount > 0 && trimCount < streamBuffer.count {
            streamBuffer.removeFirst(trimCount)
            streamBaseOffset = nextGlobal
        }

        return newMel
    }

    /// Finalize streaming extraction: add right reflect padding and extract remaining frames.
    ///
    /// Call this once after all audio has been fed via `extractIncremental`.
    /// - Returns: Flat array of remaining mel frames `[nFrames * nMels]`
    func extractFinal() -> [Float] {
        guard streamStarted else { return [] }

        // Add reflect padding at the end (same as torch.stft center=True)
        let padLen = nFFT / 2
        let bufLen = streamBuffer.count
        guard bufLen >= 2 else { return [] }
        var pad = [Float](repeating: 0, count: padLen)
        for i in 0..<padLen {
            let srcIdx = bufLen - 2 - i
            pad[i] = streamBuffer[max(0, srcIdx)]
        }
        streamBuffer.append(contentsOf: pad)

        // Extract remaining frames
        var newMel = [Float]()
        while true {
            let globalStart = streamFramesExtracted * hopLength
            let localStart = globalStart - streamBaseOffset
            guard localStart >= 0, localStart + nFFT <= streamBuffer.count else { break }

            streamBuffer.withUnsafeBufferPointer { buf in
                newMel.append(contentsOf: computeSingleFrameMel(samples: buf.baseAddress! + localStart))
            }
            streamFramesExtracted += 1
        }

        return newMel
    }

    /// Compute mel spectrum for a single FFT frame.
    ///
    /// - Parameter samples: Pointer to `nFFT` contiguous audio samples (already in correct position)
    /// - Returns: `[nMels]` log-mel values
    private func computeSingleFrameMel(samples: UnsafePointer<Float>) -> [Float] {
        let nBins = paddedFFT / 2 + 1
        let halfPadded = paddedFFT / 2

        // Window
        vDSP_vmul(samples, 1, window, 1, &sfPaddedFrame, 1, vDSP_Length(nFFT))
        for i in nFFT..<paddedFFT { sfPaddedFrame[i] = 0 }

        // Deinterleave for split complex
        for i in 0..<halfPadded {
            sfSplitReal[i] = sfPaddedFrame[2 * i]
            sfSplitImag[i] = sfPaddedFrame[2 * i + 1]
        }

        // FFT
        sfSplitReal.withUnsafeMutableBufferPointer { realBuf in
            sfSplitImag.withUnsafeMutableBufferPointer { imagBuf in
                var splitComplex = DSPSplitComplex(
                    realp: realBuf.baseAddress!,
                    imagp: imagBuf.baseAddress!)
                vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2PaddedFFT, FFTDirection(kFFTDirection_Forward))
            }
        }

        // Power spectrum
        sfPowerSpec[0] = sfSplitReal[0] * sfSplitReal[0]
        sfPowerSpec[halfPadded] = sfSplitImag[0] * sfSplitImag[0]
        for k in 1..<halfPadded {
            sfPowerSpec[k] = sfSplitReal[k] * sfSplitReal[k] + sfSplitImag[k] * sfSplitImag[k]
        }

        // Mel filterbank: dot product per mel bin
        melFilterbankT.withUnsafeBufferPointer { fbPtr in
            for m in 0..<nMels {
                var sum: Float = 0
                vDSP_dotpr(sfPowerSpec, 1, fbPtr.baseAddress! + m * nBins, 1, &sum, vDSP_Length(nBins))
                sfMelFrame[m] = sum
            }
        }

        // Log-mel
        var countN = Int32(nMels)
        var epsilon: Float = 1e-10
        vDSP_vclip(sfMelFrame, 1, &epsilon, [Float.greatestFiniteMagnitude], &sfMelFrame, 1, vDSP_Length(nMels))
        vvlogf(&sfMelFrame, sfMelFrame, &countN)

        return sfMelFrame
    }

    // MARK: - Batch Extraction

    /// Extract 128-dim log-mel features from audio.
    ///
    /// - Parameter audio: PCM Float32 samples at 16kHz
    /// - Returns: `(melSpec, nFrames)` where melSpec is a flat `[nFrames * 128]` array
    func extract(_ audio: [Float]) -> (melSpec: [Float], nFrames: Int) {
        let nBins = paddedFFT / 2 + 1
        let halfPadded = paddedFFT / 2

        // No pre-emphasis for Sortformer (NeMo default)

        guard !audio.isEmpty else { return ([], 0) }

        // Reflect padding (same as torch.stft with center=True)
        let padLength = nFFT / 2
        var paddedAudio = [Float](repeating: 0, count: padLength + audio.count + padLength)

        for i in 0..<padLength {
            let srcIdx = min(padLength - i, audio.count - 1)
            paddedAudio[i] = audio[max(0, srcIdx)]
        }
        for i in 0..<audio.count {
            paddedAudio[padLength + i] = audio[i]
        }
        for i in 0..<padLength {
            let srcIdx = audio.count - 2 - i
            paddedAudio[padLength + audio.count + i] = audio[max(0, srcIdx)]
        }

        let nFrames = (paddedAudio.count - nFFT) / hopLength + 1

        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)
        var paddedFrame = [Float](repeating: 0, count: paddedFFT)
        var powerSpec = [Float](repeating: 0, count: nFrames * nBins)

        for frame in 0..<nFrames {
            let start = frame * hopLength

            paddedAudio.withUnsafeBufferPointer { buf in
                vDSP_vmul(buf.baseAddress! + start, 1, window, 1, &paddedFrame, 1, vDSP_Length(nFFT))
            }
            for i in nFFT..<paddedFFT {
                paddedFrame[i] = 0
            }

            for i in 0..<halfPadded {
                splitReal[i] = paddedFrame[2 * i]
                splitImag[i] = paddedFrame[2 * i + 1]
            }

            splitReal.withUnsafeMutableBufferPointer { realBuf in
                splitImag.withUnsafeMutableBufferPointer { imagBuf in
                    var splitComplex = DSPSplitComplex(
                        realp: realBuf.baseAddress!,
                        imagp: imagBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2PaddedFFT, FFTDirection(kFFTDirection_Forward))
                }
            }

            let baseIdx = frame * nBins
            // Power spectrum: |X|^2
            powerSpec[baseIdx] = splitReal[0] * splitReal[0]
            powerSpec[baseIdx + halfPadded] = splitImag[0] * splitImag[0]
            for k in 1..<halfPadded {
                powerSpec[baseIdx + k] = splitReal[k] * splitReal[k] + splitImag[k] * splitImag[k]
            }
        }

        // Mel filterbank matmul: [nFrames, nBins] × [nBins, nMels] = [nFrames, nMels]
        var melSpec = [Float](repeating: 0, count: nFrames * nMels)
        var filterbankT = [Float](repeating: 0, count: nBins * nMels)
        vDSP_mtrans(melFilterbank, 1, &filterbankT, 1, vDSP_Length(nBins), vDSP_Length(nMels))

        vDSP_mmul(powerSpec, 1, filterbankT, 1, &melSpec, 1,
                  vDSP_Length(nFrames), vDSP_Length(nMels), vDSP_Length(nBins))

        // Log-mel: log(max(x, 1e-10))
        let count = melSpec.count
        var countN = Int32(count)

        var epsilon: Float = 1e-10
        vDSP_vclip(melSpec, 1, &epsilon, [Float.greatestFiniteMagnitude], &melSpec, 1, vDSP_Length(count))
        vvlogf(&melSpec, melSpec, &countN)

        return (melSpec, nFrames)
    }
}
