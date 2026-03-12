import AVFoundation
import Foundation
import Observation
import Qwen3Chat
import KokoroTTS
import ParakeetASR
import AudioCommon

/// Message displayed in chat UI.
struct ChatBubbleMessage: Identifiable {
    let id = UUID()
    let role: ChatMessage.Role
    var text: String
    let timestamp = Date()
}

@Observable
@MainActor
final class CompanionChatViewModel {
    // MARK: - UI State

    var messages: [ChatBubbleMessage] = []
    var inputText = ""
    var isLoading = false
    var isGenerating = false
    var isListening = false
    var audioLevel: Float = 0
    var loadProgress: Double = 0
    var loadingStatus = ""
    var errorMessage: String?
    var speakEnabled = true

    var modelsLoaded: Bool { chatModel != nil && ttsModel != nil && asrModel != nil }

    // MARK: - Models

    private var chatModel: Qwen3ChatModel?
    private var ttsModel: KokoroTTSModel?
    private var asrModel: ParakeetASRModel?

    // MARK: - Audio

    private var audioEngine: AVAudioEngine?
    private var recordedSamples: [Float] = []
    private let samplesLock = NSLock()
    private var playerNode: AVAudioPlayerNode?

    private let systemPrompt = """
        You are a friendly companion. Keep responses short and conversational \
        (1-2 sentences). Be warm and helpful.
        """

    // MARK: - Load Models

    func loadModels() async {
        isLoading = true
        errorMessage = nil
        loadProgress = 0

        do {
            // 1. Load ASR (Parakeet CoreML)
            loadingStatus = "Loading Parakeet ASR..."
            loadProgress = 0.1
            asrModel = try await Task.detached {
                let model = try await ParakeetASRModel.fromPretrained()
                try model.warmUp()
                return model
            }.value

            // 2. Load Chat LLM (Qwen3 CoreML)
            loadingStatus = "Loading Qwen3 Chat..."
            loadProgress = 0.4
            chatModel = try await Task.detached {
                try await Qwen3ChatModel.fromPretrained { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.4 + progress * 0.3
                        if !status.isEmpty { self?.loadingStatus = status }
                    }
                }
            }.value

            // 3. Load TTS (Kokoro CoreML)
            loadingStatus = "Loading Kokoro TTS..."
            loadProgress = 0.7
            ttsModel = try await Task.detached {
                let model = try await KokoroTTSModel.fromPretrained { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.7 + progress * 0.3
                        if !status.isEmpty { self?.loadingStatus = status }
                    }
                }
                try model.warmUp()
                return model
            }.value

            loadProgress = 1.0
            loadingStatus = "Ready"
        } catch {
            errorMessage = "Load failed: \(error.localizedDescription)"
        }

        isLoading = false
    }

    // MARK: - Send Message

    func send(_ text: String) async {
        inputText = ""
        messages.append(ChatBubbleMessage(role: .user, text: text))

        guard let chatModel else { return }
        isGenerating = true
        errorMessage = nil

        do {
            // Stream LLM response
            var responseText = ""
            let assistantIdx = messages.count
            messages.append(ChatBubbleMessage(role: .assistant, text: ""))

            let stream = chatModel.chatStream(
                text,
                systemPrompt: systemPrompt,
                sampling: .default
            )

            for try await chunk in stream {
                responseText += chunk
                messages[assistantIdx].text = responseText
            }

            // Speak response
            if speakEnabled, let tts = ttsModel, !responseText.isEmpty {
                await speakText(responseText, using: tts)
            }
        } catch {
            errorMessage = "Generation failed: \(error.localizedDescription)"
        }

        isGenerating = false
    }

    // MARK: - Voice Input

    func startListening() {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try session.setActive(true)
        } catch {
            errorMessage = "Mic access failed: \(error.localizedDescription)"
            return
        }
        #endif

        samplesLock.lock()
        recordedSamples = []
        samplesLock.unlock()

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        ) else { return }

        guard let converter = AVAudioConverter(from: hwFormat, to: targetFormat) else { return }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            guard let self else { return }

            let frameCount = AVAudioFrameCount(
                Double(buffer.frameLength) * 16000.0 / hwFormat.sampleRate
            )
            guard let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat, frameCapacity: frameCount
            ) else { return }

            var error: NSError?
            converter.convert(to: convertedBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            if error != nil { return }

            guard let channelData = convertedBuffer.floatChannelData else { return }
            let count = Int(convertedBuffer.frameLength)
            let data = Array(UnsafeBufferPointer(start: channelData[0], count: count))

            let rms = sqrt(data.reduce(0) { $0 + $1 * $1 } / max(Float(count), 1))

            self.samplesLock.lock()
            self.recordedSamples.append(contentsOf: data)
            self.samplesLock.unlock()

            DispatchQueue.main.async {
                self.audioLevel = rms
            }
        }

        do {
            try engine.start()
            audioEngine = engine
            isListening = true
        } catch {
            errorMessage = "Mic error: \(error.localizedDescription)"
        }
    }

    func stopListening() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        isListening = false
        audioLevel = 0

        samplesLock.lock()
        let samples = recordedSamples
        recordedSamples = []
        samplesLock.unlock()

        guard !samples.isEmpty, let asr = asrModel else { return }

        Task {
            do {
                let captured = samples
                let text = try await Task.detached {
                    try asr.transcribeAudio(captured, sampleRate: 16000)
                }.value

                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    inputText = trimmed
                    await send(trimmed)
                }
            } catch {
                errorMessage = "Transcription failed: \(error.localizedDescription)"
            }
        }
    }

    // MARK: - TTS Playback

    private func speakText(_ text: String, using tts: KokoroTTSModel) async {
        do {
            let samples = try await Task.detached {
                try tts.synthesize(text: text, voice: "af_heart")
            }.value

            guard !samples.isEmpty else { return }
            try playAudio(samples: samples, sampleRate: 24000)
        } catch {
            // TTS failure is non-critical
        }
    }

    private func playAudio(samples: [Float], sampleRate: Int) throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default)
        try session.setActive(true)
        #endif

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else { return }

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else { return }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        memcpy(buffer.floatChannelData![0], samples, samples.count * MemoryLayout<Float>.size)

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        try engine.start()

        player.scheduleBuffer(buffer)
        player.play()

        self.playerNode = player
        // Engine kept alive by reference; stops when ViewModel deallocates or next playback
    }

    // MARK: - Actions

    func clearChat() {
        messages = []
        chatModel?.resetConversation()
    }
}
