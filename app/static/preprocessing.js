/**
 * Client-side Audio Preprocessing Module
 * Untuk preprocessing lokal sebelum dikirim ke server atau model ONNX
 */

class AudioPreprocessor {
    constructor(targetSampleRate = 16000) {
        this.targetSampleRate = targetSampleRate;
        this.audioContext = null;
    }
    
    async init() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: this.targetSampleRate
        });
    }
    
    /**
     * Load audio file dan decode
     */
    async loadAudio(file) {
        const arrayBuffer = await file.arrayBuffer();
        
        if (!this.audioContext) {
            await this.init();
        }
        
        const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
        return audioBuffer;
    }
    
    /**
     * Convert AudioBuffer ke mono channel
     */
    convertToMono(audioBuffer) {
        const numChannels = audioBuffer.numberOfChannels;
        const length = audioBuffer.length;
        const sampleRate = audioBuffer.sampleRate;
        
        if (numChannels === 1) {
            return audioBuffer.getChannelData(0);
        }
        
        // Mix channels
        const monoData = new Float32Array(length);
        for (let i = 0; i < length; i++) {
            let sum = 0;
            for (let channel = 0; channel < numChannels; channel++) {
                sum += audioBuffer.getChannelData(channel)[i];
            }
            monoData[i] = sum / numChannels;
        }
        
        return monoData;
    }
    
    /**
     * Resample audio ke target sample rate
     */
    resample(audioData, originalSampleRate) {
        if (originalSampleRate === this.targetSampleRate) {
            return audioData;
        }
        
        const ratio = this.targetSampleRate / originalSampleRate;
        const newLength = Math.round(audioData.length * ratio);
        const result = new Float32Array(newLength);
        
        for (let i = 0; i < newLength; i++) {
            const position = i / ratio;
            const index = Math.floor(position);
            const fraction = position - index;
            
            if (index + 1 < audioData.length) {
                result[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
            } else {
                result[i] = audioData[index];
            }
        }
        
        return result;
    }
    
    /**
     * Normalisasi amplitudo ke range [-1, 1]
     */
    normalize(waveform) {
        const maxAmp = Math.max(...waveform.map(Math.abs));
        if (maxAmp > 0) {
            return waveform.map(x => x / maxAmp);
        }
        return waveform;
    }
    
    /**
     * Deteksi segmen batuk berdasarkan energy
     */
    detectCoughSegments(waveform, frameLength = 2048, hopLength = 512) {
        const frames = this.frameWaveform(waveform, frameLength, hopLength);
        const energies = frames.map(frame => this.computeRMS(frame));
        
        // Threshold adaptif
        const meanEnergy = energies.reduce((a, b) => a + b, 0) / energies.length;
        const stdEnergy = Math.sqrt(
            energies.reduce((sum, e) => sum + (e - meanEnergy) ** 2, 0) / energies.length
        );
        const threshold = meanEnergy + 0.5 * stdEnergy;
        
        // Deteksi region aktif
        const minFrames = Math.floor(0.1 * this.targetSampleRate / hopLength); // min 100ms
        const maxFrames = Math.floor(2.0 * this.targetSampleRate / hopLength); // max 2s
        
        const segments = [];
        let startFrame = null;
        
        for (let i = 0; i < energies.length; i++) {
            if (energies[i] > threshold && startFrame === null) {
                startFrame = i;
            } else if ((energies[i] <= threshold || i === energies.length - 1) && startFrame !== null) {
                const duration = i - startFrame;
                if (duration >= minFrames && duration <= maxFrames) {
                    const startSample = startFrame * hopLength;
                    const endSample = Math.min(i * hopLength, waveform.length);
                    segments.push([startSample, endSample]);
                }
                startFrame = null;
            }
        }
        
        return segments;
    }
    
    /**
     * Frame waveform untuk analisis
     */
    frameWaveform(waveform, frameLength, hopLength) {
        const frames = [];
        for (let i = 0; i + frameLength <= waveform.length; i += hopLength) {
            frames.push(waveform.slice(i, i + frameLength));
        }
        return frames;
    }
    
    /**
     * Compute RMS energy
     */
    computeRMS(frame) {
        const sum = frame.reduce((acc, val) => acc + val * val, 0);
        return Math.sqrt(sum / frame.length);
    }
    
    /**
     * Preprocessing lengkap
     */
    async preprocess(file) {
        // 1. Load audio
        const audioBuffer = await this.loadAudio(file);
        
        // 2. Convert to mono
        let waveform = this.convertToMono(audioBuffer);
        
        // 3. Resample ke 16kHz
        waveform = this.resample(waveform, audioBuffer.sampleRate);
        
        // 4. Normalisasi amplitudo
        waveform = this.normalize(waveform);
        
        // 5. Deteksi segmen batuk
        const segments = this.detectCoughSegments(waveform);
        
        // 6. Ekstrak segmen terpanjang atau gunakan audio penuh
        let processedAudio;
        if (segments.length > 0) {
            const longest = segments.reduce((a, b) => 
                (b[1] - b[0]) > (a[1] - a[0]) ? b : a
            );
            processedAudio = waveform.slice(longest[0], longest[1]);
        } else {
            processedAudio = waveform;
        }
        
        // 7. Pad atau truncate ke 5 detik
        const targetLength = this.targetSampleRate * 5;
        if (processedAudio.length < targetLength) {
            const padded = new Float32Array(targetLength);
            padded.set(processedAudio);
            processedAudio = padded;
        } else {
            processedAudio = processedAudio.slice(0, targetLength);
        }
        
        return {
            waveform: processedAudio,
            sampleRate: this.targetSampleRate,
            segments: segments,
            duration: processedAudio.length / this.targetSampleRate
        };
    }
    
    /**
     * Convert Float32Array ke WAV Blob
     */
    floatToWAV(floatArray, sampleRate = 16000) {
        const buffer = new ArrayBuffer(44 + floatArray.length * 2);
        const view = new DataView(buffer);
        
        // WAV Header
        const writeString = (view, offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + floatArray.length * 2, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, 1, true); // Mono
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, 'data');
        view.setUint32(40, floatArray.length * 2, true);
        
        // Convert float to int16
        for (let i = 0; i < floatArray.length; i++) {
            let s = Math.max(-1, Math.min(1, floatArray[i]));
            view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        
        return new Blob([buffer], { type: 'audio/wav' });
    }
}

// Export untuk digunakan di app.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AudioPreprocessor;
}
