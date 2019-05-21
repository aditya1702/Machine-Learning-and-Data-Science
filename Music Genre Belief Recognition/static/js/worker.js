const window = self;

importScripts(
    'resampler.js',
    'audio-utils.js'
);

const N_FFT = 2048;
const FFT_HOP = 1024;
const N_MELS = 128;

function logMelSpectrogram(wav) {
    const stft = AudioUtils.default.stft(wav, N_FFT, FFT_HOP);
    const stftEnergies = stft.map(wnd => AudioUtils.default.fftEnergies(wnd));
    const mel = AudioUtils.default.melSpectrogram(stftEnergies, N_MELS);
    return mel.map(wnd => wnd.map(energy => Math.log(Math.max(energy, 1e-6))));
}

async function preprocess(audio, sampleRate) {
    const resampler = new Resampler(sampleRate, 22050, 1, audio.length);
    const resampled = resampler.resampler(audio);
    return logMelSpectrogram(resampled);
}

onmessage = async function(event) {
    const [audio, sampleRate] = event.data;
    const prediction = await preprocess(audio, sampleRate);
    postMessage(prediction);
}
