'use strict';

(function() {

var worker = new Worker('js/worker.js');

const modelPromise = tf.loadModel('model/model.json');
console.log(modelPromise)

function getRandColor(brightness) {
    var rgb = [Math.random() * 256, Math.random() * 256, Math.random() * 256];
    var mix = [brightness * 51, brightness * 51, brightness * 51];
    var mixedrgb = [0, 1, 2].map(function(i) {
        return Math.round((rgb[i] + mix[i]) / 2.0);
    });
    return mixedrgb.join(',');
}

function specToInputTensor(spec) {
    const width = spec.length;
    const height = spec[0].length;
    var flatSpec = new Float32Array(width * height);
    for(var i = 0; i < width; ++i) {
        flatSpec.set(spec[i], i * height);
    }
    return tf.tensor3d(flatSpec, [1, width, height]);
}

function preprocess(data) {
    return new Promise(function(resolve, reject) {
        worker.onmessage = async function(event) {
            resolve(event.data);
        };
        worker.postMessage(data);
    });
}

function smooth(prediction) {
    const weights = [
        0.11, 0.13, 0.17, 0.21, 0.26, 0.33, 0.41, 0.51, 0.64, 0.8, 1
    ];
    var newPrediction = new Float32Array(prediction.length);
    for(var i = 0; i < prediction.length; ++i) {
        var totalWeight = 0;
        for(var j = 0; j < weights.length; ++j) {
            const k = i + (j - weights.length + 1) * 10;
            if(k >= 0 && k < prediction.length) {
                newPrediction[i] += weights[j] * prediction[k];
                totalWeight += weights[j];
            }
        }
        newPrediction[i] /= totalWeight;
    }
    return newPrediction;
}

async function process(audio, sampleRate) {
    const spec = await preprocess([audio, sampleRate]);
    const input = specToInputTensor(event.data);
    const model = await modelPromise;
    const predictionTensor = tf.tidy(function() {
        return model.predict(input);
    });
    console.log(input);
    const predictionArray = await predictionTensor.data();
    predictionTensor.dispose();
    return smooth(predictionArray);
}

function drawPieChart(canvasID, distribution, timeFn) {
    var startValue = 0;
    var data = GENRES.map(function(genre) {
        var color = GENRE_TO_COLOR.get(genre);
        return {
            value: startValue,
            color: color,
            highlight: color,
            label: genre
        };
    });

    var shown = false;
    var context = $(canvasID).get(0).getContext('2d');
    var options = {
        animationEasing: 'linear',
        animationSteps: 10
    };
    var chart = new Chart(context).Pie(data, options);

    function updateChart() {
        var i = lowerBound(distribution, timeFn(), function(x) {
            return x[0];
        });
        i = min(i, distribution.length - 1);
        GENRES.forEach(function(genre, index) {
            chart.segments[index].value = parseFloat(
                distribution[i][1][genre]
            );
        });
        chart.update();
        setTimeout(updateChart, 100);
    }
    updateChart();
}

function decodeAudio(file) {
    return new Promise(function(resolve, reject) {
        const reader = new FileReader();
        reader.onload = async function(event) {
            const buffer = event.target.result;
            const context = new AudioContext();
            const decoded = await context.decodeAudioData(buffer);
            resolve(decoded);
        }
        reader.onerror = reject;
        reader.readAsArrayBuffer(file);
    });
}

function genreDistributionOverTime(prediction, duration) {
    const dt = duration / prediction.length * GENRES.length;
    var distribution = [];
    for(var i = 0; i < prediction.length / GENRES.length; ++i) {
        const from = i * GENRES.length;
        const to = from + GENRES.length;
        distribution.push(
            [(i + 1) * dt, prediction.slice(from, to).reduce(function(acc, cur, j) {
                acc[GENRES[j]] = cur;
                return acc;
            }, {})]
        );
    }
    return distribution;
}

async function sendForm() {
    var wave = new SiriWave({
        width: window.innerWidth,
        height: window.innerHeight / 2,
        speed: 0.06,
        noise: 0.9,
        container: $('#wave').get(0),
        color1: getRandColor(3),
        color2: getRandColor(3),
        color3: getRandColor(3),
        color4: getRandColor(3),
        color5: getRandColor(3)
    });

    $('#upload').fadeOut(300, function() {
        $('#message-upload').hide();
        $('#message-wave').show();
        wave.start();
        $('body').addClass('loading');

        const file = $('#upload input')[0].files[0];
        decodeAudio(file).then(function(audioBuffer) {
            const duration = audioBuffer.duration;
            const channel = audioBuffer.getChannelData(0);
            process(channel, audioBuffer.sampleRate).then(function(prediction) {
                wave.stop();
                $('#message-wave').hide();
                $('body').removeClass('loading');
                $('.logo-big').removeClass('logo-big').addClass('logo-small');

                const distribution = genreDistributionOverTime(
                    prediction, duration
                );
                pills(URL.createObjectURL(file), distribution);
                drawPieChart('#piechart', distribution, function() {
                    return $('audio').get(0).currentTime;
                });
                $('#piechart-container').show();
            });
        });
    });
}

$(function() {
    $('#upload input').change(sendForm);
});

})();
