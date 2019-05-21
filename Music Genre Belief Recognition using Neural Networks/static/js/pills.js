/*

Copyright (c) 2016 by Justin Windle (http://codepen.io/soulwire/pen/Dscga)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

/*

  This is best viewed in Chrome since there is a bug in Safari
  when using getByteFrequencyData with MediaElementAudioSource

  @see http://goo.gl/6WLx1
 */

pills = function(audio, distribution) {
    var COLOR_STEP = 25;
    var NUM_PARTICLES = 150;
    var NUM_PILLS_PER_GENRE = 20;
    var NUM_BANDS = 128;
    var SMOOTHING = 0.5;
    var SCALE_MIN = 5.0;
    var SCALE_MAX = 80.0;
    var SPEED_MIN = 0.2;
    var SPEED_MAX = 1.0;
    var ALPHA_MIN = 0.8;
    var ALPHA_MAX = 0.9;
    var SPIN_MIN = 0.001;
    var SPIN_MAX = 0.005;
    var SIZE_MIN = 0.5;
    var SIZE_MAX = 1.25;

    var AudioAnalyser = (function() {
        AudioAnalyser.AudioContext = self.AudioContext || self.webkitAudioContext;

        AudioAnalyser.enabled = AudioAnalyser.AudioContext != null;

        function AudioAnalyser(audio, numBands, smoothing) {
            var src;
            this.audio = audio != null ? audio : new Audio();
            this.numBands = numBands != null ? numBands : 256;
            this.smoothing = smoothing != null ? smoothing : 0.3;
            if (typeof this.audio === 'string') {
                src = this.audio;
                this.audio = new Audio();
                this.audio.crossOrigin = "anonymous";
                this.audio.controls = true;
                this.audio.src = src;
            }
            this.context = new AudioAnalyser.AudioContext();
            this.jsNode = this.context.createScriptProcessor(2048, 1, 1);
            this.analyser = this.context.createAnalyser();
            this.analyser.smoothingTimeConstant = this.smoothing;
            this.analyser.fftSize = this.numBands * 2;
            this.bands = new Uint8Array(this.analyser.frequencyBinCount);
            this.audio.addEventListener('canplay', (function(_this) {
                return function() {
                    if (!_this.source) {
                        _this.source = _this.context.createMediaElementSource(_this.audio);
                    }
                    _this.source.connect(_this.analyser);
                    _this.analyser.connect(_this.jsNode);
                    _this.jsNode.connect(_this.context.destination);
                    _this.source.connect(_this.context.destination);
                    return _this.jsNode.onaudioprocess = function() {
                        _this.analyser.getByteFrequencyData(_this.bands);
                        if (!_this.audio.paused) {
                            return typeof _this.onUpdate === "function" ? _this.onUpdate(_this.bands) : void 0;
                        }
                    };
                };
            })(this));
        }

        AudioAnalyser.prototype.start = function() {
            return this.audio.play();
        };

        AudioAnalyser.prototype.stop = function() {
            return this.audio.pause();
        };

        return AudioAnalyser;

    })();

    Particle = (function() {
        function Particle(x1, y1, genre) {
            this.x = x1 != null ? x1 : 0;
            this.y = y1 != null ? y1 : 0;
            this.current_color = '#FFFFFF';
            this.target_color = '#FFFFFF';
            this.reset();
        }

        Particle.prototype.reset = function() {
            this.level = 1 + floor(random(4));
            this.scale = random(SCALE_MIN, SCALE_MAX);
            this.alpha = random(ALPHA_MIN, ALPHA_MAX);
            this.speed = random(SPEED_MIN, SPEED_MAX);
            this.size = random(SIZE_MIN, SIZE_MAX);
            this.spin = random(SPIN_MAX, SPIN_MAX);
            this.band = floor(random(NUM_BANDS));
            if (random() < 0.5) {
                this.spin = -this.spin;
            }
            this.smoothedScale = 0.0;
            this.smoothedAlpha = 0.0;
            this.decayScale = 0.0;
            this.decayAlpha = 0.0;
            this.rotation = random(TWO_PI);
            return this.energy = 0.0;
        };

        Particle.prototype.hexToRgb = function(color) {
            var hex = color.replace('#', '');
            var bigint = parseInt(hex, 16);
            var r = (bigint >> 16) & 255;
            var g = (bigint >> 8) & 255;
            var b = bigint & 255;

            return [r, g, b];
        };

        Particle.prototype.rgbToHex = function(rgb) {
            return "#" + ((1 << 24) + (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]).toString(16).slice(1);
        };

        Particle.prototype.transit_color = function() {
            rgb1 = this.hexToRgb(this.current_color);
            rgb2 = this.hexToRgb(this.target_color);
            rgb3 = [0, 0, 0];
            for (var i = 0; i < 3; i++) {
                rgb3[i] = rgb1[i] < rgb2[i] ? min(rgb1[i] + COLOR_STEP, rgb2[i]) : max(rgb1[i] - COLOR_STEP, rgb2[i]);
            }
            this.current_color = this.rgbToHex(rgb3);
        };

        Particle.prototype.move = function() {
            this.rotation += this.spin;
            return this.y -= this.speed * this.level;
        };

        Particle.prototype.draw = function(ctx) {
            var alpha, power, scale;
            power = exp(this.energy);
            scale = this.scale * power;
            alpha = this.alpha * this.energy * 1.5;
            this.decayScale = max(this.decayScale, scale);
            this.decayAlpha = max(this.decayAlpha, alpha);
            this.smoothedScale += (this.decayScale - this.smoothedScale) * 0.3;
            this.smoothedAlpha += (this.decayAlpha - this.smoothedAlpha) * 0.3;
            this.decayScale *= 0.985;
            this.decayAlpha *= 0.975;
            ctx.save();
            ctx.beginPath();
            ctx.translate(this.x + cos(this.rotation * this.speed) * 250, this.y);
            ctx.rotate(this.rotation);
            ctx.scale(this.smoothedScale * this.level, this.smoothedScale * this.level);
            ctx.moveTo(this.size * 0.5, 0);
            ctx.lineTo(this.size * -0.5, 0);
            ctx.lineWidth = 1;
            ctx.lineCap = 'round';
            ctx.globalAlpha = this.smoothedAlpha / this.level;
            ctx.strokeStyle = this.current_color;
            ctx.stroke();
            return ctx.restore();
        };

        return Particle;

    })();

    Sketch.create({
        distribution: distribution,
        particles: [],
        setup: function() {
            var analyser, i, j, particle, ref, x, y;
            for (i = j = 0, ref = NUM_PARTICLES - 1; j <= ref; i = j += 1) {
                x = random(this.width);
                y = random(this.height * 2);
                particle = new Particle(x, y);
                particle.energy = random(particle.band / 256);
                this.particles.push(particle);
            }
            if (AudioAnalyser.enabled) {
                try {
                    analyser = new AudioAnalyser(audio, NUM_BANDS, SMOOTHING);
                    analyser.onUpdate = (function(_this) {
                        return function(bands) {
                            var k, len, ref1, results, distribution;
                            ref1 = _this.particles;
                            time = analyser.audio.currentTime;
                            distribution = _this.distribution;
                            i = lowerBound(distribution, time, function(x) {
                                return x[0];
                            });
                            j = 0;
                            var maxProbIndex = '';
                            maxProb = 0;
                            for (index in distribution[i][1]) {
                                if (distribution[i][1][index] > maxProb) {
                                    maxProb = distribution[i][1][index];
                                    maxProbIndex = index;
                                }
                            }
                            results = [];
                            for (k = 0, len = ref1.length; k < len; k++) {
                                particle = ref1[k];
                                particle.target_color = GENRE_TO_COLOR.get(maxProbIndex);
                                particle.transit_color();
                                results.push(particle.energy = bands[particle.band] / 256);
                            }
                            return results;
                        };
                    })(this);
                    analyser.start();
                    document.body.appendChild(analyser.audio);
                } catch (error) {
                }
            }
        },
        draw: function() {
            var j, len, particle, ref, results;
            this.globalCompositeOperation = 'lighter';
            ref = this.particles;
            results = [];
            for (j = 0, len = ref.length; j < len; j++) {
                particle = ref[j];
                if (particle.y < -particle.size * particle.level * particle.scale * 2) {
                    particle.reset();
                    particle.x = random(this.width);
                    particle.y = this.height + particle.size * particle.scale * particle.level;
                }
                particle.move();
                results.push(particle.draw(this));
            }
            return results;
        }
    });

};
