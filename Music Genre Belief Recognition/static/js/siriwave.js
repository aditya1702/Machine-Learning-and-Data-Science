/*

The MIT License (MIT)

Copyright (c) 2015 Caffeina

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

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

(function() {

function SiriWave(opt) {
    opt = opt || {};

    this.phase = 0;
    this.run = false;

    // UI vars

    this.ratio = opt.ratio || window.devicePixelRatio || 1;

    this.width = this.ratio * (opt.width || 320);
    this.width_2 = this.width / 2;
    this.width_4 = this.width / 4;

    this.height = this.ratio * (opt.height || 100);
    this.height_2 = this.height / 2;

    this.MAX = (this.height_2) - 4;

    // Constructor opt

    this.amplitude = opt.amplitude || 1;
    this.speed = opt.speed || 0.2;
    this.frequency = opt.frequency || 6;

    // Canvas

    this.canvas = document.createElement('canvas');
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.canvas.style.width = (this.width / this.ratio) + 'px';
    this.canvas.style.height = (this.height / this.ratio) + 'px';

    this.container = opt.container || document.body;
    this.container.appendChild(this.canvas);

    this.color1 = opt.color1;
    this.color2 = opt.color2;
    this.color3 = opt.color3;
    this.color4 = opt.color4;
    this.color5 = opt.color5;

    this.ctx = this.canvas.getContext('2d');

    // Start

    if (opt.autostart) {
        this.start();
    }
}


SiriWave.prototype._GATF_cache = {};
SiriWave.prototype._globAttFunc = function (x) {
    if (SiriWave.prototype._GATF_cache[x] == null) {
        SiriWave.prototype._GATF_cache[x] = Math.pow(4 / (4 + Math.pow(x, 4)), 2);
    }
    return SiriWave.prototype._GATF_cache[x];
};

SiriWave.prototype._xpos = function (i) {
    return this.width_2 + i * this.width_4;
};

SiriWave.prototype._ypos = function (i, attenuation) {
    var att = (this.MAX * this.amplitude) / attenuation;
    return this.height_2 + this._globAttFunc(i) * att * Math.sin(this.frequency * i - this.phase);
};

SiriWave.prototype._drawLine = function (attenuation, color, width) {
    this.ctx.moveTo(0, 0);
    this.ctx.beginPath();
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = width || 1;

    var i = -2;
    while ((i += 0.01) <= 2) {
        this.ctx.lineTo(this._xpos(i), this._ypos(i, attenuation));
    }

    this.ctx.stroke();
};

SiriWave.prototype._clear = function () {
    this.ctx.globalCompositeOperation = 'destination-out';
    this.ctx.fillRect(0, 0, this.width, this.height);
    this.ctx.globalCompositeOperation = 'source-over';
};

SiriWave.prototype._draw = function () {
    if (this.run === false) return;

    this.phase = (this.phase + Math.PI * this.speed) % (2 * Math.PI);

    this._clear();
    this._drawLine(-2, 'rgba(' + this.color5 + ',0.1)', 2);
    this._drawLine(-6, 'rgba(' + this.color4 + ',0.2)', 3);
    this._drawLine(4, 'rgba(' + this.color3 + ',0.4)', 4);
    this._drawLine(2, 'rgba(' + this.color2 + ',0.6)', 5);
    this._drawLine(1, 'rgba(' + this.color1 + ',1)', 6);


    requestAnimationFrame(this._draw.bind(this));
};

/* API */

SiriWave.prototype.start = function () {
    this.phase = 0;
    this.run = true;
    this._draw();
};

SiriWave.prototype.stop = function () {
    this.phase = 0;
    this.run = false;
};

SiriWave.prototype.setSpeed = function (v) {
    this.speed = v;
};

SiriWave.prototype.setNoise = SiriWave.prototype.setAmplitude = function (v) {
    this.amplitude = Math.max(Math.min(v, 1), 0);
};

if (typeof define === 'function' && define.amd) {
	define(function(){ return SiriWave; });
} else {
	window.SiriWave = SiriWave;
}

})();
