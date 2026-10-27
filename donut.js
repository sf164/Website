// Interactive torus figure: shared controls.
// Drag (mouse or finger) turns the donut; a flick keeps it gliding and it slows
// to a stop. Left alone, it settles back to its resting tilt and turns slowly
// on its own axis. Arrow keys work once the figure has focus.
(function (global) {
    'use strict';

    /* ---------- Small quaternion kit: [x, y, z, w] ---------- */
    function qAxis(ax, ay, az, angle) {
        var s = Math.sin(angle / 2);
        return [ax * s, ay * s, az * s, Math.cos(angle / 2)];
    }

    function qMul(a, b) {
        return [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
        ];
    }

    function qNorm(q) {
        var l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
        return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
    }

    // Column-major 3x3 rotation matrix (what WebGL's uniformMatrix3fv expects).
    function qToMat3(q, out) {
        var x = q[0], y = q[1], z = q[2], w = q[3];
        out[0] = 1 - 2 * (y * y + z * z); out[3] = 2 * (x * y - w * z);     out[6] = 2 * (x * z + w * y);
        out[1] = 2 * (x * y + w * z);     out[4] = 1 - 2 * (x * x + z * z); out[7] = 2 * (y * z - w * x);
        out[2] = 2 * (x * z - w * y);     out[5] = 2 * (y * z + w * x);     out[8] = 1 - 2 * (x * x + y * y);
        return out;
    }

    /* ---------- Geometry shared by both looks ---------- */
    // Torus with its axis along y. u goes around the axis, v around the tube.
    var TORUS = { R: 2.2, r: 0.95, p: 2, q: 5 }; // the (2,5) torus knot

    function torusPoint(u, v, lift) {
        var rr = TORUS.r + (lift || 0);
        var ring = TORUS.R + rr * Math.cos(v);
        return [ring * Math.cos(u), rr * Math.sin(v), ring * Math.sin(u)];
    }

    function torusNormal(u, v) {
        return [Math.cos(v) * Math.cos(u), Math.sin(v), Math.cos(v) * Math.sin(u)];
    }

    /* ---------- The controller ---------- */
    var REST_TILT = 38 * Math.PI / 180;   // how far the top leans toward you at rest
    var IDLE_SPIN = 0.22;                 // rad/s around its own axis when left alone
    var MAX_SPEED = 14;                   // rad/s cap on a flick

    function mount(root, makeRenderer) {
        var img = root.querySelector('img');
        var canvas = document.createElement('canvas');
        canvas.className = 'donut-canvas';
        canvas.setAttribute('aria-hidden', 'true');

        var renderer;
        try { renderer = makeRenderer(canvas); } catch (err) { renderer = null; }
        if (!renderer) return null; // keep the still picture

        root.insertBefore(canvas, root.firstChild);
        if (img) img.draggable = false;
        // Never let the browser start dragging the picture underneath.
        root.addEventListener('dragstart', function (e) { e.preventDefault(); });
        root.setAttribute('tabindex', '0');
        root.setAttribute('role', 'img');
        root.setAttribute('aria-roledescription', 'interactive 3D figure');
        var label = root.getAttribute('data-label') || (img && img.alt) || '';
        root.setAttribute('aria-label', (label ? label + '. ' : '') + 'Drag or use the arrow keys to turn it.');

        var hint = document.createElement('span');
        hint.className = 'donut-hint';
        hint.setAttribute('aria-hidden', 'true');
        hint.textContent = 'Drag to turn';
        root.appendChild(hint);

        var reduceMotion = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)');
        function calm() { return !!(reduceMotion && reduceMotion.matches); }

        // Orientation and motion
        var q = qAxis(1, 0, 0, REST_TILT);
        q = qMul(q, qAxis(0, 1, 0, -0.9));
        var mat = new Float32Array(9);
        var wx = 0, wy = 0;              // angular velocity about screen x / y (rad/s)
        var spin = calm() ? 0 : IDLE_SPIN;
        var dragging = false, pointerId = null;
        var lastX = 0, lastY = 0, pendX = 0, pendY = 0;
        var samples = [];
        var lastTouch = -1e9;            // time of the last hand-on interaction (ms)
        var size = 1, dpr = 1;

        function radPerPx() { return 3.4 / Math.max(size, 160); }

        function rotateScreen(ax, ay, az, angle) {
            if (angle) q = qMul(qAxis(ax, ay, az, angle), q);
        }

        // Ease the torus's axis back to its resting tilt, keeping how far it has
        // turned on that axis, so it always comes to rest looking like the figure.
        function settle(dt) {
            var m = qToMat3(q, mat);
            var ax = m[3], ay = m[4], az = m[5];                 // where the model's y axis points now
            var hx = 0, hy = Math.cos(REST_TILT), hz = Math.sin(REST_TILT);
            var cx = ay * hz - az * hy, cy = az * hx - ax * hz, cz = ax * hy - ay * hx;
            var s = Math.hypot(cx, cy, cz), c = ax * hx + ay * hy + az * hz;
            var angle = Math.atan2(s, c);
            if (angle < 1e-4) return;
            if (s < 1e-6) { cx = 1; cy = 0; cz = 0; s = 1; }      // upside down: tip forward
            var step = angle * (1 - Math.exp(-dt / 0.85));
            q = qMul(qAxis(cx / s, cy / s, cz / s, step), q);
        }

        /* Pointer input */
        function onDown(e) {
            if (e.pointerType === 'mouse' && e.button !== 0) return;
            if (dragging) return;
            dragging = true;
            pointerId = e.pointerId;
            try { root.setPointerCapture(e.pointerId); } catch (_) { }
            lastX = e.clientX; lastY = e.clientY;
            wx = wy = 0;
            samples.length = 0;
            root.classList.add('is-dragging', 'was-touched');
            lastTouch = performance.now();
            wake();
        }

        function onMove(e) {
            if (!dragging || e.pointerId !== pointerId) return;
            var dx = e.clientX - lastX, dy = e.clientY - lastY;
            lastX = e.clientX; lastY = e.clientY;
            pendX += dx; pendY += dy;
            var now = performance.now();
            samples.push(now, dx, dy);
            while (samples.length > 3 && now - samples[0] > 100) samples.splice(0, 3);
            lastTouch = now;
        }

        function onUp(e) {
            if (!dragging || e.pointerId !== pointerId) return;
            dragging = false;
            root.classList.remove('is-dragging');
            var now = performance.now();
            // Speed from the last ~100ms of movement; none if the finger paused.
            var sx = 0, sy = 0, first = now;
            for (var i = 0; i < samples.length; i += 3) {
                if (now - samples[i] <= 100) { first = Math.min(first, samples[i]); sx += samples[i + 1]; sy += samples[i + 2]; }
            }
            var last = samples.length ? samples[samples.length - 3] : 0;
            var span = Math.max(now - first, 16);
            if (e.type === 'pointercancel' || now - last > 70) sx = sy = 0;
            var k = radPerPx() * 1000 / span;
            wy = clamp(sx * k, -MAX_SPEED, MAX_SPEED);
            wx = clamp(sy * k, -MAX_SPEED, MAX_SPEED);
            if (calm()) { wx *= 0.4; wy *= 0.4; }
            lastTouch = now;
            wake();
        }

        root.addEventListener('pointerdown', onDown);
        root.addEventListener('pointermove', onMove);
        root.addEventListener('pointerup', onUp);
        root.addEventListener('pointercancel', onUp);
        root.addEventListener('lostpointercapture', onUp);

        /* Keyboard: arrows give it a push */
        root.addEventListener('keydown', function (e) {
            var push = 3.2;
            if (e.key === 'ArrowLeft') wy -= push;
            else if (e.key === 'ArrowRight') wy += push;
            else if (e.key === 'ArrowUp') wx -= push;
            else if (e.key === 'ArrowDown') wx += push;
            else return;
            e.preventDefault();
            wy = clamp(wy, -MAX_SPEED, MAX_SPEED);
            wx = clamp(wx, -MAX_SPEED, MAX_SPEED);
            root.classList.add('was-touched');
            lastTouch = performance.now();
            wake();
        });

        /* Frame loop: runs only while the figure is on screen */
        var raf = 0, visible = false, prev = 0, shown = false;

        function frame(t) {
            raf = 0;
            var dt = prev ? Math.min((t - prev) / 1000, 0.05) : 1 / 60;
            prev = t;

            var k = radPerPx();
            if (dragging) {
                rotateScreen(0, 1, 0, pendX * k);
                rotateScreen(1, 0, 0, pendY * k);
                pendX = pendY = 0;
            } else {
                rotateScreen(0, 1, 0, wy * dt);
                rotateScreen(1, 0, 0, wx * dt);
                var friction = Math.exp(-dt / (calm() ? 0.25 : 0.6));
                wx *= friction; wy *= friction;
                if (Math.abs(wx) < 1e-3) wx = 0;
                if (Math.abs(wy) < 1e-3) wy = 0;
            }

            var idle = !dragging && performance.now() - lastTouch > 1400 && Math.hypot(wx, wy) < 0.6;
            if (!calm()) {
                var target = dragging ? 0 : IDLE_SPIN;
                spin += (target - spin) * (1 - Math.exp(-dt / 0.9));
                q = qMul(q, qAxis(0, 1, 0, spin * dt));   // turn on its own axis
                if (idle) settle(dt);
            }
            q = qNorm(q);

            renderer.draw(qToMat3(q, mat));
            if (!shown) { shown = true; root.classList.add('is-live'); }

            var moving = dragging || wx || wy || (!calm());
            if (visible && moving) raf = requestAnimationFrame(frame);
            else prev = 0;
        }

        function wake() {
            if (!raf && visible) { prev = 0; raf = requestAnimationFrame(frame); }
        }

        function resize() {
            var rect = root.getBoundingClientRect();
            size = Math.max(1, Math.round(rect.width));
            dpr = Math.min(global.devicePixelRatio || 1, 2);
            canvas.width = Math.round(size * dpr);
            canvas.height = Math.round(size * dpr);
            renderer.resize(canvas.width, canvas.height, dpr);
            if (!raf) {
                renderer.draw(qToMat3(q, mat));
                if (!shown) { shown = true; root.classList.add('is-live'); }
            }
        }

        if ('ResizeObserver' in global) new ResizeObserver(resize).observe(root);
        else global.addEventListener('resize', resize);
        resize();

        if ('IntersectionObserver' in global) {
            new IntersectionObserver(function (entries) {
                visible = entries[entries.length - 1].isIntersecting && !document.hidden;
                if (visible) wake();
            }, { rootMargin: '120px 0px' }).observe(root);
        } else {
            visible = true; wake();
        }

        document.addEventListener('visibilitychange', function () {
            if (document.hidden) { visible = false; }
            else { visible = true; wake(); }
        });

        if (reduceMotion && reduceMotion.addEventListener) {
            reduceMotion.addEventListener('change', wake);
        }

        canvas.addEventListener('webglcontextlost', function (e) {
            e.preventDefault();
            root.classList.remove('is-live');
            visible = false;
        });

        return { root: root };
    }

    function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

    global.Donut = {
        mount: mount,
        TORUS: TORUS,
        torusPoint: torusPoint,
        torusNormal: torusNormal
    };
})(window);

// Look B, "Solid": a lit, dark donut with the graticule engraved into its skin
// and the knot as a raised amber cord. WebGL 2; falls back to the still picture.
(function (global) {
    'use strict';
    var D = global.Donut;

    var VERT = [
        '#version 300 es',
        'in vec3 aPos;',
        'in vec3 aNor;',
        'in vec2 aUv;',
        'uniform mat3 uRot;',
        'uniform float uDist;',
        'uniform float uFocal;',
        'out vec3 vPos;',
        'out vec3 vNor;',
        'out vec2 vUv;',
        'void main() {',
        '  vec3 p = uRot * aPos;',
        '  vPos = p;',
        '  vNor = uRot * aNor;',
        '  vUv = aUv;',
        '  float depth = uDist - p.z;',
        '  float near = uDist - 4.0, far = uDist + 4.0;',
        '  float z = (depth - near) / (far - near) * 2.0 - 1.0;',
        '  gl_Position = vec4(p.x * uFocal, p.y * uFocal, z * depth, depth);',
        '}'
    ].join('\n');

    var FRAG = [
        '#version 300 es',
        'precision highp float;',
        'in vec3 vPos;',
        'in vec3 vNor;',
        'in vec2 vUv;',
        'uniform float uDist;',
        'uniform int uKnot;',
        'out vec4 outColor;',
        'void main() {',
        '  vec3 n = normalize(vNor);',
        '  vec3 V = normalize(vec3(0.0, 0.0, uDist) - vPos);',
        '  vec3 L = normalize(vec3(-0.5, 0.8, 0.6));',
        '  vec3 L2 = normalize(vec3(0.7, -0.3, 0.2));',
        '  float ndl = dot(n, L);',
        '  float wrap = clamp(ndl * 0.5 + 0.5, 0.0, 1.0);',
        '  float back = clamp(dot(n, L2), 0.0, 1.0);',
        '  vec3 H = normalize(L + V);',
        '  float nh = max(dot(n, H), 0.0);',
        '  float fres = pow(1.0 - max(dot(n, V), 0.0), 3.0);',
        '  vec3 col;',
        '  if (uKnot == 1) {',
        '    vec3 amber = vec3(0.957, 0.635, 0.349);',
        '    col = amber * (0.32 + 0.95 * wrap * wrap);',
        '    col += amber * 0.18;',
        '    col += vec3(1.0, 0.93, 0.84) * pow(nh, 48.0) * 0.75;',
        '    col += amber * fres * 0.45;',
        '  } else {',
        '    float ao = mix(0.5, 1.0, 0.5 + 0.5 * cos(vUv.y * 6.2831853));',
        '    vec3 base = vec3(0.105, 0.128, 0.165);',
        '    col = base * (0.35 + 1.25 * pow(wrap, 1.6)) * ao;',
        '    col += vec3(0.32, 0.36, 0.42) * back * 0.18 * ao;',
        '    col += vec3(1.0, 0.97, 0.92) * pow(nh, 36.0) * 0.16;',
        '    col += vec3(0.55, 0.62, 0.72) * fres * 0.22;',
        '    vec2 g = vUv * vec2(36.0, 24.0);',
        '    vec2 w = fwidth(g);',
        '    vec2 d = abs(fract(g - 0.5) - 0.5) / max(w, 1e-4);',
        '    float line = 1.0 - clamp(min(d.x, d.y) - 0.15, 0.0, 1.0);',
        '    float fade = clamp(1.0 - max(w.x, w.y) * 1.6, 0.0, 1.0);',
        '    col = mix(col, vec3(0.925, 0.92, 0.905) * (0.35 + 0.65 * wrap), line * fade * 0.26);',
        '  }',
        '  outColor = vec4(col, 1.0);',
        '}'
    ].join('\n');

    function compile(gl, type, src) {
        var s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
        return s;
    }

    function torusMesh(NU, NV) {
        var pos = [], nor = [], uv = [], idx = [];
        for (var i = 0; i <= NU; i++) {
            var u = (i / NU) * Math.PI * 2;
            for (var j = 0; j <= NV; j++) {
                var v = (j / NV) * Math.PI * 2;
                push(pos, D.torusPoint(u, v)); push(nor, D.torusNormal(u, v));
                uv.push(i / NU, j / NV);
            }
        }
        for (i = 0; i < NU; i++) {
            for (j = 0; j < NV; j++) {
                var a = i * (NV + 1) + j, b = a + NV + 1;
                idx.push(a, b, a + 1, b, b + 1, a + 1);
            }
        }
        return { pos: pos, nor: nor, uv: uv, idx: idx };
    }

    // A round cord laid along the knot, sitting on the surface.
    function knotMesh(steps, sides, cord) {
        var T = D.TORUS, pos = [], nor = [], uv = [], idx = [];
        var lift = cord * 0.55;
        function center(t) { return D.torusPoint(T.p * t, T.q * t, lift); }
        for (var i = 0; i <= steps; i++) {
            var t = (i / steps) * Math.PI * 2, e = 1e-3;
            var c = center(t), a = center(t - e), b = center(t + e);
            var tan = norm([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
            var sn = D.torusNormal(T.p * t, T.q * t);
            var side = norm(cross(tan, sn));
            var up = cross(side, tan);
            for (var j = 0; j <= sides; j++) {
                var ang = (j / sides) * Math.PI * 2, cs = Math.cos(ang), sn2 = Math.sin(ang);
                var n = [up[0] * cs + side[0] * sn2, up[1] * cs + side[1] * sn2, up[2] * cs + side[2] * sn2];
                pos.push(c[0] + n[0] * cord, c[1] + n[1] * cord, c[2] + n[2] * cord);
                push(nor, n);
                uv.push(i / steps, j / sides);
            }
        }
        for (i = 0; i < steps; i++) {
            for (j = 0; j < sides; j++) {
                var p0 = i * (sides + 1) + j, p1 = p0 + sides + 1;
                idx.push(p0, p0 + 1, p1, p1, p0 + 1, p1 + 1);
            }
        }
        return { pos: pos, nor: nor, uv: uv, idx: idx };
    }

    function makeSolid(canvas) {
        var gl = canvas.getContext('webgl2', { antialias: true, alpha: true, premultipliedAlpha: true, powerPreference: 'low-power' });
        if (!gl) return null;

        var prog = gl.createProgram();
        gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
        gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
        gl.useProgram(prog);

        var loc = {
            pos: gl.getAttribLocation(prog, 'aPos'),
            nor: gl.getAttribLocation(prog, 'aNor'),
            uv: gl.getAttribLocation(prog, 'aUv'),
            rot: gl.getUniformLocation(prog, 'uRot'),
            dist: gl.getUniformLocation(prog, 'uDist'),
            focal: gl.getUniformLocation(prog, 'uFocal'),
            knot: gl.getUniformLocation(prog, 'uKnot')
        };

        function upload(mesh) {
            var vao = gl.createVertexArray();
            gl.bindVertexArray(vao);
            attr(loc.pos, mesh.pos, 3); attr(loc.nor, mesh.nor, 3); attr(loc.uv, mesh.uv, 2);
            var ib = gl.createBuffer();
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(mesh.idx), gl.STATIC_DRAW);
            gl.bindVertexArray(null);
            return { vao: vao, count: mesh.idx.length };
        }

        function attr(l, data, size) {
            var buf = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, buf);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
            gl.enableVertexAttribArray(l);
            gl.vertexAttribPointer(l, size, gl.FLOAT, false, 0, 0);
        }

        var torus = upload(torusMesh(144, 64));
        var knot = upload(knotMesh(900, 12, 0.062));

        var DIST = 11, RADIUS = D.TORUS.R + D.TORUS.r, MARGIN = 1.1;
        // Clip space spans -1..1; the edge of the bounding sphere lands at 1/MARGIN in any pose.
        var focal = Math.sqrt(DIST * DIST - RADIUS * RADIUS) / RADIUS / MARGIN;

        gl.enable(gl.DEPTH_TEST);
        gl.clearColor(0, 0, 0, 0);
        gl.uniform1f(loc.dist, DIST);
        gl.uniform1f(loc.focal, focal);

        return {
            resize: function (w, h) { gl.viewport(0, 0, w, h); },
            draw: function (m) {
                gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
                gl.uniformMatrix3fv(loc.rot, false, m);
                gl.uniform1i(loc.knot, 0);
                gl.bindVertexArray(torus.vao);
                gl.drawElements(gl.TRIANGLES, torus.count, gl.UNSIGNED_SHORT, 0);
                gl.uniform1i(loc.knot, 1);
                gl.bindVertexArray(knot.vao);
                gl.drawElements(gl.TRIANGLES, knot.count, gl.UNSIGNED_SHORT, 0);
            }
        };
    }

    function push(arr, v) { arr.push(v[0], v[1], v[2]); }
    function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
    function norm(a) { var l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }

    global.DonutSolid = makeSolid;
})(window);

// Start every figure marked data-donut on the page.
(function () {
    if (!window.Donut || !window.DonutSolid) return;
    document.querySelectorAll('[data-donut]').forEach(function (el) {
        window.Donut.mount(el, window.DonutSolid);
    });
})();
