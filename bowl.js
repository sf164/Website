// Interactive steepest-descent bowl (Year 5, APM462).
// The graph of f(x) = ½ x·Qx drawn as a dark bowl, with the steepest descent
// path traced down it in amber. On the page it turns slowly on its own; clicking
// it opens a larger view where you can turn it, change the step size and edit Q.
// donut.js provides the turning controls.
(function (global) {
    'use strict';
    var D = global.Donut;
    if (!D) return;

    /* ---------- The problem: f(x) = ½ x·Qx ---------- */
    var A1 = 2.5;                       // half-width of the bowl along its long axis (fixed)
    var H = 2.2;                        // depth of the bowl (fixed)
    var BOTTOM = -0.72 * H;             // keeps the bowl centred in the frame when tilted
    var MAX_STEPS = 2000, DONE = 0.006;
    var LIMIT = 50;                     // largest entry allowed in Q

    // Everything that depends on Q. The bowl is scaled so its rim is always the level
    // set through the ends of the long axis, so only the shape changes with Q.
    var M = null;

    function eigen(a, b, c) {
        var mid = (a + c) / 2, rad = Math.hypot((a - c) / 2, b);
        var l1 = mid - rad, l2 = mid + rad, e1;
        if (Math.abs(b) > 1e-12) e1 = norm2([b, l1 - a]);
        else e1 = a <= c ? [1, 0] : [0, 1];
        return { l1: l1, l2: l2, e1: e1, e2: [-e1[1], e1[0]] };
    }

    function setQ(a, b, c) {
        if (!(a > 0) || !(a * c - b * b > 0)) return false;   // Sylvester's criterion
        var e = eigen(a, b, c);
        var C = 0.5 * e.l1 * A1 * A1;
        var m = { a: a, b: b, c: c, l1: e.l1, l2: e.l2, e1: e.e1, e2: e.e2, C: C, A2: Math.sqrt(2 * C / e.l2) };
        // The start that makes the exact step zigzag the most (equality in
        // Kantorovich's inequality), placed just inside the rim.
        var s = Math.sqrt(2 * 0.93 * 0.93 * C / (e.l1 * e.l2 * (e.l1 + e.l2)));
        m.start = toXZ(m, -e.l2 * s, e.l1 * s);
        M = m;
        return true;
    }

    function toXZ(m, u, v) { return [u * m.e1[0] + v * m.e2[0], u * m.e1[1] + v * m.e2[1]]; }
    function f(x, z) { return 0.5 * (M.a * x * x + 2 * M.b * x * z + M.c * z * z); }
    function grad(x, z) { return [M.a * x + M.b * z, M.b * x + M.c * z]; }
    function height(x, z) { return BOTTOM + H * f(x, z) / M.C; }
    function upNormal(x, z) {           // unit normal pointing into the bowl
        var k = H / M.C, g = grad(x, z);
        return norm([-k * g[0], 1, -k * g[1]]);
    }

    // Iterates of x_{k+1} = x_k − α_k ∇f(x_k). alpha = null means the exact step
    // α_k = |g|² / (g·Qg) from Example 7.6.
    function descend(alpha) {
        var x = M.start[0], z = M.start[1], pts = [[x, z]], out = false;
        for (var k = 0; k < MAX_STEPS; k++) {
            if (Math.hypot(x, z) < DONE) break;
            var g = grad(x, z), gx = g[0], gz = g[1];
            var a = alpha;
            if (a == null) {
                var qg = grad(gx, gz);
                a = (gx * gx + gz * gz) / (gx * qg[0] + gz * qg[1]);
            }
            var nx = x - a * gx, nz = z - a * gz;
            if (f(nx, nz) > M.C) {          // left the bowl: stop at the rim
                var lo = 0, hi = 1;
                for (var i = 0; i < 40; i++) {
                    var mm = (lo + hi) / 2;
                    if (f(x + mm * (nx - x), z + mm * (nz - z)) > M.C) hi = mm; else lo = mm;
                }
                pts.push([x + lo * (nx - x), z + lo * (nz - z)]);
                out = true;
                break;
            }
            x = nx; z = nz;
            pts.push([x, z]);
        }
        return { pts: pts, out: out, steps: pts.length - 1, done: !out && Math.hypot(x, z) < DONE };
    }

    /* ---------- Meshes ---------- */
    function bowlMesh(NR, NT) {
        var pos = [], nor = [], uv = [], idx = [];
        for (var i = 0; i <= NT; i++) {
            var t = (i / NT) * Math.PI * 2;
            for (var j = 0; j <= NR; j++) {
                var r = j / NR;
                var p = toXZ(M, r * A1 * Math.cos(t), r * M.A2 * Math.sin(t)), x = p[0], z = p[1];
                push(pos, [x, height(x, z), z]); push(nor, upNormal(x, z));
                uv.push(i / NT, r * r);   // r² is f / f(rim), so rings are level sets
            }
        }
        for (i = 0; i < NT; i++) {
            for (j = 0; j < NR; j++) {
                var a = i * (NR + 1) + j, b = a + NR + 1;
                idx.push(a, b, a + 1, b, b + 1, a + 1);
            }
        }
        return { pos: pos, nor: nor, uv: uv, idx: idx };
    }

    function addTube(m, centers, normals, radius, sides) {
        var base = m.pos.length / 3;
        for (var i = 0; i < centers.length; i++) {
            var a = centers[Math.max(i - 1, 0)], b = centers[Math.min(i + 1, centers.length - 1)];
            var tan = norm([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
            var side = norm(cross(tan, normals[i]));
            var up = cross(side, tan);
            for (var j = 0; j <= sides; j++) {
                var ang = (j / sides) * Math.PI * 2, cs = Math.cos(ang), sn = Math.sin(ang);
                var n = [up[0] * cs + side[0] * sn, up[1] * cs + side[1] * sn, up[2] * cs + side[2] * sn];
                m.pos.push(centers[i][0] + n[0] * radius, centers[i][1] + n[1] * radius, centers[i][2] + n[2] * radius);
                push(m.nor, n); m.uv.push(0, 0);
            }
        }
        for (i = 0; i < centers.length - 1; i++) {
            for (j = 0; j < sides; j++) {
                var p0 = base + i * (sides + 1) + j, p1 = p0 + sides + 1;
                m.idx.push(p0, p0 + 1, p1, p1, p0 + 1, p1 + 1);
            }
        }
    }

    function addSphere(m, c, radius, NA, NB) {
        var base = m.pos.length / 3;
        for (var i = 0; i <= NB; i++) {
            var th = (i / NB) * Math.PI;
            for (var j = 0; j <= NA; j++) {
                var ph = (j / NA) * Math.PI * 2;
                var n = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
                m.pos.push(c[0] + n[0] * radius, c[1] + n[1] * radius, c[2] + n[2] * radius);
                push(m.nor, n); m.uv.push(0, 0);
            }
        }
        for (i = 0; i < NB; i++) {
            for (j = 0; j < NA; j++) {
                var a = base + i * (NA + 1) + j, b = a + NA + 1;
                m.idx.push(a, b, a + 1, b, b + 1, a + 1);
            }
        }
    }

    function addRing(m, c, R, r, NU, NV) {   // a small flat torus lying on the bottom
        var base = m.pos.length / 3;
        for (var i = 0; i <= NU; i++) {
            var u = (i / NU) * Math.PI * 2;
            for (var j = 0; j <= NV; j++) {
                var v = (j / NV) * Math.PI * 2, ring = R + r * Math.cos(v);
                m.pos.push(c[0] + ring * Math.cos(u), c[1] + r * Math.sin(v), c[2] + ring * Math.sin(u));
                push(m.nor, [Math.cos(v) * Math.cos(u), Math.sin(v), Math.cos(v) * Math.sin(u)]); m.uv.push(0, 0);
            }
        }
        for (i = 0; i < NU; i++) {
            for (j = 0; j < NV; j++) {
                var a = base + i * (NV + 1) + j, b = a + NV + 1;
                m.idx.push(a, b, a + 1, b, b + 1, a + 1);
            }
        }
    }

    var CORD = 0.034, LIFT = 0.05;

    // The path as an amber cord lying on the bowl, with a bead at every iterate.
    function pathMesh(run) {
        var m = { pos: [], nor: [], uv: [], idx: [] };
        var centers = [], normals = [], pts = run.pts;
        for (var k = 0; k < pts.length - 1; k++) {
            var ax = pts[k][0], az = pts[k][1], bx = pts[k + 1][0], bz = pts[k + 1][1];
            var n = Math.max(2, Math.ceil(Math.hypot(bx - ax, bz - az) / 0.03));
            for (var s = k ? 1 : 0; s <= n; s++) {
                var x = ax + (bx - ax) * s / n, z = az + (bz - az) * s / n, up = upNormal(x, z);
                centers.push([x + up[0] * LIFT, height(x, z) + up[1] * LIFT, z + up[2] * LIFT]);
                normals.push(up);
            }
        }
        if (centers.length > 1) addTube(m, centers, normals, CORD, 8);
        var beads = Math.min(pts.length, 90);
        for (k = 0; k < beads; k++) {
            var px = pts[k][0], pz = pts[k][1], u = upNormal(px, pz);
            addSphere(m, [px + u[0] * LIFT, height(px, pz) + u[1] * LIFT, pz + u[2] * LIFT], k === 0 ? 0.1 : 0.058, 10, 6);
        }
        return m;
    }

    /* ---------- Rendering (WebGL 2, same look as the donut) ---------- */
    var VERT = [
        '#version 300 es',
        'in vec3 aPos;', 'in vec3 aNor;', 'in vec2 aUv;',
        'uniform mat3 uRot;', 'uniform float uDist;', 'uniform float uFocal;',
        'out vec3 vPos;', 'out vec3 vNor;', 'out vec2 vUv;',
        'void main() {',
        '  vec3 p = uRot * aPos;',
        '  vPos = p; vNor = uRot * aNor; vUv = aUv;',
        '  float depth = uDist - p.z;',
        '  float near = uDist - 4.0, far = uDist + 4.0;',
        '  float z = (depth - near) / (far - near) * 2.0 - 1.0;',
        '  gl_Position = vec4(p.x * uFocal, p.y * uFocal, z * depth, depth);',
        '}'
    ].join('\n');

    var FRAG = [
        '#version 300 es',
        'precision highp float;',
        'in vec3 vPos;', 'in vec3 vNor;', 'in vec2 vUv;',
        'uniform float uDist;', 'uniform int uMode;',
        'out vec4 outColor;',
        'void main() {',
        '  vec3 V = normalize(vec3(0.0, 0.0, uDist) - vPos);',
        '  vec3 n = normalize(vNor);',
        '  if (uMode == 0 && dot(n, V) < 0.0) n = -n;',   // the bowl is seen from both sides
        '  vec3 L = normalize(vec3(-0.5, 0.8, 0.6));',
        '  vec3 L2 = normalize(vec3(0.7, -0.3, 0.2));',
        '  float wrap = clamp(dot(n, L) * 0.5 + 0.5, 0.0, 1.0);',
        '  float back = clamp(dot(n, L2), 0.0, 1.0);',
        '  vec3 Hh = normalize(L + V);',
        '  float nh = max(dot(n, Hh), 0.0);',
        '  float fres = pow(1.0 - max(dot(n, V), 0.0), 3.0);',
        '  vec3 col;',
        '  if (uMode == 1) {',
        '    vec3 amber = vec3(0.957, 0.635, 0.349);',
        '    col = amber * (0.32 + 0.95 * wrap * wrap) + amber * 0.18;',
        '    col += vec3(1.0, 0.93, 0.84) * pow(nh, 48.0) * 0.75;',
        '    col += amber * fres * 0.45;',
        '  } else if (uMode == 2) {',
        '    vec3 ink = vec3(0.925, 0.92, 0.905);',
        '    col = ink * (0.35 + 0.6 * wrap);',
        '  } else {',
        '    float ao = mix(0.62, 1.0, sqrt(vUv.y));',
        '    vec3 base = vec3(0.105, 0.128, 0.165);',
        '    col = base * (0.35 + 1.25 * pow(wrap, 1.6)) * ao;',
        '    col += vec3(0.32, 0.36, 0.42) * back * 0.18 * ao;',
        '    col += vec3(1.0, 0.97, 0.92) * pow(nh, 36.0) * 0.16;',
        '    col += vec3(0.55, 0.62, 0.72) * fres * 0.22;',
        '    vec2 g = vUv * vec2(48.0, 12.0);',
        '    vec2 w = fwidth(g);',
        '    vec2 d = abs(fract(g - 0.5) - 0.5) / max(w, 1e-4);',
        '    float spokes = 1.0 - clamp(d.x - 0.15, 0.0, 1.0);',
        '    float rings = 1.0 - clamp(d.y - 0.15, 0.0, 1.0);',
        '    float fade = clamp(1.0 - w.x * 1.6, 0.0, 1.0);',
        '    float line = max(spokes * fade * 0.7, rings);',
        '    col = mix(col, vec3(0.925, 0.92, 0.905) * (0.35 + 0.65 * wrap), line * 0.26);',
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

    function makeBowl(canvas) {
        var gl = canvas.getContext('webgl2', { antialias: true, alpha: true, premultipliedAlpha: true, powerPreference: 'low-power' });
        if (!gl) return null;
        var prog = gl.createProgram();
        gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
        gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;
        gl.useProgram(prog);

        var loc = {
            pos: gl.getAttribLocation(prog, 'aPos'), nor: gl.getAttribLocation(prog, 'aNor'), uv: gl.getAttribLocation(prog, 'aUv'),
            rot: gl.getUniformLocation(prog, 'uRot'), dist: gl.getUniformLocation(prog, 'uDist'),
            focal: gl.getUniformLocation(prog, 'uFocal'), mode: gl.getUniformLocation(prog, 'uMode')
        };

        function makeVao() {
            var o = { vao: gl.createVertexArray(), bufs: [gl.createBuffer(), gl.createBuffer(), gl.createBuffer()], ib: gl.createBuffer(), count: 0 };
            gl.bindVertexArray(o.vao);
            [[loc.pos, 3], [loc.nor, 3], [loc.uv, 2]].forEach(function (a, i) {
                gl.bindBuffer(gl.ARRAY_BUFFER, o.bufs[i]);
                gl.enableVertexAttribArray(a[0]);
                gl.vertexAttribPointer(a[0], a[1], gl.FLOAT, false, 0, 0);
            });
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, o.ib);
            gl.bindVertexArray(null);
            return o;
        }

        function fill(o, mesh, usage) {
            gl.bindVertexArray(o.vao);
            [mesh.pos, mesh.nor, mesh.uv].forEach(function (data, i) {
                gl.bindBuffer(gl.ARRAY_BUFFER, o.bufs[i]);
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), usage);
            });
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, o.ib);
            gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(mesh.idx), usage);
            gl.bindVertexArray(null);
            o.count = mesh.idx.length;
        }

        var bowl = makeVao(), path = makeVao(), mark = makeVao();
        var m = { pos: [], nor: [], uv: [], idx: [] };
        addRing(m, [0, BOTTOM + 0.035, 0], 0.13, 0.024, 40, 10);
        fill(mark, m, gl.STATIC_DRAW);

        var DIST = 11, RADIUS = Math.max(Math.hypot(A1, H + BOTTOM), -BOTTOM) + 0.12, MARGIN = 1.14;
        var focal = Math.sqrt(DIST * DIST - RADIUS * RADIUS) / RADIUS / MARGIN;
        gl.enable(gl.DEPTH_TEST);
        gl.clearColor(0, 0, 0, 0);
        gl.uniform1f(loc.dist, DIST);
        gl.uniform1f(loc.focal, focal);

        function drawOne(o, mode) {
            if (!o.count) return;
            gl.uniform1i(loc.mode, mode);
            gl.bindVertexArray(o.vao);
            gl.drawElements(gl.TRIANGLES, o.count, gl.UNSIGNED_INT, 0);
        }

        return {
            setShape: function () { fill(bowl, bowlMesh(44, 160), gl.DYNAMIC_DRAW); },
            setRun: function (run) { fill(path, pathMesh(run), gl.DYNAMIC_DRAW); },
            resize: function (w, h) { gl.viewport(0, 0, w, h); },
            draw: function (mat) {
                gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
                gl.uniformMatrix3fv(loc.rot, false, mat);
                drawOne(bowl, 0);
                drawOne(mark, 2);
                drawOne(path, 1);
            }
        };
    }

    function push(arr, v) { arr.push(v[0], v[1], v[2]); }
    function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
    function norm(a) { var l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
    function norm2(a) { var l = Math.hypot(a[0], a[1]) || 1; return [a[0] / l, a[1] / l]; }
    function moveTo(to, at) {
        return 'translate(' + (to.left - at.left) + 'px, ' + (to.top - at.top) + 'px) scale(' + to.width / at.width + ')';
    }
    function fmt(v) {
        var s = Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2);
        if (s.indexOf('.') >= 0) s = s.replace(/0+$/, '').replace(/\.$/, '');
        return s.replace('-', '−');
    }

    /* ---------- Wiring ---------- */
    var TILT = 60 * Math.PI / 180;
    var PRESETS = { round: [3, 0, 3], valley: [5, 4, 5] };

    function setup(small) {
        var dialog = document.getElementById(small.getAttribute('data-bowl'));
        var openBtn = small.parentNode.querySelector('[data-bowl-open]');
        setQ(3, 0, 3);                                // the default Q is diag(3, 3)

        var views = [];
        function addView(root, interactive, hint) {
            var renderer = null;
            var fig = D.mount(root, function (canvas) { renderer = makeBowl(canvas); return renderer; },
                { tilt: TILT, interactive: interactive, hint: hint });
            if (!fig || !renderer) return null;
            var v = { fig: fig, renderer: renderer };
            views.push(v);
            return v;
        }

        var smallView = addView(small, false, 'Click to explore');
        if (!smallView) return;                       // no WebGL: keep the still picture

        var useExact = true, ratio = 0.9;             // slider position, as a fraction of 2/λmax
        var run = null;

        function stepFromRatio() { return ratio * 2 / M.l2; }

        /* The larger view */
        var bigView = null, ui = {};
        if (dialog && typeof dialog.showModal === 'function') {
            bigView = addView(dialog.querySelector('[data-bowl-big]'), true, 'Drag to turn');
            ui = {
                exact: dialog.querySelector('[data-descent-exact]'),
                slider: dialog.querySelector('[data-descent-step]'),
                readout: dialog.querySelector('[data-descent-readout]'),
                controls: dialog.querySelector('[data-descent-controls]'),
                q11: dialog.querySelector('[data-q="11"]'),
                q12: dialog.querySelector('[data-q="12"]'),
                q21: dialog.querySelector('[data-q="21"]'),
                q22: dialog.querySelector('[data-q="22"]'),
                facts: dialog.querySelector('[data-q-facts]'),
                limits: dialog.querySelector('[data-q-limits]'),
                error: dialog.querySelector('[data-q-error]'),
                presets: dialog.querySelectorAll('[data-q-preset]')
            };
        }
        if (openBtn) openBtn.hidden = !bigView;

        function redraw(shapeChanged) {
            run = descend(useExact ? null : stepFromRatio());
            views.forEach(function (v) {
                if (shapeChanged) v.renderer.setShape();
                v.renderer.setRun(run);
                v.fig.refresh();
            });
            updatePanel();
        }

        function updatePanel() {
            if (!bigView) return;
            var alpha = stepFromRatio();
            ui.exact.setAttribute('aria-pressed', useExact ? 'true' : 'false');
            ui.controls.classList.toggle('is-exact', useExact);
            ui.slider.value = String(ratio);
            ui.slider.setAttribute('aria-valuetext', 'step size ' + fmt(alpha));
            var label = useExact ? 'Exact step' : 'α = ' + fmt(alpha);
            var what;
            if (run.out) what = 'overshoots out of the bowl';
            else if (run.done) what = 'reaches the bottom in ' + run.steps + (run.steps === 1 ? ' step' : ' steps');
            else what = 'still going after ' + MAX_STEPS + ' steps';
            ui.readout.textContent = label + ' · ' + what;
            ui.facts.textContent = 'Eigenvalues ' + fmt(M.l1) + ' and ' + fmt(M.l2) + ' · condition number ' + fmt(M.l2 / M.l1);
            ui.limits.textContent = 'A fixed step reaches the bottom only below 2/λmax = ' + fmt(2 / M.l2) +
                ', and is fastest at 2/(λmin + λmax) = ' + fmt(2 / (M.l1 + M.l2)) + '.';
            Array.prototype.forEach.call(ui.presets, function (b) {
                var p = PRESETS[b.getAttribute('data-q-preset')];
                b.setAttribute('aria-pressed', p[0] === M.a && p[1] === M.b && p[2] === M.c ? 'true' : 'false');
            });
        }

        function readQ() {
            var a = parseFloat(ui.q11.value), b = parseFloat(ui.q12.value), c = parseFloat(ui.q22.value);
            var bad = [a, b, c].some(function (v) { return !isFinite(v) || Math.abs(v) > LIMIT; });
            var ok = !bad && setQ(a, b, c);
            ui.error.hidden = ok;
            if (!ok) {
                ui.error.textContent = bad
                    ? 'Each entry needs to be a number between −' + LIMIT + ' and ' + LIMIT + '.'
                    : 'This Q is not positive-definite, so f has no lowest point. By Sylvester’s criterion the top-left entry and det Q = ac − b² both need to be positive. The bowl keeps the last Q that worked.';
            }
            [ui.q11, ui.q12, ui.q21, ui.q22].forEach(function (el) { el.setAttribute('aria-invalid', ok ? 'false' : 'true'); });
            if (ok) redraw(true);
            else Array.prototype.forEach.call(ui.presets, function (btn) { btn.setAttribute('aria-pressed', 'false'); });
        }

        if (bigView) {
            ui.exact.addEventListener('click', function () { useExact = true; redraw(false); });
            ui.slider.addEventListener('input', function () { useExact = false; ratio = parseFloat(ui.slider.value); redraw(false); });
            ui.q11.addEventListener('input', readQ);
            ui.q22.addEventListener('input', readQ);
            // Q is symmetric: the two off-diagonal entries always hold the same number.
            ui.q12.addEventListener('input', function () { ui.q21.value = ui.q12.value; readQ(); });
            ui.q21.addEventListener('input', function () { ui.q12.value = ui.q21.value; readQ(); });
            Array.prototype.forEach.call(ui.presets, function (b) {
                b.addEventListener('click', function () {
                    var p = PRESETS[b.getAttribute('data-q-preset')];
                    ui.q11.value = p[0]; ui.q12.value = p[1]; ui.q21.value = p[1]; ui.q22.value = p[2];
                    readQ();
                });
            });
        }

        redraw(true);
        if (!bigView || !openBtn) return;

        /* Opening and closing: the bowl grows out of the page into the larger view */
        var reduce = global.matchMedia('(prefers-reduced-motion: reduce)');
        var stage = dialog.querySelector('.bowlr-stage');
        var side = dialog.querySelector('.bowlr-side');
        var panel = dialog.querySelector('.bowlr-panel');
        var scrim = dialog.querySelector('.bowlr-scrim');
        var closeBtn = dialog.querySelector('.bowlr-close');
        var state = 'closed';
        var clear = [{ backgroundColor: 'rgba(11, 14, 19, 0)', borderColor: 'rgba(236, 235, 231, 0)' }];

        async function open() {
            if (state !== 'closed') return;
            state = 'opening';
            var from = small.getBoundingClientRect();
            document.documentElement.classList.add('cv-lock');
            dialog.showModal();
            bigView.fig.setPose(smallView.fig.getPose());
            closeBtn.focus({ preventScroll: true });
            if (reduce.matches) {
                await dialog.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 160 }).finished;
                state = 'open';
                return;
            }
            var to = stage.getBoundingClientRect();
            scrim.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 360, easing: 'ease-out' });
            panel.animate(clear.concat([{}]), { duration: 420, easing: 'ease-out' });
            side.animate([{ opacity: 0, transform: 'translateY(12px)' }, { opacity: 1, transform: 'none' }],
                { duration: 320, delay: 220, easing: 'ease-out', fill: 'backwards' });
            closeBtn.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 240, delay: 260, fill: 'backwards' });
            await stage.animate([{ transform: moveTo(from, to) }, { transform: 'none' }],
                { duration: 520, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' }).finished;
            state = 'open';
        }

        async function close() {
            if (state !== 'open') return;
            state = 'closing';
            smallView.fig.setPose(bigView.fig.getPose());   // the small one carries on from here
            if (reduce.matches) {
                await dialog.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 140, fill: 'forwards' }).finished;
            } else {
                var from = stage.getBoundingClientRect(), to = small.getBoundingClientRect();
                side.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 160, fill: 'forwards' });
                closeBtn.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 160, fill: 'forwards' });
                panel.animate([{}].concat(clear), { duration: 300, fill: 'forwards' });
                scrim.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 420, easing: 'ease-in', fill: 'forwards' });
                await stage.animate([{ transform: 'none' }, { transform: moveTo(to, from) }],
                    { duration: 440, easing: 'cubic-bezier(0.65, 0, 0.35, 1)', fill: 'forwards' }).finished;
            }
            dialog.close();
            dialog.getAnimations({ subtree: true }).forEach(function (a) { a.cancel(); });
            document.documentElement.classList.remove('cv-lock');
            openBtn.focus({ preventScroll: true });
            state = 'closed';
        }

        openBtn.addEventListener('click', open);
        dialog.addEventListener('cancel', function (e) { e.preventDefault(); close(); });
        dialog.addEventListener('click', function (e) {
            if (e.target.closest('[data-bowl-close]') || e.target === scrim || e.target.classList.contains('bowlr-scroll')) close();
        });
    }

    document.querySelectorAll('[data-bowl]').forEach(setup);
})(window);
