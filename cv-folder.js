// CV folder: click the folder, its doors swing open, the CV rises out of the
// pocket and grows into full pages you can read. Close (button, Esc or a click
// outside) sends it back into the folder.
(function () {
    'use strict';

    var PAGE_W = 816;

    function wait(ms) {
        return new Promise(function (resolve) { setTimeout(resolve, ms); });
    }

    // Scale a page laid out at 816px wide to fit its box.
    function fit(box) {
        var page = box.querySelector('.cvp');
        if (page) page.style.transform = 'scale(' + box.clientWidth / PAGE_W + ')';
    }

    // The transform that makes an element sitting at rect `at` appear at rect `to`.
    function moveTo(to, at) {
        return 'translate(' + (to.left - at.left) + 'px, ' + (to.top - at.top) + 'px) scale(' + to.width / at.width + ')';
    }

    // A copy of a page for decoration: hidden from screen readers, links made plain.
    function copyOf(page) {
        var copy = page.cloneNode(true);
        copy.setAttribute('aria-hidden', 'true');
        copy.querySelectorAll('a').forEach(function (a) {
            var span = document.createElement('span');
            span.innerHTML = a.innerHTML;
            a.replaceWith(span);
        });
        return copy;
    }

    function setup(root) {
        var button = root.querySelector('.cvw');
        var paper = root.querySelector('.cvw-paper');
        var dialog = document.getElementById(button.getAttribute('aria-controls'));
        if (!dialog || typeof dialog.showModal !== 'function') return;

        var sheets = Array.prototype.slice.call(dialog.querySelectorAll('.cvr-sheet'));
        var first = sheets[0];
        var scroller = dialog.querySelector('.cvr-scroll');
        var scrim = dialog.querySelector('.cvr-scrim');
        var bar = dialog.querySelector('.cvr-bar');
        var reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
        var state = 'closed';

        // The folder's paper shows the first page, scaled down.
        function setPaper(sheet) {
            paper.innerHTML = '';
            paper.appendChild(copyOf(sheet.querySelector('.cvp')));
            fit(paper);
        }
        setPaper(first);

        var boxes = [paper].concat(sheets);
        if ('ResizeObserver' in window) {
            var ro = new ResizeObserver(function (entries) { entries.forEach(function (e) { fit(e.target); }); });
            boxes.forEach(function (b) { ro.observe(b); });
        }
        boxes.forEach(fit);

        function flyer(sheet, rect) {
            var f = document.createElement('div');
            f.className = 'cvr-flyer';
            f.setAttribute('aria-hidden', 'true');
            f.style.left = rect.left + 'px';
            f.style.top = rect.top + 'px';
            f.style.width = rect.width + 'px';
            f.style.height = rect.height + 'px';
            var page = copyOf(sheet.querySelector('.cvp'));
            page.style.transform = 'scale(' + rect.width / PAGE_W + ')';
            f.appendChild(page);
            dialog.appendChild(f);
            return f;
        }

        // The page that takes up most of the screen right now.
        function mostVisible() {
            var best = first, bestArea = -1, h = window.innerHeight;
            sheets.forEach(function (s) {
                var r = s.getBoundingClientRect();
                var area = Math.max(0, Math.min(r.bottom, h) - Math.max(r.top, 0));
                if (area > bestArea) { bestArea = area; best = s; }
            });
            return best;
        }

        function show() {
            document.documentElement.classList.add('cv-lock');
            dialog.showModal();
            scroller.scrollTop = 0;
            boxes.forEach(fit);
        }

        function hide() {
            dialog.close();
            sheets.forEach(function (s) { s.style.visibility = ''; });
            [scrim, bar, scroller, dialog].concat(sheets).forEach(function (el) {
                el.getAnimations().forEach(function (a) { a.cancel(); });
            });
            document.documentElement.classList.remove('cv-lock');
            button.focus({ preventScroll: true });
        }

        async function open() {
            if (state !== 'closed') return;
            state = 'opening';
            root.classList.remove('is-peek');

            if (reduce.matches) {
                show();
                dialog.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 160 });
                state = 'open';
                return;
            }

            root.classList.add('is-open');            // doors swing out
            await wait(200);
            root.classList.add('is-lifted');          // the CV rises out of the pocket
            await wait(400);

            var from = paper.getBoundingClientRect();
            show();
            var to = first.getBoundingClientRect();
            var f = flyer(first, to);
            first.style.visibility = 'hidden';
            root.classList.add('is-flying');

            var flight = f.animate(
                [{ transform: moveTo(from, to) }, { transform: 'none' }],
                { duration: 560, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' }
            );
            scrim.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 380, easing: 'ease-out' });
            bar.animate(
                [{ opacity: 0, transform: 'translateY(-6px)' }, { opacity: 1, transform: 'none' }],
                { duration: 260, delay: 320, easing: 'ease-out', fill: 'backwards' }
            );
            sheets.slice(1).forEach(function (s, i) {
                s.animate(
                    [{ opacity: 0, transform: 'translateY(16px)' }, { opacity: 1, transform: 'none' }],
                    { duration: 320, delay: 380 + i * 60, easing: 'ease-out', fill: 'backwards' }
                );
            });
            await flight.finished;
            first.style.visibility = '';
            f.remove();
            state = 'open';
        }

        async function close() {
            if (state !== 'open') return;
            state = 'closing';

            if (reduce.matches) {
                await dialog.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 140, fill: 'forwards' }).finished;
                hide();
                root.classList.remove('is-flying', 'is-lifted', 'is-open');
                state = 'closed';
                return;
            }

            // Send back whichever page is on screen.
            var sheet = mostVisible();
            if (sheet !== first) setPaper(sheet);
            var from = sheet.getBoundingClientRect();
            var to = paper.getBoundingClientRect();
            var f = flyer(sheet, from);
            sheet.style.visibility = 'hidden';

            var flight = f.animate(
                [{ transform: 'none' }, { transform: moveTo(to, from) }],
                { duration: 480, easing: 'cubic-bezier(0.65, 0, 0.35, 1)', fill: 'forwards' }
            );
            scroller.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 220, fill: 'forwards' });
            scrim.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 440, easing: 'ease-in', fill: 'forwards' });
            bar.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 160, fill: 'forwards' });
            await flight.finished;

            root.classList.remove('is-flying');
            f.remove();
            hide();
            await wait(40);
            root.classList.remove('is-lifted');       // back into the pocket
            await wait(220);
            root.classList.remove('is-open');         // doors close
            await wait(460);
            if (sheet !== first) setPaper(first);     // the first page goes back on top, out of sight
            state = 'closed';
        }

        button.addEventListener('click', function (e) {
            if (e.metaKey || e.ctrlKey || e.shiftKey) return; // a modified click opens the PDF itself
            e.preventDefault();
            open();
        });

        dialog.addEventListener('cancel', function (e) {
            e.preventDefault();
            close();
        });

        dialog.addEventListener('click', function (e) {
            if (e.target.closest('[data-cv-close]') || e.target === scrim || e.target === scroller || e.target.classList.contains('cvr-pages')) close();
        });

        dialog.querySelectorAll('.cvr-download[aria-disabled="true"]').forEach(function (el) {
            el.addEventListener('click', function (e) { e.preventDefault(); });
        });

        // On touch screens there is no hover, so the doors part once when the folder scrolls into view.
        if ('IntersectionObserver' in window && window.matchMedia('(hover: none)').matches && !reduce.matches) {
            var io = new IntersectionObserver(function (entries) {
                if (!entries[0].isIntersecting) return;
                io.disconnect();
                setTimeout(function () {
                    if (state !== 'closed') return;
                    root.classList.add('is-peek');
                    setTimeout(function () { root.classList.remove('is-peek'); }, 900);
                }, 400);
            }, { threshold: 0.7 });
            io.observe(button);
        }
    }

    document.querySelectorAll('[data-cv-folder]').forEach(setup);
})();
