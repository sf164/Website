// Entrance animations for the landing page.
// Without this script (or with reduced motion) everything simply shows.
(function () {
    var root = document.documentElement;

    // Hero: fade the picture in and stagger the title, sentence and buttons.
    document.querySelectorAll('[data-hero]').forEach(function (el, i) {
        el.style.setProperty('--delay', 250 + i * 90 + 'ms');
    });

    // No requestAnimationFrame here: it never fires while a tab is hidden,
    // which would leave the first screen blank.
    function heroIn() {
        document.body.classList.add('is-loaded');
    }

    var heroImage = document.querySelector('.hero-image');
    if (!heroImage || heroImage.complete) {
        setTimeout(heroIn, 30);
    } else {
        heroImage.addEventListener('load', heroIn, { once: true });
        heroImage.addEventListener('error', heroIn, { once: true });
        setTimeout(heroIn, 1500); // never wait forever for the picture
    }

    // Sections: each part rises in as it scrolls into view, in order.
    var items = Array.prototype.slice.call(document.querySelectorAll('[data-reveal]'));

    if (!('IntersectionObserver' in window)) {
        items.forEach(function (el) { el.classList.add('is-visible'); });
        return;
    }

    // Parts that come into view together rise one after another, 70ms apart.
    var observer = new IntersectionObserver(function (entries) {
        var n = 0;
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                entry.target.style.setProperty('--delay', Math.min(n++, 6) * 70 + 'ms');
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target);
            }
        });
    }, { rootMargin: '0px 0px -8% 0px' });

    items.forEach(function (el) { observer.observe(el); });

    // Keyboard users: anything that receives focus shows at once.
    root.addEventListener('focusin', function (e) {
        var el = e.target.closest && e.target.closest('[data-reveal]');
        if (el) el.classList.add('is-visible');
    });
})();

// Copy-email button and the footer's "Last updated" date.
(function () {
    // Copy email: puts the address on the clipboard; if that isn't allowed, selects it instead.
    document.querySelectorAll('[data-copy]').forEach(function (button) {
        var label = button.textContent;
        var status = button.parentElement.querySelector('[data-copy-status]');
        var timer;

        function done(text) {
            button.textContent = text;
            button.classList.add('is-copied');
            if (status) status.textContent = text;
            clearTimeout(timer);
            timer = setTimeout(function () {
                button.textContent = label;
                button.classList.remove('is-copied');
            }, 2000);
        }

        function selectInstead() {
            var link = button.parentElement.querySelector('.contact-email');
            if (!link) return;
            var range = document.createRange();
            range.selectNodeContents(link);
            var sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
            done('Press Ctrl+C to copy');
        }

        button.addEventListener('click', function () {
            var text = button.getAttribute('data-copy');
            if (navigator.clipboard && window.isSecureContext) {
                navigator.clipboard.writeText(text).then(function () { done('Copied'); }, selectInstead);
            } else {
                selectInstead();
            }
        });
    });

    // Last updated: GitHub Pages sends the time of the latest publish with every page.
    // If the browser didn't get that time, it reports "now", so the date typed in the page stays.
    var MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
        'August', 'September', 'October', 'November', 'December'];
    var stamp = new Date(document.lastModified);
    if (!isNaN(stamp) && Date.now() - stamp.getTime() > 60000) {
        document.querySelectorAll('[data-last-updated]').forEach(function (el) {
            el.textContent = stamp.getDate() + ' ' + MONTHS[stamp.getMonth()] + ' ' + stamp.getFullYear();
            el.setAttribute('datetime', stamp.getFullYear() + '-' +
                String(stamp.getMonth() + 1).padStart(2, '0') + '-' + String(stamp.getDate()).padStart(2, '0'));
        });
    }
})();
