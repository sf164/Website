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
