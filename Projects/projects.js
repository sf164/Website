// Projects page: one project open at a time, a subject filter, and a
// preview panel beside the list on wide screens.
// Without this script every project simply shows in full, one after another.
(function () {
    var wide = window.matchMedia('(min-width: 901px)');
    var canHover = window.matchMedia('(hover: hover) and (pointer: fine)');
    var panel = document.querySelector('.work-panel');
    var filterGroup = document.querySelector('.work-filters');

    var projects = Array.prototype.map.call(document.querySelectorAll('.work-item'), function (el) {
        return {
            el: el,
            toggle: el.querySelector('.work-toggle'),
            detail: el.querySelector('.work-detail'),
            category: el.getAttribute('data-category')
        };
    });
    if (!projects.length) return;

    var active = null;

    function visible(p) {
        return !p.el.hidden;
    }

    function setActive(p) {
        active = p;
        projects.forEach(function (q) {
            var on = q === p;
            q.detail.classList.toggle('is-active', on);
            q.toggle.setAttribute('aria-expanded', on ? 'true' : 'false');
        });
    }

    // Wide screens: every detail lives in the panel; narrow screens: under its own title.
    function place() {
        projects.forEach(function (p) {
            if (wide.matches) {
                panel.appendChild(p.detail);
            } else {
                p.el.appendChild(p.detail);
            }
        });
        if (wide.matches && (!active || !visible(active))) {
            setActive(projects.filter(visible)[0] || null);
        }
    }

    projects.forEach(function (p) {
        p.toggle.addEventListener('click', function () {
            if (!wide.matches && active === p) {
                setActive(null); // tapping the open project on a phone closes it
            } else {
                setActive(p);
            }
        });
        p.toggle.addEventListener('focus', function () {
            if (wide.matches) setActive(p);
        });
        p.toggle.addEventListener('mouseenter', function () {
            if (wide.matches && canHover.matches && active !== p) setActive(p);
        });
    });

    // Subject filter
    if (filterGroup) {
        filterGroup.hidden = false;
        var buttons = filterGroup.querySelectorAll('.filter');
        Array.prototype.forEach.call(buttons, function (button) {
            button.addEventListener('click', function () {
                var cat = button.getAttribute('data-filter');
                Array.prototype.forEach.call(buttons, function (b) {
                    b.setAttribute('aria-pressed', b === button ? 'true' : 'false');
                });
                projects.forEach(function (p) {
                    p.el.hidden = !(cat === 'all' || p.category === cat);
                    if (p.el.hidden) p.el.classList.add('is-visible'); // never leave it waiting on a reveal
                });
                if (!active || !visible(active)) {
                    setActive(wide.matches ? (projects.filter(visible)[0] || null) : null);
                }
            });
        });
    }

    if (wide.addEventListener) {
        wide.addEventListener('change', place);
    } else if (wide.addListener) {
        wide.addListener(place);
    }

    setActive(projects[0]);
    place();
})();
