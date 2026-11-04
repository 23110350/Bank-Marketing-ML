/**
 * main.js — Shared site JS
 * - Mobile navbar toggle
 * - Active nav link highlight
 * - Page-specific scroll animations (only run if elements exist)
 *
 * Usage:
 *   1) Create: website/js/main.js (this file)
 *   2) Replace each page's inline <script> ... </script> with:
 *        <script src="js/main.js"></script>
 *      (place it right before </body>)
 */

(function () {
  "use strict";

  // -------------------------------
  // Mobile Menu Toggle (global)
  // -------------------------------
  function toggleMenu() {
    const navLinks = document.getElementById("navLinks");
    if (!navLinks) return;
    navLinks.classList.toggle("open");
  }

  // Make it available for onclick="toggleMenu()"
  window.toggleMenu = toggleMenu;

  // -------------------------------
  // Active nav link highlight
  // -------------------------------
  function setActiveNavLink() {
    const links = document.querySelectorAll(".nav-links a");
    if (!links.length) return;

    // Normalize current URL (ignore hash)
    const current = window.location.href.split("#")[0];

    links.forEach((link) => {
      const href = (link.href || "").split("#")[0];
      if (href === current) link.classList.add("active");
    });
  }

  // -------------------------------
  // Helpers
  // -------------------------------
  function createObserver({ threshold = 0.2, rootMargin = "0px", once = true } = {}) {
    return new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;

          // Expect observe() wrapper to set entry.target.__onVisible
          if (typeof entry.target.__onVisible === "function") {
            entry.target.__onVisible();
          }

          if (once) observer.unobserve(entry.target);
        });
      },
      { threshold, rootMargin }
    );
  }

  function observeSimpleAddClass(selector, className, { threshold = 0.2 } = {}) {
    const nodes = document.querySelectorAll(selector);
    if (!nodes.length) return;

    const obs = createObserver({ threshold, once: true });
    nodes.forEach((node) => {
      node.__onVisible = () => node.classList.add(className);
      obs.observe(node);
    });
  }

  function observeStaggerAddClass(selector, className, { threshold = 0.2, stepMs = 120 } = {}) {
    const nodes = document.querySelectorAll(selector);
    if (!nodes.length) return;

    const obs = new IntersectionObserver(
      (entries) => {
        // Sort for stable staggering
        const visible = entries
          .filter((e) => e.isIntersecting)
          .map((e) => e.target);

        visible.forEach((node) => {
          const index = node.__staggerIndex ?? 0;
          setTimeout(() => node.classList.add(className), index * stepMs);
        });
      },
      { threshold }
    );

    nodes.forEach((node, i) => {
      node.__staggerIndex = i;
      obs.observe(node);
    });
  }

  // -------------------------------
  // Page-specific animations
  // -------------------------------
  function initAnimations() {
    // EDA
    observeSimpleAddClass(".viz-img", "viz-img-visible", { threshold: 0.2 });
    observeStaggerAddClass(".summary-card", "card-visible", { threshold: 0.1, stepMs: 100 });

    // Roadmap
    observeSimpleAddClass(".timeline-item", "tl-visible", { threshold: 0.1 });

    // Preprocessing
    observeSimpleAddClass(".pre-step-card", "step-visible", { threshold: 0.1 });

    // Models (Logistic / DT / RF / XGB)
    observeStaggerAddClass(".metric-result-card", "metric-visible", {
      threshold: 0.2,
      stepMs: 120,
    });

    // Random Forest extra (trees)
    observeStaggerAddClass(".rf-tree-box", "rf-tree-visible", {
      threshold: 0.2,
      stepMs: 100,
    });

    // XGBoost extra (boost steps)
    observeStaggerAddClass(".boost-step", "boost-visible", {
      threshold: 0.2,
      stepMs: 150,
    });

    // Results page (animated metric bars)
    const mcBars = document.querySelectorAll(".mc-fill");
    if (mcBars.length) {
      const barObs = createObserver({ threshold: 0.3, once: true });
      mcBars.forEach((bar) => {
        bar.__onVisible = () => {
          const targetWidth = bar.getAttribute("data-width");
          if (targetWidth != null) bar.style.width = targetWidth + "%";
        };
        barObs.observe(bar);
      });
    }

    // Results page (winner card)
    observeSimpleAddClass(".winner-card", "winner-visible", { threshold: 0.2 });

    // Conclusion
    observeStaggerAddClass(".finding-card", "finding-visible", { threshold: 0.1, stepMs: 100 });
    observeStaggerAddClass(".rec-item", "rec-visible", { threshold: 0.1, stepMs: 120 });

    // Team
    observeStaggerAddClass(".team-card", "team-card-visible", { threshold: 0.1, stepMs: 150 });
    observeStaggerAddClass(".project-info-card", "pi-visible", { threshold: 0.1, stepMs: 100 });

    // Dataset (your old code was not really animating; leaving it out on purpose)
    // If you want bar animation here too, tell me what exact behavior you want.
  }

  // -------------------------------
  // Boot
  // -------------------------------
  document.addEventListener("DOMContentLoaded", () => {
    setActiveNavLink();
    initAnimations();
  });
})();