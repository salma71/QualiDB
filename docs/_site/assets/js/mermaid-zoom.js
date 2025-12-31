// Mermaid diagram zoom functionality
(function () {
    'use strict';

    // Wait for both DOM and Mermaid to be ready
    function initZoom() {
        // Check if Mermaid is loaded
        if (typeof mermaid === 'undefined') {
            setTimeout(initZoom, 100);
            return;
        }

        // Wait for Mermaid to finish rendering
        function waitForMermaid() {
            const mermaidDiagrams = document.querySelectorAll('.mermaid');

            // Check if diagrams are rendered (have SVG content)
            const rendered = Array.from(mermaidDiagrams).some(function (diagram) {
                return diagram.querySelector('svg') !== null;
            });

            if (!rendered && mermaidDiagrams.length > 0) {
                // Not rendered yet, wait a bit more
                setTimeout(waitForMermaid, 200);
                return;
            }

            // Diagrams are rendered, set up zoom
            setupZoom(mermaidDiagrams);
        }

        // Start checking after a short delay
        setTimeout(waitForMermaid, 500);
    }

    function setupZoom(mermaidDiagrams) {
        if (mermaidDiagrams.length === 0) return;

        // Create zoom overlay first (only once)
        let overlay = document.getElementById('mermaid-zoom-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'mermaid-zoom-overlay';
            overlay.id = 'mermaid-zoom-overlay';

            const closeBtn = document.createElement('button');
            closeBtn.className = 'mermaid-zoom-close';
            closeBtn.innerHTML = '&times;';
            closeBtn.setAttribute('aria-label', 'Close');
            closeBtn.addEventListener('click', closeZoom);

            overlay.appendChild(closeBtn);
            document.body.appendChild(overlay);

            // Close on overlay click (but not on diagram click)
            overlay.addEventListener('click', function (e) {
                if (e.target === overlay || e.target === closeBtn) {
                    closeZoom();
                }
            });

            // Close on Escape key
            document.addEventListener('keydown', function (e) {
                if (e.key === 'Escape' && overlay.classList.contains('active')) {
                    closeZoom();
                }
            });
        }

        // Wrap each diagram in a container
        mermaidDiagrams.forEach(function (diagram) {
            // Skip if already wrapped
            if (diagram.parentElement && diagram.parentElement.classList.contains('mermaid-container')) {
                return;
            }

            const container = document.createElement('div');
            container.className = 'mermaid-container';
            diagram.parentNode.insertBefore(container, diagram);
            container.appendChild(diagram);

            // Add zoom hint
            const hint = document.createElement('div');
            hint.className = 'mermaid-zoom-hint';
            hint.textContent = 'Click to enlarge';
            container.appendChild(hint);

            // Add click handler for zoom
            container.addEventListener('click', function (e) {
                // Don't trigger if clicking on links or other interactive elements
                if (e.target.tagName === 'A' || e.target.closest('a')) {
                    return;
                }
                openZoom(diagram, container);
            });
        });
    }

    // Start initialization
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initZoom);
    } else {
        initZoom();
    }
})();

function openZoom(diagram, container) {
    const overlay = document.getElementById('mermaid-zoom-overlay');
    if (!overlay) return;

    // Clear overlay first
    const existingDiagram = overlay.querySelector('.mermaid');
    if (existingDiagram) {
        existingDiagram.remove();
    }

    // Clone the diagram with all its content
    const cloned = diagram.cloneNode(true);
    cloned.style.maxWidth = 'none';
    cloned.style.width = 'auto';
    cloned.style.cursor = 'zoom-out';

    overlay.appendChild(cloned);
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Prevent body scroll
    document.documentElement.style.overflow = 'hidden';
}

function closeZoom() {
    const overlay = document.getElementById('mermaid-zoom-overlay');
    if (!overlay) return;

    overlay.classList.remove('active');
    document.body.style.overflow = '';
    document.documentElement.style.overflow = '';

    // Clean up cloned diagram
    const cloned = overlay.querySelector('.mermaid');
    if (cloned) {
        cloned.remove();
    }
}
