let currentSlideIndex = 0;

function showSlide(index) {
    const items = document.querySelectorAll('.slide .item');
    const nextButton = document.querySelector('.next');
    const prevButton = document.querySelector('.prev');

    if (!items.length) return;

    // Ensure the index is within bounds
    currentSlideIndex = Math.max(0, Math.min(index, items.length - 1));

    // Show the current slide and hide others
    items.forEach((item, idx) => {
        item.style.display = idx === currentSlideIndex ? 'flex' : 'none';
    });

    // Enable or disable buttons based on the current index
    prevButton.disabled = currentSlideIndex === 0;
    nextButton.disabled = currentSlideIndex === items.length - 1;
}

function initializeSlideshow() {
    const nextButton = document.querySelector('.next');
    const prevButton = document.querySelector('.prev');

    // Ensure the buttons are functional
    nextButton.addEventListener('click', () => {
        showSlide(currentSlideIndex + 1);
    });

    prevButton.addEventListener('click', () => {
        showSlide(currentSlideIndex - 1);
    });

    // Show the first slide on initialization
    showSlide(currentSlideIndex);
}

function showVisualizations() {
    const visualizationSection = document.getElementById('visualizationSection');
    const introSection = document.getElementById('intro');

    visualizationSection.style.display = 'block';
    introSection.style.display = 'none';

    // Initialize slideshow when visualization is shown
    currentSlideIndex = 0; // Reset to the first slide
    showSlide(0); 
}

function showIntro() {
    const introSection = document.getElementById('intro');
    const visualizationSection = document.getElementById('visualizationSection');

    introSection.style.display = 'block';
    visualizationSection.style.display = 'none';
}

// Ensure everything is set up after the page loads
window.onload = () => {
    const visualizationSection = document.getElementById('visualizationSection');
    const introSection = document.getElementById('intro');

    visualizationSection.style.display = 'none'; // Initially hide visualizations
    introSection.style.display = 'block'; // Show intro

    // Initialize slideshow functionality
    initializeSlideshow();
};

