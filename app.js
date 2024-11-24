let currentSlideIndex = 0; // Tracks the current slide index

// Function to show a specific slide by its index
function showSlide(index) {
    const items = document.querySelectorAll('.slide .item'); // Select all slides
    const nextButton = document.querySelector('.next');
    const prevButton = document.querySelector('.prev');

    if (!items.length) return; // Exit if no slides exist

    // Ensure the index is within valid bounds
    currentSlideIndex = Math.min(Math.max(index, 0), items.length - 1);

    // Hide all slides
    items.forEach((item, idx) => {
        item.style.display = idx === currentSlideIndex ? 'flex' : 'none';
    });

    // Enable or disable navigation buttons
    prevButton.disabled = currentSlideIndex === 0;
    nextButton.disabled = currentSlideIndex === items.length - 1;
}

// Function to initialize slideshow navigation
function initializeSlideshow() {
    const nextButton = document.querySelector('.next');
    const prevButton = document.querySelector('.prev');

    // Remove any existing event listeners to prevent duplication
    nextButton.replaceWith(nextButton.cloneNode(true));
    prevButton.replaceWith(prevButton.cloneNode(true));

    const newNextButton = document.querySelector('.next');
    const newPrevButton = document.querySelector('.prev');

    // Show the first slide initially
    showSlide(currentSlideIndex);

    // Event listener for the Next button
    newNextButton.addEventListener('click', () => {
        const items = document.querySelectorAll('.slide .item');
        if (currentSlideIndex < items.length - 1) {
            showSlide(currentSlideIndex + 1); // Move to the next slide
        }
    });

    // Event listener for the Previous button
    newPrevButton.addEventListener('click', () => {
        if (currentSlideIndex > 0) {
            showSlide(currentSlideIndex - 1); // Move to the previous slide
        }
    });
}

// Functions to toggle between sections
function showVisualizations() {
    const visualizationSection = document.getElementById('visualizationSection');
    const introSection = document.getElementById('intro');

    visualizationSection.style.display = 'block';
    introSection.style.display = 'none';

    // Reset the slideshow state
    currentSlideIndex = 0; // Reset index to start from the first slide
    showSlide(0); // Ensure the first slide is displayed
    initializeSlideshow(); // Ensure slideshow navigation is properly initialized
}

function showIntro() {
    const introSection = document.getElementById('intro');
    const visualizationSection = document.getElementById('visualizationSection');

    introSection.style.display = 'block';
    visualizationSection.style.display = 'none';
}

// Initial page setup
window.onload = () => {
    const visualizationSection = document.getElementById('visualizationSection');
    const introSection = document.getElementById('intro');

    // Ensure only the intro section is displayed initially
    visualizationSection.style.display = 'none';
    introSection.style.display = 'block';
};
