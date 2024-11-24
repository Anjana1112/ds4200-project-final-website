let currentSlideIndex = 0; 
function showSlide(index) {
    const items = document.querySelectorAll('.slide .item'); 
    const nextButton = document.querySelector('.next');
    const prevButton = document.querySelector('.prev');

    if (!items.length) return;

    currentSlideIndex = Math.min(Math.max(index, 0), items.length - 1);

    items.forEach((item, idx) => {
        item.style.display = idx === currentSlideIndex ? 'flex' : 'none';
    });

    prevButton.disabled = currentSlideIndex === 0;
    nextButton.disabled = currentSlideIndex === items.length - 1;
}

function initializeSlideshow() {
    const nextButton = document.querySelector('.next');
    const prevButton = document.querySelector('.prev');

    nextButton.replaceWith(nextButton.cloneNode(true));
    prevButton.replaceWith(prevButton.cloneNode(true));

    const newNextButton = document.querySelector('.next');
    const newPrevButton = document.querySelector('.prev');

    showSlide(currentSlideIndex);

    newNextButton.addEventListener('click', () => {
        const items = document.querySelectorAll('.slide .item');
        if (currentSlideIndex < items.length - 1) {
            showSlide(currentSlideIndex + 1);
        }
    });

    newPrevButton.addEventListener('click', () => {
        if (currentSlideIndex > 0) {
            showSlide(currentSlideIndex - 1);
        }
    });
}

function showVisualizations() {
    const visualizationSection = document.getElementById('visualizationSection');
    const introSection = document.getElementById('intro');

    visualizationSection.style.display = 'block';
    introSection.style.display = 'none';

    currentSlideIndex = 0; 
    showSlide(0); 
    initializeSlideshow(); 
}

function showIntro() {
    const introSection = document.getElementById('intro');
    const visualizationSection = document.getElementById('visualizationSection');

    introSection.style.display = 'block';
    visualizationSection.style.display = 'none';
}

window.onload = () => {
    const visualizationSection = document.getElementById('visualizationSection');
    const introSection = document.getElementById('intro');

    visualizationSection.style.display = 'none';
    introSection.style.display = 'block';
};
