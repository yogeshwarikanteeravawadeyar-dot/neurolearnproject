// Main JavaScript functionality
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });

    // Auto-dismiss alerts after 5 seconds
    setTimeout(function() {
        var alerts = document.querySelectorAll('.alert');
        alerts.forEach(function(alert) {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);
});

// Speech recognition functionality
function startSpeechRecognition(text) {
    fetch('/textspeech', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({text: text})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showFeedback('Correct! Well done!', 'success');
        } else {
            showFeedback('Try again!', 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showFeedback('An error occurred. Please try again.', 'error');
    });
}

function showFeedback(message, type) {
    const feedbackDiv = document.createElement('div');
    feedbackDiv.className = `alert alert-${type === 'success' ? 'success' : 'danger'} alert-dismissible fade show`;
    feedbackDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.querySelector('.container').prepend(feedbackDiv);
    
    setTimeout(() => {
        feedbackDiv.remove();
    }, 5000);
}

// Gesture detection functions
function startGestureDetection() {
    // This would connect to your gesture detection backend
    console.log('Starting gesture detection...');
}

// Eye blink detection functions
function startEyeBlinkDetection() {
    // This would connect to your eye blink detection backend
    console.log('Starting eye blink detection...');
}