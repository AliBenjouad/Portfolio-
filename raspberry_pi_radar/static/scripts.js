$(document).ready(function() {
    const radarCanvas = document.getElementById('radar');
    const ctx = radarCanvas.getContext('2d');
    const radius = radarCanvas.width / 2;
    const center = { x: radius, y: radius };
    let maxDetectionDistance = 80;  // Starting range
    let isScanning = false;  // Variable to track if scanning is in progress

    // Update maximum detection range
    $('#setMaxRange').click(function() {
        maxDetectionDistance = parseInt($('#inputMaxRange').val());
        drawRadarBackground();  // Redraw the background with new range
    });

    // Button to start manual scan or sweep
    $('#startScan').click(function() {
        toggleScanning();
    });

    $('#startSweep').click(function() {  // Ensure this button is correctly handled
        sweepRadar();  // Perform a 180° sweep
    });

    // Toggle scanning on and off
    function toggleScanning() {
        isScanning = !isScanning;
        if (isScanning) {
            $('#startScan').text('Stop Scan');
            continuousManualScan();  // Start continuous manual scanning
        } else {
            $('#startScan').text('Start Scan');
            $('#status').text('Scan stopped.');
        }
    }

    // Perform continuous manual scan based on input angle
    function continuousManualScan() {
        const angle = parseInt($('#inputAngle').val());  // Get angle from input
        if (isScanning) {
            updateRadar(angle);
            setTimeout(continuousManualScan, 1000);  // Continue scanning after 1 second
        }
    }

    // Sweep the radar from 0 to 180 degrees
    function sweepRadar() {
        for (let angle = 0; angle <= 180; angle += 15) {
            setTimeout(function() {
                updateRadar(angle);
            }, 1000 * (angle / 15));  // Delay for sensor reading
        }
    }

    // AJAX request to backend for scanning at a given angle
    function updateRadar(angle) {
        $.ajax({
            url: '/scan',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ angle: angle, max_range: maxDetectionDistance }),
            success: function(data) {
                if (data.distance !== 'Error') {
                    $('#status').text(`Scanning at angle ${angle}°...`);
                    drawRadar(data);
                } else {
                    $('#status').text(`Error in scanning at angle ${angle}`);
                }
            },
            error: function(xhr, status, error) {
                $('#status').text('Scan failed. Retrying...');
                console.error("Error - Status:", status, "Error:", error);
            }
        });
    }

    // Draws radar visualization based on sensor data
    function drawRadar(data) {
        ctx.clearRect(0, 0, radarCanvas.width, radarCanvas.height);
        drawRadarBackground();
        drawOrientationPoints();
        if (data.distance !== 'Error' && data.distance <= maxDetectionDistance) {
            drawDetectedPoint(data);
        }
    }

    // Draws radar background
    function drawRadarBackground() {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, radarCanvas.width, radarCanvas.height);
        ctx.strokeStyle = 'green';
        ctx.beginPath();
        ctx.arc(center.x, center.y, radius, 0, 2 * Math.PI);
        ctx.stroke();
    }

    // Draws orientation points (N, E, S, W)
    function drawOrientationPoints() {
        const points = [
            { angle: 0, label: 'E' },
            { angle: 90, label: 'N' },
            { angle: 180, label: 'W' },
            { angle: 270, label: 'S' }
        ];
        points.forEach(point => {
            const angleRad = point.angle * Math.PI / 180;
            const labelOffset = 20;
            const x = center.x + (radius + labelOffset) * Math.cos(angleRad);
            const y = center.y - (radius + labelOffset) * Math.sin(angleRad);
            ctx.fillStyle = 'lime';
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
            ctx.font = '16px Arial';
            ctx.fillText(point.label, x - 10, y + 10);
        });
    }

    // Draws a detected point
    function drawDetectedPoint(data) {
        const angleRad = (90 - data.angle) * Math.PI / 180;
        const scaledDistance = data.distance / maxDetectionDistance * radius;
        const x = center.x + scaledDistance * Math.cos(angleRad);
        const y = center.y - scaledDistance * Math.sin(angleRad);
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = 'red';
        ctx.fill();
    }
});
