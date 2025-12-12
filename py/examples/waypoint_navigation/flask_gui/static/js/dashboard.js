// Socket.IO connection
const socket = io();

// D3.js plot variables
let svg, xScale, yScale, plotWidth, plotHeight;
let detectionSvg, detectionXScale, detectionYScale, detectionPlotWidth, detectionPlotHeight;

// ==================== Socket.IO Events ====================

socket.on('connect', () => {
    console.log('Connected to server');
    updateConnectionStatus(true);
    addLog('Connected to navigation server', 'success');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    updateConnectionStatus(false);
    addLog('Disconnected from server', 'error');
});

socket.on('status_update', (data) => {
    updateRobotStatus(data);
});

socket.on('success', (data) => {
    addLog(data.message, 'success');
});

socket.on('error', (data) => {
    addLog(data.message, 'error');
});

socket.on('warning', (data) => {
    addLog(data.message, 'warning');
});

socket.on('status', (data) => {
    addLog(data.message, 'info');
});

socket.on('nav_log', (data) => {
    // Navigation terminal output
    addLog('[NAV] ' + data.message, 'info');
});

// ==================== UI Update Functions ====================

function updateConnectionStatus(connected) {
    const badge = document.getElementById('connection-status');
    if (connected) {
        badge.textContent = 'Connected';
        badge.classList.remove('disconnected');
        badge.classList.add('connected');
    } else {
        badge.textContent = 'Disconnected';
        badge.classList.remove('connected');
        badge.classList.add('disconnected');
    }
}

function updateRobotStatus(status) {
    // Update pose
    if (status.pose) {
        document.getElementById('pos-x').textContent = status.pose.x.toFixed(2) + ' m';
        document.getElementById('pos-y').textContent = status.pose.y.toFixed(2) + ' m';
        document.getElementById('heading').textContent = status.pose.heading_deg.toFixed(1) + '°';
    }

    // Update waypoint
    document.getElementById('waypoint').textContent =
        `${status.current_waypoint} / ${status.total_waypoints}`;

    // Update track status with color coding
    const trackBadge = document.getElementById('track-status');
    trackBadge.textContent = status.track_status;
    trackBadge.style.color = '#ffffff';  // White text

    // Color code based on status
    if (status.track_status === 'TRACK_FINISHED') {
        trackBadge.style.backgroundColor = '#BDDB94';  // Green for finished
    } else if (status.track_status === 'ABORTED' || status.track_status === 'TRACK_ABORTED' || status.track_status === 'TRACK_FAILED') {
        trackBadge.style.backgroundColor = '#ef4444';  // Red for aborted/failed
    } else {
        // IDLE, TRACK_FOLLOWING, or other states - match stop button (warning/orange)
        trackBadge.style.background = 'linear-gradient(135deg, #BDDB94 100%, #d97706 100%)';
    }

    // Update filter convergence with color coding
    const filterBadge = document.getElementById('filter-converged');
    filterBadge.textContent = status.filter_converged ? 'YES' : 'NO';
    if (status.filter_converged) {
        filterBadge.style.color = '#ffffff';  // White text
        filterBadge.style.backgroundColor = '#BDDB94';  // Green background for YES
    } else {
        filterBadge.style.color = '#ffffff';  // White text
        filterBadge.style.backgroundColor = '#ef4444';  // Red background for NO
    }

    // Update button states
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');

    if (status.navigation_running) {
        btnStart.disabled = true;
        btnStop.disabled = false;
    } else {
        btnStart.disabled = false;
        btnStop.disabled = true;
    }
}

function addLog(message, type = 'info') {
    const logContainer = document.getElementById('log-container');
    const timestamp = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${timestamp}] ${message}`;
    logContainer.appendChild(entry);

    // Auto-scroll to bottom
    logContainer.scrollTop = logContainer.scrollHeight;

    // Limit to 100 entries
    while (logContainer.children.length > 100) {
        logContainer.removeChild(logContainer.firstChild);
    }
}

// ==================== Control Functions ====================

function startNavigation() {
    socket.emit('start_navigation');
    addLog('Sending start command...', 'info');
}

function stopNavigation() {
    socket.emit('stop_navigation');
    addLog('Sending stop command...', 'info');
}

function emergencyStop() {
    if (confirm('⚠️ EMERGENCY STOP - Are you sure?')) {
        socket.emit('emergency_stop');
        addLog('🛑 EMERGENCY STOP ACTIVATED', 'error');
    }
}

// ==================== Waypoint Plot (D3.js) ====================

function initializePlot() {
    const container = d3.select('#plot-container');
    const containerNode = container.node();
    const containerWidth = containerNode.clientWidth;
    const containerHeight = containerNode.clientHeight;

    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    plotWidth = containerWidth - margin.left - margin.right;
    plotHeight = containerHeight - margin.top - margin.bottom;

    svg = container.append('svg')
        .attr('width', containerWidth)
        .attr('height', containerHeight)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Initial scales (will be updated with data)
    xScale = d3.scaleLinear().range([0, plotWidth]);
    yScale = d3.scaleLinear().range([plotHeight, 0]);

    // Add axes
    svg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${plotHeight})`)
        .style('color', '#0294D0');

    svg.append('g')
        .attr('class', 'y-axis')
        .style('color', '#0294D0');

    // Add grid (horizontal lines)
    svg.append('g')
        .attr('class', 'grid-y')
        .style('stroke', '#374151')
        .style('stroke-opacity', 0.3)
        .style('stroke-dasharray', '2,2');

    // Add grid (vertical lines)
    svg.append('g')
        .attr('class', 'grid-x')
        .attr('transform', `translate(0,${plotHeight})`)
        .style('stroke', '#374151')
        .style('stroke-opacity', 0.3)
        .style('stroke-dasharray', '2,2');

    // Add axis labels
    svg.append('text')
        .attr('class', 'x-label')
        .attr('text-anchor', 'middle')
        .attr('x', plotWidth / 2)
        .attr('y', plotHeight + 35)
        .style('fill', '#0294D0')
        .text('X (meters)');

    svg.append('text')
        .attr('class', 'y-label')
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .attr('x', -plotHeight / 2)
        .attr('y', -45)
        .style('fill', '#0294D0')
        .text('Y (meters)');
}

function updatePlot(data) {
    if (!svg) return;

    const waypoints = data.waypoints || [];
    const robot = data.robot;
    const currentIndex = data.current_index || 0;

    if (waypoints.length === 0) return;

    // Update scales
    const xExtent = d3.extent(waypoints, d => d.x);
    const yExtent = d3.extent(waypoints, d => d.y);

    // Add padding
    const xPadding = (xExtent[1] - xExtent[0]) * 0.1 || 1;
    const yPadding = (yExtent[1] - yExtent[0]) * 0.1 || 1;

    xScale.domain([xExtent[0] - xPadding, xExtent[1] + xPadding]);
    yScale.domain([yExtent[0] - yPadding, yExtent[1] + yPadding]);

    // Update axes
    svg.select('.x-axis').call(d3.axisBottom(xScale).ticks(5));
    svg.select('.y-axis').call(d3.axisLeft(yScale).ticks(5));

    // Update grid (horizontal lines)
    svg.select('.grid-y')
        .call(d3.axisLeft(yScale)
            .ticks(5)
            .tickSize(-plotWidth)
            .tickFormat(''));

    // Update grid (vertical lines)
    svg.select('.grid-x')
        .call(d3.axisBottom(xScale)
            .ticks(5)
            .tickSize(-plotHeight)
            .tickFormat(''));

    // Update waypoint markers
    const circles = svg.selectAll('.waypoint')
        .data(waypoints, d => d.index);

    circles.enter()
        .append('circle')
        .attr('class', 'waypoint')
        .merge(circles)
        .attr('cx', d => xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .attr('r', d => d.index === currentIndex ? 8 : 5)
        .attr('fill', d => {
            if (d.index === currentIndex) return '#3b82f6';
            if (d.index < currentIndex) return '#BDDB94';
            return '#ef4444';
        })
        .attr('stroke', d => d.index === currentIndex ? 'white' : 'none')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .append('title')
        .text(d => `Waypoint ${d.index}\nX: ${d.x.toFixed(2)}\nY: ${d.y.toFixed(2)}`);

    circles.exit().remove();

    // Update robot position
    if (robot && robot.x !== undefined && robot.y !== undefined) {
        let robotMarker = svg.select('.robot-marker');

        if (robotMarker.empty()) {
            robotMarker = svg.append('g').attr('class', 'robot-marker');

            // Robot body (circle)
            robotMarker.append('circle')
                .attr('r', 10)
                .attr('fill', '#FBB24B ')
                .attr('stroke', 'white')
                .attr('stroke-width', 2);

            // Heading arrow
            robotMarker.append('path')
                .attr('class', 'heading-arrow')
                .attr('fill', 'white');
        }

        // Update robot position
        robotMarker.attr('transform',
            `translate(${xScale(robot.x)}, ${yScale(robot.y)})`);

        // Update heading arrow
        if (robot.heading !== undefined) {
            const arrowPath = `M 0,-12 L 4,0 L 0,8 L -4,0 Z`;
            robotMarker.select('.heading-arrow')
                .attr('d', arrowPath)
                .attr('transform', `rotate(${-robot.heading * 180 / Math.PI})`);
        }
    }
}

function fetchPlotData() {
    fetch('/plot_data')
        .then(response => response.json())
        .then(data => {
            console.log('Plot data received:', data);
            if (data.waypoints && data.waypoints.length > 0) {
                console.log(`Loaded ${data.waypoints.length} waypoints for plot`);
            } else {
                console.warn('No waypoints in plot data');
            }
            updatePlot(data);
        })
        .catch(error => {
            console.error('Error fetching plot data:', error);
            addLog(`Plot data error: ${error.message}`, 'error');
        });
}

// ==================== Detection Scatter Plot ====================

function initializeDetectionPlot() {
    const container = d3.select('#detection-plot-container');
    const containerNode = container.node();
    const containerWidth = containerNode.clientWidth;
    const containerHeight = containerNode.clientHeight;

    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    detectionPlotWidth = containerWidth - margin.left - margin.right;
    detectionPlotHeight = containerHeight - margin.top - margin.bottom;

    detectionSvg = container.append('svg')
        .attr('width', containerWidth)
        .attr('height', containerHeight)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales: Y (left/right), X (forward)
    // Note: X-axis shows Y (left/right) with LEFT being positive, so we flip the domain
    const MAX_RANGE_X = 5.0;  // meters
    const MAX_RANGE_Y = 10.0; // meters
    detectionXScale = d3.scaleLinear().domain([MAX_RANGE_X, -MAX_RANGE_X]).range([0, detectionPlotWidth]); // Flipped for left=positive
    detectionYScale = d3.scaleLinear().domain([0, MAX_RANGE_Y]).range([detectionPlotHeight, 0]);

    // Add axes
    detectionSvg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${detectionPlotHeight})`)
        .style('color', '#0294D0')
        .call(d3.axisBottom(detectionXScale).ticks(5));

    detectionSvg.append('g')
        .attr('class', 'y-axis')
        .style('color', '#0294D0')
        .call(d3.axisLeft(detectionYScale).ticks(5));

    // Add axis labels
    detectionSvg.append('text')
        .attr('text-anchor', 'middle')
        .attr('x', detectionPlotWidth / 2)
        .attr('y', detectionPlotHeight + 35)
        .style('fill', '#0294D0')
        .text('Y (meters) [left +, right -]');

    detectionSvg.append('text')
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .attr('x', -detectionPlotHeight / 2)
        .attr('y', -45)
        .style('fill', '#0294D0')
        .text('X (meters) [forward +]');

    // Draw FOV wedge (95° HFOV for RGB camera)
    drawFOVWedge();

    // Draw range rings
    drawRangeRings();
}

function drawFOVWedge() {
    const MAX_RANGE_X = 5.0;
    const MAX_RANGE_Y = 10.0;
    const RGB_HFOV_DEG = 95;
    const halfFOV = (RGB_HFOV_DEG / 2) * Math.PI / 180;

    // Generate FOV wedge boundaries
    // In robot frame: +Y is LEFT, -Y is RIGHT
    // At distance z forward, FOV extends to ±z*tan(halfFOV) in Y
    const points = 200;
    const pathPoints = [];

    // Build path: start at origin, trace left edge, then right edge back
    pathPoints.push([detectionXScale(0), detectionYScale(0)]); // Origin

    // Left edge (positive Y = left)
    for (let i = 0; i <= points; i++) {
        const z = (MAX_RANGE_Y * i) / points;
        const y = z * Math.tan(halfFOV); // Positive Y (left)
        const yClipped = Math.min(MAX_RANGE_X, y);
        pathPoints.push([detectionXScale(yClipped), detectionYScale(z)]);
    }

    // Right edge (negative Y = right), trace backwards
    for (let i = points; i >= 0; i--) {
        const z = (MAX_RANGE_Y * i) / points;
        const y = -z * Math.tan(halfFOV); // Negative Y (right)
        const yClipped = Math.max(-MAX_RANGE_X, y);
        pathPoints.push([detectionXScale(yClipped), detectionYScale(z)]);
    }

    // Draw filled wedge
    detectionSvg.append('path')
        .attr('d', d3.line()(pathPoints) + 'Z')
        .attr('fill', '#3b82f6')
        .attr('fill-opacity', 0.1)
        .attr('stroke', 'none');

    // Draw FOV edges (dashed lines)
    // Left edge line
    const leftY = Math.min(MAX_RANGE_X, MAX_RANGE_Y * Math.tan(halfFOV));
    detectionSvg.append('line')
        .attr('x1', detectionXScale(0))
        .attr('y1', detectionYScale(0))
        .attr('x2', detectionXScale(leftY))
        .attr('y2', detectionYScale(MAX_RANGE_Y))
        .attr('stroke', '#6b7280')
        .attr('stroke-dasharray', '4,2')
        .attr('stroke-width', 1);

    // Right edge line
    const rightY = Math.max(-MAX_RANGE_X, -MAX_RANGE_Y * Math.tan(halfFOV));
    detectionSvg.append('line')
        .attr('x1', detectionXScale(0))
        .attr('y1', detectionYScale(0))
        .attr('x2', detectionXScale(rightY))
        .attr('y2', detectionYScale(MAX_RANGE_Y))
        .attr('stroke', '#6b7280ff')
        .attr('stroke-dasharray', '4,2')
        .attr('stroke-width', 1);
}

function drawRangeRings() {
    const MAX_RANGE_X = 5.0;
    const MAX_RANGE_Y = 10.0;
    const RGB_HFOV_DEG = 95;
    const halfFOV = (RGB_HFOV_DEG / 2) * Math.PI / 180;
    const ranges = [1, 2, 3, 4, 6]; // meters

    ranges.forEach(r => {
        // Draw range arc within FOV
        const points = 360;
        const arcPoints = [];

        for (let i = 0; i <= points; i++) {
            const theta = -halfFOV + (2 * halfFOV * i) / points;
            const x = r * Math.sin(theta);
            const y = r * Math.cos(theta);

            // Check if point is within bounds
            if (y >= 0 && y <= MAX_RANGE_Y && Math.abs(x) <= MAX_RANGE_X) {
                arcPoints.push([detectionXScale(x), detectionYScale(y)]);
            }
        }

        if (arcPoints.length > 0) {
            detectionSvg.append('path')
                .attr('d', d3.line()(arcPoints))
                .attr('stroke', '#6b7280')
                .attr('stroke-dasharray', '2,2')
                .attr('stroke-width', 1)
                .attr('fill', 'none')
                .attr('opacity', 0.5);

            // Add range label
            if (r <= MAX_RANGE_Y) {
                detectionSvg.append('text')
                    .attr('x', detectionXScale(0.2))
                    .attr('y', detectionYScale(r * 0.98))
                    .attr('font-size', '10px')
                    .attr('fill', '#0294D0')
                    .text(`${r}m`);
            }
        }
    });
}

function updateDetectionPlot(detections) {
    if (!detectionSvg || !detections) return;

    // Update detection markers
    const circles = detectionSvg.selectAll('.detection-marker')
        .data(detections, (d, i) => i);

    circles.enter()
        .append('circle')
        .attr('class', 'detection-marker')
        .merge(circles)
        .attr('cx', d => detectionXScale(d.y))  // Y = left/right
        .attr('cy', d => detectionYScale(d.x))  // X = forward
        .attr('r', 6)
        .attr('fill', '#0294D0')
        .attr('stroke', 'white')
        .attr('stroke-width', 1)
        .style('cursor', 'pointer')
        .append('title')
        .text(d => `${d.label} (${(d.confidence * 100).toFixed(0)}%)\nDistance: ${d.distance.toFixed(2)}m`);

    circles.exit().remove();
}

function fetchDetectionData() {
    fetch('/detection_data')
        .then(response => response.json())
        .then(data => {
            updateDetectionPlot(data.detections);
        })
        .catch(error => {
            console.error('Error fetching detection data:', error);
        });
}

// ==================== Camera Feed ====================

function initializeCameraFeed() {
    const cameraFeed = document.getElementById('camera-feed');
    const cameraOverlay = document.getElementById('camera-overlay');

    cameraFeed.addEventListener('load', () => {
        // Hide "Waiting for camera..." message once feed loads
        if (cameraOverlay) {
            cameraOverlay.style.display = 'none';
        }
    });

    cameraFeed.addEventListener('error', () => {
        // Show error message if feed fails
        if (cameraOverlay) {
            cameraOverlay.style.display = 'block';
            cameraOverlay.innerHTML = '<div class="camera-info">Camera feed unavailable</div>';
        }
    });
}

// ==================== Initialization ====================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize camera feed
    initializeCameraFeed();

    // Initialize waypoint plot
    initializePlot();

    // Initialize detection scatter plot (only if container exists)
    const detectionContainer = document.getElementById('detection-plot-container');
    if (detectionContainer) {
        initializeDetectionPlot();
        // Fetch detection data every second
        setInterval(fetchDetectionData, 1000);
        fetchDetectionData();
    }

    // Fetch plot data every second
    setInterval(fetchPlotData, 1000);
    fetchPlotData();

    // Initial log
    addLog('Dashboard loaded', 'info');
});
