document.addEventListener("DOMContentLoaded", function () {
    var socket = io();


    socket.on('connect', function () {
        console.log("Connected to WebSocket");
    });

    // Handle incoming frame data stats
    socket.on('frame_data', function (data) {
        // data.students = [{name: 'John', status: 'Focused'}, ...]
        // data.avg_attention = 85

        // Update Score
        document.getElementById('avg-score').innerText = Math.round(data.avg_attention) + '%';

        // Update Student List
        const list = document.getElementById('student-list');
        list.innerHTML = '';
        data.students.forEach(student => {
            let colorClass = 'status-active';
            if (student.status === 'Distracted') colorClass = 'status-distracted';
            if (student.status === 'Sleeping') colorClass = 'status-asleep';

            const item = document.createElement('div');
            item.className = 'list-group-item d-flex justify-content-between align-items-center';
            item.innerHTML = `
                <span>
                    <span class="status-indicator ${colorClass}"></span>
                    <strong>${student.roll_no}</strong>: ${student.name}
                    <small class="text-muted ms-2">(Acc: ${student.accuracy}%)</small>
                </span>
                <span class="badge bg-secondary">${student.status}</span>
            `;

            list.appendChild(item);
        });
    });

    // Handle specific alerts
    socket.on('alert_event', function (event) {
        const log = document.getElementById('event-log');
        const item = document.createElement('li');
        item.className = 'list-group-item small';
        item.innerText = `[${new Date().toLocaleTimeString()}] ${event.message}`;
        log.prepend(item);
    });
});
