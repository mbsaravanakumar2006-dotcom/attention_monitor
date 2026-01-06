document.addEventListener("DOMContentLoaded", function () {
    const ctx = document.getElementById('attentionChart').getContext('2d');
    const eventLogTable = document.getElementById('report-event-log');
    const refreshBtn = document.getElementById('refreshGraph');

    let chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Class Attention Level (%)',
                data: [],
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Attention %'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                }
            }
        }
    });

    function updateReports() {
        fetch('/api/attention_history')
            .then(response => response.json())
            .then(data => {
                if (data.length === 0) {
                    eventLogTable.innerHTML = '<tr><td colspan="5" class="text-center">No data available yet.</td></tr>';
                    return;
                }

                // Populate Table
                eventLogTable.innerHTML = '';
                data.forEach(event => {
                    const row = document.createElement('tr');
                    const time = new Date(event.timestamp).toLocaleTimeString();

                    let statusBadge = 'bg-secondary';
                    if (event.event_type === 'Focused' || event.event_type === 'Attentive' || event.event_type === 'Listening') statusBadge = 'bg-success';
                    else if (event.event_type === 'Distracted') statusBadge = 'bg-warning text-dark';
                    else if (event.event_type === 'Sleeping') statusBadge = 'bg-danger';
                    else if (event.event_type === 'Bored') statusBadge = 'bg-info text-dark';

                    row.innerHTML = `
                        <td>${time}</td>
                        <td><strong>${event.roll_no}</strong></td>
                        <td>${event.name}</td>
                        <td>${event.event_type}</td>
                        <td><span class="badge ${statusBadge}">${event.event_type}</span></td>
                    `;
                    eventLogTable.appendChild(row);
                });

                // Process Data for Graph
                // Group by timestamp (minute) and calculate average
                const timeMap = {}; // {HH:MM: [score1, score2, ...]}

                data.forEach(event => {
                    const date = new Date(event.timestamp);
                    const timeStr = date.getHours().toString().padStart(2, '0') + ':' +
                        date.getMinutes().toString().padStart(2, '0');

                    let score = 50; // Neutral
                    if (event.event_type === 'Focused' || event.event_type === 'Attentive' || event.event_type === 'Listening') score = 100;
                    if (event.event_type === 'Distracted') score = 30;
                    if (event.event_type === 'Sleeping') score = 0;
                    if (event.event_type === 'Bored') score = 40;

                    if (!timeMap[timeStr]) timeMap[timeStr] = [];
                    timeMap[timeStr].push(score);
                });

                const labels = Object.keys(timeMap).sort();
                const chartData = labels.map(label => {
                    const scores = timeMap[label];
                    return scores.reduce((a, b) => a + b, 0) / scores.length;
                });

                chart.data.labels = labels;
                chart.data.datasets[0].data = chartData;
                chart.update();
            })
            .catch(error => {
                console.error('Error fetching reports:', error);
                eventLogTable.innerHTML = '<tr><td colspan="5" class="text-center text-danger">Error loading data.</td></tr>';
            });
    }

    refreshBtn.addEventListener('click', updateReports);

    // Initial load
    updateReports();

    // Auto-refresh every 30 seconds
    setInterval(updateReports, 30000);
});
