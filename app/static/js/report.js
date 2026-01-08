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

    function showError(message) {
        const container = document.getElementById('report-error-container');
        const msgSpan = document.getElementById('report-error-msg');
        if (container && msgSpan) {
            msgSpan.innerText = message;
            container.classList.remove('d-none');
        }
    }

    function hideError() {
        const container = document.getElementById('report-error-container');
        if (container) container.classList.add('d-none');
    }

    function updateReports() {
        hideError();

        // 1. Fetch Summary for Chart
        fetch('/api/attention_summary')
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                return response.json();
            })
            .then(summaryData => {
                if (!summaryData || summaryData.length === 0) {
                    console.log('No summary data available');
                    chart.data.labels = [];
                    chart.data.datasets[0].data = [];
                } else {
                    const labels = summaryData.map(s => s.time);
                    const chartData = summaryData.map(s => s.score);
                    chart.data.labels = labels;
                    chart.data.datasets[0].data = chartData;
                }
                chart.update();
            })
            .catch(error => {
                console.error('Error fetching summary:', error);
                showError('Failed to load chart data. Please try again later.');
            });

        // 2. Fetch History for Table
        fetch('/api/attention_history')
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                return response.json();
            })
            .then(data => {
                if (!data || data.length === 0) {
                    eventLogTable.innerHTML = '<tr><td colspan="5" class="text-center">No data available yet. Start the monitor to gather data.</td></tr>';
                    return;
                }

                // Populate Table using DocumentFragment for performance
                const fragment = document.createDocumentFragment();
                data.forEach(event => {
                    const row = document.createElement('tr');
                    const time = new Date(event.timestamp).toLocaleTimeString();

                    let statusBadge = 'bg-secondary';
                    const type = event.event_type;
                    if (['Focused', 'Attentive', 'Listening'].includes(type)) statusBadge = 'bg-success';
                    else if (type === 'Distracted') statusBadge = 'bg-warning text-dark';
                    else if (type === 'Sleeping') statusBadge = 'bg-danger';
                    else if (type === 'Bored') statusBadge = 'bg-info text-dark';

                    row.innerHTML = `
                        <td>${time}</td>
                        <td><strong>${event.roll_no}</strong></td>
                        <td>${event.name}</td>
                        <td>${type}</td>
                        <td><span class="badge ${statusBadge}">${type}</span></td>
                    `;
                    fragment.appendChild(row);
                });

                eventLogTable.innerHTML = '';
                eventLogTable.appendChild(fragment);
            })
            .catch(error => {
                console.error('Error fetching reports:', error);
                eventLogTable.innerHTML = '<tr><td colspan="5" class="text-center text-danger">Error loading history table.</td></tr>';
                showError('Failed to load event log. Please check your connection.');
            });
    }

    refreshBtn.addEventListener('click', updateReports);

    // Initial load
    updateReports();

    // Auto-refresh every 30 seconds
    setInterval(updateReports, 30000);
});
