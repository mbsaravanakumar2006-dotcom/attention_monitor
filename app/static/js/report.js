document.addEventListener("DOMContentLoaded", function () {
    const ctx = document.getElementById('attentionChart').getContext('2d');
    const eventLogTable = document.getElementById('report-event-log');
    const refreshBtn = document.getElementById('refreshGraph');
    const searchInput = document.getElementById('reportSearch');
    const clearBtn = document.getElementById('clearData');

    let allEvents = [];
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
                    title: { display: true, text: 'Attention %' }
                },
                x: {
                    title: { display: true, text: 'Time' }
                }
            }
        }
    });

    function formatDuration(seconds) {
        if (!seconds || seconds === 0) return "Ongoing";
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.round(seconds % 60);
        let res = "";
        if (h > 0) res += `${h}h `;
        if (m > 0) res += `${m}m `;
        if (s > 0 || res === "") res += `${s}s`;
        return res.trim();
    }

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

    function renderTable() {
        if (!allEvents || allEvents.length === 0) {
            eventLogTable.innerHTML = '<tr><td colspan="5" class="text-center py-4">No data available yet.</td></tr>';
            return;
        }

        const searchTerm = searchInput ? searchInput.value.toLowerCase() : "";
        const filtered = allEvents.filter(e =>
            e.name.toLowerCase().includes(searchTerm) ||
            e.roll_no.toLowerCase().includes(searchTerm) ||
            e.event_type.toLowerCase().includes(searchTerm)
        );

        if (filtered.length === 0) {
            eventLogTable.innerHTML = '<tr><td colspan="5" class="text-center py-4">No matching records found.</td></tr>';
            return;
        }

        const fragment = document.createDocumentFragment();
        filtered.forEach(event => {
            const row = document.createElement('tr');
            const startTime = new Date(event.timestamp);
            const endTime = event.duration > 0 ? new Date(startTime.getTime() + event.duration * 1000) : null;

            const timeInterval = endTime
                ? `${startTime.toLocaleTimeString()} - ${endTime.toLocaleTimeString()}`
                : `${startTime.toLocaleTimeString()} - Ongoing`;

            let statusBadge = 'bg-secondary';
            const type = event.event_type;
            if (['Focused', 'Attentive', 'Listening'].includes(type)) statusBadge = 'bg-success';
            else if (type === 'Distracted') statusBadge = 'bg-warning text-dark';
            else if (type === 'Sleeping') statusBadge = 'bg-danger';
            else if (type === 'Bored') statusBadge = 'bg-info text-dark';
            else if (type === 'Out of Sight') statusBadge = 'bg-dark';

            row.innerHTML = `
                <td><small class="text-muted">${timeInterval}</small></td>
                <td><strong>${event.roll_no}</strong></td>
                <td>${event.name}</td>
                <td><span class="badge ${statusBadge}">${type}</span></td>
                <td><span class="text-primary fw-bold">${formatDuration(event.duration)}</span></td>
            `;
            fragment.appendChild(row);
        });

        eventLogTable.innerHTML = '';
        eventLogTable.appendChild(fragment);
    }

    function updateReports() {
        hideError();

        // 1. Fetch Summary for Chart
        const chartLoader = document.getElementById('chart-loader');
        if (chartLoader) chartLoader.classList.remove('d-none');

        fetch('/api/attention_summary')
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                return response.json();
            })
            .then(summaryData => {
                if (chartLoader) chartLoader.classList.add('d-none');
                if (!summaryData || summaryData.length === 0) {
                    chart.data.labels = [];
                    chart.data.datasets[0].data = [];
                } else {
                    chart.data.labels = summaryData.map(s => s.time);
                    chart.data.datasets[0].data = summaryData.map(s => s.score);
                }
                chart.update();
            })
            .catch(error => {
                if (chartLoader) chartLoader.classList.add('d-none');
                console.error('Error fetching summary:', error);
                showError('Failed to load chart data.');
            });

        // 2. Fetch History for Table
        fetch('/api/attention_history')
            .then(response => {
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                return response.json();
            })
            .then(data => {
                allEvents = data;
                renderTable();
            })
            .catch(error => {
                console.error('Error fetching reports:', error);
                eventLogTable.innerHTML = '<tr><td colspan="5" class="text-center text-danger py-4">Error loading history table.</td></tr>';
            });
    }

    // Event Listeners
    if (searchInput) {
        searchInput.addEventListener('input', renderTable);
    }

    if (refreshBtn) {
        refreshBtn.addEventListener('click', updateReports);
    }

    if (clearBtn) {
        clearBtn.addEventListener('click', function () {
            if (confirm("Are you sure you want to DELETE ALL DATA? This action cannot be undone.")) {
                fetch('/api/clear_data', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'success') {
                            alert(data.message);
                            updateReports();
                        } else {
                            alert("Error: " + data.message);
                        }
                    })
                    .catch(err => alert("Request Failed: " + err.message));
            }
        });
    }

    // Initial load and periodic refresh
    updateReports();
    setInterval(updateReports, 30000);
});
