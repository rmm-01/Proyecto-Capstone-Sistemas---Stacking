document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('excel-file');
    const loadingMessage = document.getElementById('loading-message');
    const errorMessage = document.getElementById('error-message');
    const dataSummary = document.getElementById('data-summary');
    const tableBody = document.querySelector('#data-table tbody');
    const predictionChartContainer = document.getElementById('prediction-chart');

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        const file = fileInput.files[0];

        if (!file) {
            errorMessage.style.display = 'block';
            errorMessage.textContent = 'Por favor selecciona un archivo.';
            return;
        }

        errorMessage.style.display = 'none';
        loadingMessage.style.display = 'block';
        dataSummary.style.display = 'none';
        predictionChartContainer.innerHTML = '';

        const formData = new FormData();
        formData.append('excel-file', file);

        fetch('/prediccion1', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingMessage.style.display = 'none';

            if (data.error) {
                errorMessage.textContent = data.error;
                errorMessage.style.display = 'block';
                return;
            }

            const predictions = data.predictions;
            tableBody.innerHTML = '';

            predictions.forEach((pred) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${pred.fecha}</td>
                    <td>${pred.codigo}</td>
                    <td>${pred.demanda_predicha.toFixed(2)}</td>
                    <td>${pred.precio.toFixed(2)}</td>
                `;
                tableBody.appendChild(row);
            });

            dataSummary.style.display = 'block';

            // Crear gráfico
            const canvas = document.createElement('canvas');
            predictionChartContainer.appendChild(canvas);
            const ctx = canvas.getContext('2d');

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: predictions.map((pred, i) => pred.codigo),
                    datasets: [{
                        label: 'Demanda Predicha',
                        data: predictions.map(pred => pred.demanda_predicha),
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        })
        .catch(error => {
            loadingMessage.style.display = 'none';
            errorMessage.textContent = 'Error al procesar el archivo. Intenta nuevamente.';
            errorMessage.style.display = 'block';
            console.error(error);
        });
    });
});
