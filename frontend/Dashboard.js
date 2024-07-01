document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("barChart").getContext("2d");
  const barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: [], // Labels for the x-axis (time sequence)
      datasets: [
        {
          label: "Values",
          data: [], // Initial empty data
          backgroundColor: "rgba(54, 162, 235, 0.2)",
          borderColor: "rgba(54, 162, 235, 1)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
        },
      },
      layout: {
        padding: {
          left: 10,
          right: 10,
          top: 0,
          bottom: 0,
        },
      },
      indexAxis: "x",
      barPercentage: 0.95,
      categoryPercentage: 1.0,
    },
  });

  // Funktion zum Abrufen der Daten mit Promise-Resolve
  function fetchData(resolve) {
    fetch("http://localhost:8000/data/data.txt")
      .then((response) => response.text())
      .then((data) => {
        const values = data
          .split(",")
          .map((value) => parseInt(value.trim(), 10));
        updateChart(values); // Funktion zum Aktualisieren des Charts aufrufen
        resolve(); // Promise auflösen, um anzuzeigen, dass die Funktion abgeschlossen ist
      })
      .catch((error) => {
        console.error("Fehler beim Abrufen der Daten:", error);
        alert("Fehler beim Abrufen der Daten. Bitte versuchen Sie es erneut."); // Fehlerbehandlung
        resolve(); // Promise auflösen, um anzuzeigen, dass die Funktion abgeschlossen ist (auch im Fehlerfall)
      });
  }

  function updateChart(values) {
    let index = 0;
    const backgroundColors = []; // Array to store background colors

    const intervalId = setInterval(() => {
      if (index < values.length) {
        barChart.data.labels.push(index + 1); // Add time sequence label

        // Check condition and set background color
        if (values[index] > 5) {
          backgroundColors.push("red");
          const anomalyText = document.createElement("div");
          anomalyText.classList.add("warning");
          anomalyText.textContent = `Index ${index} shows a potential anomaly`;

          const infoBox = document.querySelector(".info-box");
          infoBox.appendChild(anomalyText);
          infoBox.scrollTop = infoBox.scrollHeight;
        } else {
          backgroundColors.push("grey");
        }

        barChart.data.datasets[0].data.push(values[index]); // Add the value
        barChart.data.datasets[0].backgroundColor = backgroundColors; // Update background colors array
        barChart.update();
        index++;
      } else {
        clearInterval(intervalId); // Stop the interval when data is exhausted
      }
    }, 100);
  }

  // Funktion zum Hervorheben des Buttons mit Promise-Resolve
  function highlightButton(resolve) {
    const button = document.querySelector(".start-button");

    // Ändere die Hintergrundfarbe auf grau
    button.style.backgroundColor = "#dcdcdc";

    // Setze die Hintergrundfarbe nach einer kurzen Verzögerung zurück
    setTimeout(() => {
      button.style.backgroundColor = "#2196f3"; // Originalfarbe wiederherstellen
      resolve(); // Promise auflösen, um anzuzeigen, dass die Funktion abgeschlossen ist
    }, 200); // Verzögerung in Millisekunden
  }

  function starting(resolve) {
    const infoBox = document.querySelector(".info-box");

    // Nachrichten hinzufügen
    const messages = [
      "Drilling in Progress",
      "Anomaly-Detection is currently running",
    ];

    // Nachrichten zur Info-Box hinzufügen
    messages.forEach((message) => {
      const messageElement = document.createElement("div");
      messageElement.textContent = message;
      infoBox.appendChild(messageElement);
    });

    // Automatisch nach unten scrollen, wenn die Info-Box überläuft
    infoBox.scrollTop = infoBox.scrollHeight;
  }

  document
    .querySelector(".start-button")
    .addEventListener("click", function () {
      const messages = [
        "Drilling in Progress",
        "Anomaly-Detection is currently running",
      ];

      // running following functions simultaneously
      Promise.all([
        new Promise((resolve, reject) => {
          highlightButton(resolve);
        }),
        new Promise((resolve, reject) => {
          starting(resolve);
          fetchData(resolve);
        }),
      ])
        .then(() => {
          console.log("All good!");
        })
        .catch((error) => {
          console.error(
            "There was a problem while visualizing the data:",
            error
          );
        });
    });
});
