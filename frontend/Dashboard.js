document.addEventListener("DOMContentLoaded", function () {
  const ctx = document.getElementById("barChart").getContext("2d");
  const barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: [],
      datasets: [
        {
          label: "Values",
          data: [],
          backgroundColor: "grey",
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
      barPercentage: 0.99,
      categoryPercentage: 1.0,
    },
  });

  // reads data from the givel localhost location, splits the given value into a list and calls
  // the updateChart function with those values
  async function fetchData() {
    try {
      const response = await fetch("http://localhost:8000/data/data.json");
      if (!response.ok) {
        throw new Error("Failed to fetch data");
      }
      const data = await response.json();

      const valuesList = Object.keys(data).map(
        (key) => data[key].neg_log_likelihoods
      );

      for (let i = 0; i < valuesList.length; i++) {
        console.log("i= " + i);
        await updateChart(valuesList[i]);
      }

      console.log("All charts updated successfully");
    } catch (error) {
      console.error("Error while fetching data:", error);
      alert(
        "Error while fetching data. Please check your (local) server and try again."
      );
    }
  }

  // takes a list of values and iterates over it (every 250ms per element)
  // it checks the value of every element; if it is bigger than a given value (5)
  // it draws the value into a chart of the colour red, else grey
  async function updateChart(values) {
    return new Promise((resolve, reject) => {
      resetChart();
      starting();
      let index = 0;
      const backgroundColors = [];

      const intervalId = setInterval(() => {
        if (index < values.length) {
          barChart.data.labels.push(index + 1);

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

          barChart.data.datasets[0].data.push(values[index]);
          barChart.data.datasets[0].backgroundColor = backgroundColors;
          barChart.update();
          index++;
        } else {
          clearInterval(intervalId);
          resolve();
        }
      }, 50);
    });
  }

  // sets the backroundColor value of the start-button element to grey for 200ms
  function highlightButton() {
    const button = document.querySelector(".start-button");

    button.style.backgroundColor = "grey";

    setTimeout(() => {
      button.style.backgroundColor = "#2196f3";
    }, 200);
  }

  // adds "Grinding in Progress" and "Anomaly-Detection is currently running" into the info-box element
  function starting() {
    const infoBox = document.querySelector(".info-box");
    const messages = [
      "Grinding in Progress",
      "Anomaly-Detection is currently running",
    ];

    messages.forEach((message) => {
      const messageElement = document.createElement("div");
      messageElement.textContent = message;
      infoBox.appendChild(messageElement);
    });

    infoBox.scrollTop = infoBox.scrollHeight;
  }

  function resetChart() {
    barChart.data.labels = [];
    barChart.data.datasets[0].data = [];
    barChart.data.datasets[0].backgroundColor = [];
    barChart.update();
  }

  document
    .querySelector(".start-button")
    .addEventListener("click", function () {
      highlightButton();
      fetchData();
    });
});
