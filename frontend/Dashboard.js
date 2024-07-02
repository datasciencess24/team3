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
          borderColor: "rgba(54, 162, 235, 1)",
          borderWidth: 1,
        },
        {
          label: "Anomalies",
          backgroundColor: "red",
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
      categoryPercentage: 2.0,
    },
  });

  // reads data from the givel localhost location, splits the given value into a list and calls
  // the updateChart function with those values
  function fetchData(resolve) {
    fetch("http://localhost:8000/data/data.txt")
      .then((response) => response.text())
      .then((data) => {
        const values = data
          .split(",")
          .map((value) => parseInt(value.trim(), 10));
        updateChart(values);
        resolve();
      })
      .catch((error) => {
        console.error("Error while fethcing data:", error);
        alert(
          "Error while fetching data. Please check your (local) server and try again."
        );
        resolve();
      });
  }

  // takes a list of values and iterates over it (every 250ms per element)
  // it checks the value of every element; if it is bigger than a given value (5)
  // it draws the value into a chart of the colour red, else grey
  function updateChart(values) {
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
      }
    }, 250);
  }

  // sets the backroundColor value of the start-button element to grey for 200ms
  function highlightButton(resolve) {
    const button = document.querySelector(".start-button");

    button.style.backgroundColor = "grey";

    setTimeout(() => {
      button.style.backgroundColor = "#2196f3";
      resolve();
    }, 200);
  }

  // adds "Drilling in Progress" and "Anomaly-Detection is currently running" into the info-box element
  function starting(resolve) {
    const infoBox = document.querySelector(".info-box");
    const messages = [
      "Drilling in Progress",
      "Anomaly-Detection is currently running",
    ];

    messages.forEach((message) => {
      const messageElement = document.createElement("div");
      messageElement.textContent = message;
      infoBox.appendChild(messageElement);
    });

    infoBox.scrollTop = infoBox.scrollHeight;
  }

  document
    .querySelector(".start-button")
    .addEventListener("click", function () {
      // starts highlight button and starting/fetchData in parallel
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
