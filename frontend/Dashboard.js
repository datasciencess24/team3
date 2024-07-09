let abortFlag = false;
let current = 0;

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
        {
          label: "Potential anomalies",
          data: [],
          backgroundColor: "red",
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
      categoryPercentage: 2.0,
    },
  });

  // reads data from the givel localhost location, splits the given value into a list and calls
  // the updateChart function with those values
  async function fetchData() {
    try {
      //random parameter to bypass cache
      const response = await fetch(`http://localhost:8000/data/data.json?nocache=${new Date().getTime()}`);
      if (!response.ok) {
        throw new Error("Failed to fetch data");
      }
      const data = await response.json();

      const statusList = Object.keys(data).map((key) => data[key].status);

      const proportionAnomalousList = Object.keys(data).map(
        (key) => data[key].proportion_anomalous_in_percent
      );

      const percentileList = Object.keys(data).map(
        (key) => data[key].percentile
      );

      const valuesList = Object.keys(data).map(
        (key) => data[key].neg_log_likelihoods
      );

      for (let i = 0; i < valuesList.length; i++) {
        current = i + 1;
        if (abortFlag) {
          return;
        }
        console.log("i = " + i);
        resetChart();
        writeIntoInfoBox(
          "Grinding-process " +
            current +
            " in progress. Monitoring starting time: " +
            new Date()
        );
        await updateChart(valuesList[i]);
        if (abortFlag) {
          return;
        }

        writeIntoInfoBox("Monitoring end time: " + new Date());
        writeIntoInfoBox("Full process classification: " + statusList[i]);
        writeIntoInfoBox(
          "Proportion of segments that can be classified as (approximately) anomalous: " +
            proportionAnomalousList[i].toFixed(2) +
            "%"
        );
        writeIntoInfoBox(
          "Proportion of normal data used for training that the current process exceeds with the number of its abnormal segments: " +
            percentileList[i].toFixed(2) +
            "%"
        );

        if (i < valuesList.length - 1) {
          if (statusList[i] == "anomalous") {
            console.log("anomalous");
            await showWarningDialog();
          } else {
            await showContinueDialog(i);
          }
        }
        if (abortFlag) {
          return;
        }
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
      let index = 0;
      const backgroundColors = [];

      const intervalId = setInterval(() => {
        if (abortFlag) {
          clearInterval(intervalId);
          resolve();
          return;
        }
        if (index < values.length) {
          const xValue = (index + 1) * 0.05;
          barChart.data.labels.push(xValue.toFixed(2));

          if (values[index] > 5) {
            backgroundColors.push("red");
            const anomalyText = document.createElement("div");
            anomalyText.classList.add("warning");
            anomalyText.textContent = `Timestamp ${((index + 1) * 0.05).toFixed(
              2
            )}s shows a potential anomaly!`;

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

  // adds given text into the info-box element
  function writeIntoInfoBox(text) {
    const infoBox = document.querySelector(".info-box");
    const messageElement = document.createElement("div");
    messageElement.textContent = text;
    infoBox.appendChild(messageElement);
    infoBox.scrollTop = infoBox.scrollHeight;
  }

  function resetChart() {
    barChart.data.labels = [];
    barChart.data.datasets[0].data = [];
    barChart.data.datasets[0].backgroundColor = [];
    barChart.update();
  }

  function showContinueDialog(i) {
    return new Promise((resolve) => {
      const dialog = document.createElement("div");
      dialog.classList.add("modal");

      const dialogContent = `
        <div class="modal-content">
          <p>Process ${i + 1} has finished without major complications.<br /> Do you want to continue?</p>
          <button id="continue-button">Continue</button>
          <button id="export-diagram-button">Export Diagram</button>
        </div>
      `;
      dialog.innerHTML = dialogContent;
      document.body.appendChild(dialog);

      document
        .getElementById("continue-button")
        .addEventListener("click", () => {
          document.body.removeChild(dialog);
          resolve();
        });

      document
        .getElementById("export-diagram-button")
        .addEventListener("click", () => {
          exportChartAsPNG();
        });
    });
  }

  function showWarningDialog() {
    return new Promise((resolve) => {
      const dialog = document.createElement("div");
      dialog.classList.add("modal");

      const dialogContent = `
        <div class="modal-content">
          <p><span style="font-weight: bold; color: red;">Warning this process seems to be anomalous!</span><br/>
          It is recommended to keep the diagram and the log and to stop further runs for the time being!</p>
          <button id="continue-button">Continue</button>
          <button id="export-diagram-button">Export</button>
          <button id="abort-button">Abort</button>
        </div>
      `;
      dialog.innerHTML = dialogContent;
      document.body.appendChild(dialog);

      document
        .getElementById("continue-button")
        .addEventListener("click", () => {
          document.body.removeChild(dialog);
          resolve();
        });

      document
        .getElementById("export-diagram-button")
        .addEventListener("click", () => {
          exportChartAsPNG();
          exportLogAsTXT();
        });

      document.getElementById("abort-button").addEventListener("click", () => {
        document.body.removeChild(dialog);
        abortFlag = true;
        writeIntoInfoBox("Aborted at: " + new Date());
        resolve();
      });
    });
  }
/*
  // black backround
  function exportChartAsPNG() {
    const link = document.createElement("a");
    link.href = ctx.canvas.toDataURL("image/png");
    link.download = "chart of process " + current + ", " + new Date() + ".png";
    link.click();
  }
    */

  function exportChartAsPNG() {
    const canvas = document.getElementById('barChart');
    const ctx = canvas.getContext('2d');
  
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
  
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
  
    tempCtx.drawImage(canvas, 0, 0);
  
    const link = document.createElement('a');
    link.href = tempCanvas.toDataURL('image/png');
    link.download = "chart of process " + current + ", " + new Date() + ".png";
    link.click();
  }

  function exportLogAsTXT() {
    const infoBox = document.querySelector(".info-box");
    const logContent = Array.from(infoBox.childNodes)
      .map((node) => node.textContent)
      .join("\n");

    const blob = new Blob([logContent], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "log " + new Date() + ".txt";
    link.click();
  }

  //Event listener for start-button
  document
    .querySelector(".start-button")
    .addEventListener("click", function () {
      abortFlag = false;
      highlightButton();
      fetchData();
    });

  //Event listener for export-log-button
  document
    .querySelector(".export-log-button")
    .addEventListener("click", function () {
      exportLogAsTXT();
    });

  document
    .querySelector(".abort-button")
    .addEventListener("click", function () {
      abortFlag = true;
      writeIntoInfoBox("Aborted at: " + new Date());
    });

  document
    .querySelector(".export-diagram-button")
    .addEventListener("click", function () {
      exportChartAsPNG();
    });
});
