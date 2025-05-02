const symptomSelect = document.getElementById("symptom-select");
const selectedSymptomsDiv = document.getElementById("selected-symptoms");
const selectedSymptomsListInput = document.getElementById(
  "selected-symptoms-list"
);
let selectedSymptoms = [];

// Automatically add symptom when selected from dropdown
symptomSelect.addEventListener("change", function () {
  const selectedOption = this.value;
  if (selectedOption && !selectedSymptoms.includes(selectedOption)) {
    selectedSymptoms.push(selectedOption);
    updateSelectedSymptomsDisplay();
    this.value = ""; // Reset the dropdown
  }
});

function removeSymptom(symptomToRemove) {
  selectedSymptoms = selectedSymptoms.filter(
    (symptom) => symptom !== symptomToRemove
  );
  updateSelectedSymptomsDisplay();
}

function updateSelectedSymptomsDisplay() {
  selectedSymptomsDiv.innerHTML = "";
  selectedSymptoms.forEach((symptom) => {
    const tag = document.createElement("span");
    tag.classList.add('symptom-tag');
    tag.textContent = symptom;
    const removeButton = document.createElement("span");
    removeButton.innerHTML = '&times;';
    removeButton.style.cursor = "pointer";
    removeButton.style.marginLeft = '5px';
    removeButton.onclick = () => removeSymptom(symptom);
    tag.appendChild(removeButton);
    selectedSymptomsDiv.appendChild(tag);
  });
  selectedSymptomsListInput.value = selectedSymptoms.join(",");
}

async function typeWriter(text, element, speed = 30) {
  return new Promise((resolve) => {
    let i = 0;
    const typingInterval = setInterval(() => {
      if (i < text.length) {
        element.innerHTML = text.substring(0, i + 1) + "<span>|</span>";
        i++;
      } else {
        element.innerHTML = text;
        clearInterval(typingInterval);
        resolve();
      }
    }, speed);
  });
}

async function predictDisease() {
  const resultsDiv = document.getElementById("prediction_results");
  resultsDiv.innerHTML = "<p >Analyzing symptoms... <span>|</span></p>";

  const formData = new FormData();
  selectedSymptoms.forEach((symptom) => {
    formData.append("symptoms", symptom);
  });

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.error) {
      resultsDiv.innerHTML = `<p>${data.error}</p>`;
      return;
    }

    resultsDiv.innerHTML = `
              <div>
                  <p><span>This app is for informational purposes only and does not constitute professional medical advice. Please consult a licensed healthcare provider.</span></p>
              </div>
              <div>
                  <h3>🦠 Predicted Disease: <strong>${data.predicted_disease}</strong></h3>
              </div>
              <div>
                  <p>💊 Recommended Treatment: <span>${data.recommended_drug}</span></p>
              </div>
              <div>
                  <h4>📖 Disease Information</h4>
                  <div id="disease-info-container"></div>
              </div>
          `;

    const infoContainer = document.getElementById(
      "disease-info-container"
    );
    const infoLines = data.disease_info.split("<br>");

    for (const line of infoLines) {
      if (line.trim()) {
        const lineElement = document.createElement("div");

        if (/^\d+\./.test(line)) {
          lineElement.innerHTML = line;
        } else if (line.startsWith("•")) {
          const ul = document.createElement("ul");
          const li = document.createElement("li");
          li.textContent = line.substring(1).trim();
          ul.appendChild(li);
          lineElement.appendChild(ul);
        } else {
          lineElement.textContent = line;
        }

        infoContainer.appendChild(lineElement);
        await typeWriter(line, lineElement, 20);
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
    }
  } catch (error) {
    resultsDiv.innerHTML = `<p>An error occurred: ${error.message}</p>`;
  }
}