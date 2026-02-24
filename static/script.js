async function predict() {
    const text = document.getElementById("textInput").value;
    const resultDiv = document.getElementById("result");
    const progressBar = document.getElementById("progressBar");
    const button = document.getElementById("analyzeBtn");

    if (text.trim() === "") {
        resultDiv.innerHTML = "<span style='color:#f87171'>Please enter text.</span>";
        return;
    }

    button.disabled = true;
    button.innerText = "Analyzing...";
    resultDiv.innerHTML = "Processing...";
    progressBar.style.width = "0%";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (data.error) {
            resultDiv.innerHTML = "<span style='color:#f87171'>" + data.error + "</span>";
        } else {
            resultDiv.innerHTML =
                "<strong>Prediction:</strong> " + data.prediction +
                "<br><strong>Probability:</strong> " + data.probability + "%";

            progressBar.style.width = data.probability + "%";

            if (data.prediction === "Depressed") {
                progressBar.style.background = "#ef4444";
            } else {
                progressBar.style.background = "#22c55e";
            }
        }
    } catch (error) {
        resultDiv.innerHTML = "<span style='color:#f87171'>Server error. Try again.</span>";
    }

    button.disabled = false;
    button.innerText = "Analyze";
}