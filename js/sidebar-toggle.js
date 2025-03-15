document.addEventListener("DOMContentLoaded", function () {
    const quickBtn = document.getElementById("quick-scan-btn");
    const advancedBtn = document.getElementById("advanced-scan-btn");
    const scanDetails = document.querySelector(".scan-details");
    const advancedScan = document.querySelector(".advanced-scan");

    // Function to show quick scan details and hide advanced scan
    function showQuickScan() {
        quickBtn.classList.add("active");
        advancedBtn.classList.remove("active");
        scanDetails.classList.add("show");
        advancedScan.classList.remove("show");
    }

    // Function to show advanced scan details and hide quick scan
    function showAdvancedScan() {
        advancedBtn.classList.add("active");
        quickBtn.classList.remove("active");
        advancedScan.classList.add("show");
        scanDetails.classList.remove("show");
    }

    // Attach event listeners to buttons
    quickBtn.addEventListener("click", showQuickScan);
    advancedBtn.addEventListener("click", showAdvancedScan);

    // Set default state: quick scan visible, advanced hidden.
    showQuickScan();
});