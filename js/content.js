let lastEmail = "";  // Prevents duplicate scans

// Function to check if an email is currently open
function isEmailOpen() {
    // Check for the presence of the email body element (specific to Gmail)
    return document.querySelector(".a3s") !== null;
}

// Function to update the sidebar with scan results
function updateSidebar(data) {
    console.log("Updating sidebar with data:", data);

    const timestampElement = document.getElementById("scan-timestamp");
    const durationElement = document.getElementById("scan-duration");
    const statusElement = document.getElementById("scan-status");
    const classificationElement = document.getElementById("scan-classification");
    const scoreElement = document.getElementById("scan-score");

    if (timestampElement) timestampElement.textContent = data.timestamp || "--";
    if (durationElement) durationElement.textContent = data.duration ? `${data.duration}s` : "--";
    if (statusElement) statusElement.textContent = data.status || "--";
    if (classificationElement) classificationElement.textContent = data.classification || "--";
    if (scoreElement) scoreElement.textContent = data.score ? `${data.score}%` : "--";
}

const serverUrl ="https://phishshield-ai-app-51ce73e83779.herokuapp.com"
// Function to listen for SSE updates
function listenForScanStatus() {
    if (window.eventSource) {
        window.eventSource.close();  // Close previous SSE connection
    }

    window.eventSource = new EventSource(`${serverUrl}/scan_status`);

    window.eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log("Received SSE update:", data);

        updateSidebar({
            timestamp: data.timestamp,
            duration: data.duration,
            status: data.status,
            classification: data.classification,
            score: data.score
        });

        // If the scan is complete, stop listening for updates
        if (data.status === "Complete") {
            window.eventSource.close();
            window.eventSource = null;
        }
    };

    window.eventSource.onerror = (error) => {
        console.error("SSE error:", error);
        window.eventSource.close();
        window.eventSource = null;
    };
}

// Function to run Quick Scan when an email is opened
function quickScanEmail() {
    if (!isEmailOpen()) return;  // Exit if no email is open

    let emailBody = document.querySelector(".a3s");  // Gmail email content
    let emailTitle = document.querySelector(".hP"); // Email subject
    let sender = document.querySelector(".gD");     // Email sender

    if (!emailBody || !emailTitle || !sender) return;

    let extractedText = emailBody.innerText.trim();
    let subject = emailTitle.innerText.trim();
    let senderEmail = sender.getAttribute("email");
    console.log(senderEmail)

    if (extractedText === lastEmail) return;  // Prevent re-scanning the same email
    lastEmail = extractedText;

    // Extract URLs from the email
    let links = Array.from(emailBody.querySelectorAll("a")).map(a => a.href);
    console.log("Detected Links:", links);

    // Send data for a quick scan
    fetch(`${serverUrl}/quick_scan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subject, sender: senderEmail, text: extractedText, links: links })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        return response.json();
    })
    .then(data => {
        console.log("Scan started:", data);
        // Start listening for SSE updates
        listenForScanStatus();
    })
    .catch(error => console.error("Scan error:", error));
}

// Inject the sidebar into the page
function injectSidebar() {
    return new Promise((resolve, reject) => {
        fetch(chrome.runtime.getURL("html/sidebar.html"))
            .then(response => response.text())
            .then(html => {
                let sidebarContainer = document.createElement("div");
                sidebarContainer.innerHTML = html;
                document.body.appendChild(sidebarContainer);

                // Inject the sidebar CSS
                let sidebarCSS = document.createElement("link");
                sidebarCSS.rel = "stylesheet";
                sidebarCSS.href = chrome.runtime.getURL("css/sidebar.css");
                document.head.appendChild(sidebarCSS);

                resolve(); // Resolve the promise when the sidebar is fully loaded
            })
            .catch(error => {
                console.error("Failed to load sidebar:", error);
                reject(error);
            });
    });
}

// Inject the toggle button into the page
function injectToggleButton() {
    fetch(chrome.runtime.getURL("html/toggle-button.html"))
        .then(response => response.text())
        .then(html => {
            let buttonContainer = document.createElement("div");
            buttonContainer.innerHTML = html;
            document.body.appendChild(buttonContainer);

            // Inject the toggle button CSS
            let toggleButtonCSS = document.createElement("link");
            toggleButtonCSS.rel = "stylesheet";
            toggleButtonCSS.href = chrome.runtime.getURL("css/toggle-button.css");
            document.head.appendChild(toggleButtonCSS);

            // Wait for sidebar to load before adding event listener
            setTimeout(() => {
                let toggleButton = document.querySelector(".toggle-button");
                let sidebar = document.querySelector(".sidebar-container");

                if (toggleButton && sidebar) {
                    toggleButton.addEventListener("click", function () {
                        sidebar.style.display = (sidebar.style.display === "none" || sidebar.style.display === "")
                            ? "block"
                            : "none";
                    });
                } else {
                    console.error("Sidebar or toggle button not found.");
                }
            }, 500); // Wait 500ms to ensure sidebar loads
        })
        .catch(error => console.error("Failed to load toggle button:", error));
}

// Observe Gmail for email opens
function observeEmails() {
    let observer = new MutationObserver(() => {
        if (isEmailOpen()) {
            let emailBody = document.querySelector(".a3s");
            if (emailBody && emailBody.innerText.trim() !== "") {
                quickScanEmail();
            }
        } else {
            // Reset when email is closed
            resetScanStatus();
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });
}

function resetScanStatus() {
    console.log("Resetting scan status...");

    // Reset sidebar values
    updateSidebar({
        timestamp: "--",
        duration: "--",
        status: "Idle",
        classification: "--",
        score: "--"
    });

    // Stop listening for SSE updates
    if (window.eventSource) {
        window.eventSource.close();
        window.eventSource = null;
    }
}



// Initialize the sidebar and toggle button
injectSidebar();
injectToggleButton();

// Start observing Gmail for email opens
observeEmails();