document.getElementById("scanEmail").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            function: scanEmail  // ✅ Directly execute scanEmail()
        });
    });
});
