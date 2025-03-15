chrome.runtime.onMessage.addListener((request, sender, sendResponse)=>{
    if (request.images){
        request.images.forEach((imgUrl, index)=>{
            fetch(imgUrl)
            .then(response => response.blob())
            .then(blob => {
                let url = URL.createObjectURL(blob);
                chrome.downloads.download({url: url, filename: `email_image_${index}.png`})})})
        }})