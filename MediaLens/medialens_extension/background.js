chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyzeHeadline",
    title: "Analyzovať titulok (MediaLens)",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId !== "analyzeHeadline") return;

  const selectedText = (info.selectionText || "").trim();
  const slovakLabels = {
    clickbait: "Clikbait",
    conspiracy: "Konšpirácia",
    false_news: "Falošné správy",
    propaganda: "Propaganda",
    satire: "Satira",
    misleading: "Zavádzajúce",
    biased: "Zaujaté",
    legitimate: "Dôveryhodné"
  };
  if (!selectedText) {
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icon.png",
      title: "MediaLens – chyba",
      message: "Najprv označte text, ktorý chcete analyzovať."
    });
    return;
  }

  fetch("http://127.0.0.1:5000/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: selectedText })
  })
    .then(res => {
      if (!res.ok) {
        throw new Error(`API odpovedalo chybou (${res.status})`);
      }
      return res.json();
    })
    .then(data => {
      const title = data.error ? "MediaLens – chyba" : "MediaLens – výsledok";
      const label = slovakLabels[data.label] || data.label;
      const message = data.error
        ? String(data.error)
        : `Kategória: ${label}\nIstota: ${(data.confidence * 100).toFixed(1)} %`;

      chrome.notifications.create({
        type: "basic",
        iconUrl: "icon.png",
        title,
        message
      });
    })
    .catch(err => {
      chrome.notifications.create({
        type: "basic",
        iconUrl: "icon.png",
        title: "MediaLens – chyba",
        message: "Nepodarilo sa spojiť so službou na 127.0.0.1:5000"
      });
      console.error(err);
    });
});
