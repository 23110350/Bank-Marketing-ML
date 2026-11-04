(function () {
  const modelSelect = document.getElementById("modelSelect");
  const slider = document.getElementById("thresholdSlider");
  const status = document.getElementById("explorerStatus");
  if (!modelSelect || !slider || !status) return;
  let predictions = null;
  const ids = ["precisionValue", "recallValue", "f1Value", "accuracyValue", "tnValue", "fpValue", "fnValue", "tpValue"];
  const el = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
  const fmt = x => Number.isFinite(x) ? x.toFixed(3) : "0.000";
  function update() {
    if (!predictions) return;
    const name = modelSelect.value;
    const item = predictions[name];
    if (!item) { status.textContent = "Predictions are unavailable for this model."; return; }
    const threshold = Number(slider.value);
    const y = item.y_true, p = item.probability;
    let tn=0, fp=0, fn=0, tp=0;
    for (let i=0; i<y.length; i++) {
      const pred = p[i] >= threshold ? 1 : 0;
      if (y[i]===1 && pred===1) tp++;
      else if (y[i]===1 && pred===0) fn++;
      else if (y[i]===0 && pred===1) fp++;
      else tn++;
    }
    const precision = tp+fp ? tp/(tp+fp) : 0;
    const recall = tp+fn ? tp/(tp+fn) : 0;
    const f1 = precision+recall ? 2*precision*recall/(precision+recall) : 0;
    const accuracy = (tp+tn)/(tp+tn+fp+fn);
    document.getElementById("thresholdValue").textContent = threshold.toFixed(2);
    el.precisionValue.textContent=fmt(precision); el.recallValue.textContent=fmt(recall);
    el.f1Value.textContent=fmt(f1); el.accuracyValue.textContent=fmt(accuracy);
    el.tnValue.textContent=tn.toLocaleString(); el.fpValue.textContent=fp.toLocaleString();
    el.fnValue.textContent=fn.toLocaleString(); el.tpValue.textContent=tp.toLocaleString();
    status.textContent = `${name}: ${y.length.toLocaleString()} out-of-fold training observations shown for threshold sensitivity analysis; threshold ${threshold.toFixed(2)}.`;
  }
  fetch("data/validation_oof_predictions.json").then(r => { if (!r.ok) throw new Error("Could not load out-of-fold validation predictions."); return r.json(); })
    .then(data => { predictions=data; status.className=""; slider.addEventListener("input", update); modelSelect.addEventListener("change", update); update(); })
    .catch(err => { status.textContent = err.message + " Check that data/validation_oof_predictions.json is included in the published website."; });
})();
