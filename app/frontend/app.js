var map = L.map('map').setView([18.0, 99.0], 10);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18
}).addTo(map);

map.on('click', function(e){
  fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({lat: e.latlng.lat, lon: e.latlng.lng})
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById("output").innerHTML =
      `<b>Bands (Satellite *100):</b> ${JSON.stringify(data.bands_satellite)}<br>
       <b>Bands (Ground Predicted):</b> ${JSON.stringify(data.bands_ground)}<br>
       <b>Indices:</b> ${JSON.stringify(data.indices)}<br>
       <b>AGB:</b> ${data.AGB.toFixed(2)} t/ha`;
  });
});