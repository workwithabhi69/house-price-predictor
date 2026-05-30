<<<<<<< HEAD
form = document.getElementById('predictForm');
resultDiv = document.getElementById('result');
errorDiv = document.getElementById('error');
loader = document.getElementById('loader');
button = document.getElementById('predictBtn');

form.addEventListener('submit', async function (e) {
  e.preventDefault();

  resultDiv.style.display = 'none';
  errorDiv.textContent = '';
  loader.style.display = 'block';
  button.disabled = true;

  const data = {
    Area: parseFloat(document.getElementById('area').value),
    Bedrooms: parseInt(document.getElementById('bedrooms').value),
    Bathrooms: parseInt(document.getElementById('bathrooms').value),
    Floors: parseInt(document.getElementById('floors').value),
    YearBuilt: parseInt(document.getElementById('yearbuilt').value),
    Location: document.getElementById('location').value,
    Condition: document.getElementById('condition').value,
    Garage: document.getElementById('garage').value
  };

  try {
    const res = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    const json = await res.json();

    if (res.ok) {
      loader.style.display = 'none';
      button.disabled = false;

      resultDiv.style.display = 'block';
      resultDiv.textContent = `Estimated House Price: ${json.prediction}`;
    } else {
      throw new Error(json.error);
    }
  } catch (err) {
    loader.style.display = 'none';
    button.disabled = false;
    errorDiv.textContent = err.message || 'Error connecting to server.';
  }
});
=======
form = document.getElementById('predictForm');
resultDiv = document.getElementById('result');
errorDiv = document.getElementById('error');
loader = document.getElementById('loader');
button = document.getElementById('predictBtn');

form.addEventListener('submit', async function (e) {
  e.preventDefault();

  resultDiv.style.display = 'none';
  errorDiv.textContent = '';
  loader.style.display = 'block';
  button.disabled = true;

  const data = {
    Area: parseFloat(document.getElementById('area').value),
    Bedrooms: parseInt(document.getElementById('bedrooms').value),
    Bathrooms: parseInt(document.getElementById('bathrooms').value),
    Floors: parseInt(document.getElementById('floors').value),
    YearBuilt: parseInt(document.getElementById('yearbuilt').value),
    Location: document.getElementById('location').value,
    Condition: document.getElementById('condition').value,
    Garage: document.getElementById('garage').value
  };

  try {
    const res = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });

    const json = await res.json();

    if (res.ok) {
      loader.style.display = 'none';
      button.disabled = false;

      resultDiv.style.display = 'block';
      resultDiv.textContent = `Estimated House Price: ${json.prediction}`;
    } else {
      throw new Error(json.error);
    }
  } catch (err) {
    loader.style.display = 'none';
    button.disabled = false;
    errorDiv.textContent = err.message || 'Error connecting to server.';
  }
});
>>>>>>> ecd2909c32b6a754e02c050b36674d3e8499487e
