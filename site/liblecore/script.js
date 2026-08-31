(() => {
  const canvas = document.querySelector("#vector-demo");
  const similarityNode = document.querySelector("#similarity");
  const seedNode = document.querySelector("#seed");
  const reseedButton = document.querySelector("#reseed");
  const copyButton = document.querySelector("#copy-command");
  const commandNode = document.querySelector("#build-command");

  if (!canvas || !similarityNode || !seedNode || !reseedButton) return;

  const context = canvas.getContext("2d");
  const dimension = 32;
  let seed = 7;

  function randomSource(initialSeed) {
    let state = initialSeed >>> 0 || 1;
    return () => {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      return (state >>> 0) / 4294967296;
    };
  }

  function normalize(vector) {
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    return norm === 0 ? vector.slice() : vector.map((value) => value / norm);
  }

  function unitaryRole(initialSeed) {
    const random = randomSource(initialSeed);
    const spectrum = Array.from({ length: dimension }, () => ({ re: 0, im: 0 }));
    spectrum[0].re = random() < 0.5 ? -1 : 1;
    spectrum[dimension / 2].re = random() < 0.5 ? -1 : 1;

    for (let index = 1; index < dimension / 2; index += 1) {
      const phase = random() * Math.PI * 2;
      spectrum[index] = { re: Math.cos(phase), im: Math.sin(phase) };
      spectrum[dimension - index] = { re: Math.cos(phase), im: -Math.sin(phase) };
    }

    const role = [];
    for (let time = 0; time < dimension; time += 1) {
      let value = 0;
      for (let frequency = 0; frequency < dimension; frequency += 1) {
        const angle = (Math.PI * 2 * frequency * time) / dimension;
        value += spectrum[frequency].re * Math.cos(angle) - spectrum[frequency].im * Math.sin(angle);
      }
      role.push(value / dimension);
    }
    return normalize(role);
  }

  function randomVector(initialSeed) {
    const random = randomSource(initialSeed);
    return normalize(Array.from({ length: dimension }, () => random() * 2 - 1));
  }

  function convolve(left, right) {
    return left.map((_, outputIndex) => {
      let total = 0;
      for (let index = 0; index < dimension; index += 1) {
        total += left[index] * right[(outputIndex - index + dimension) % dimension];
      }
      return total;
    });
  }

  function involution(vector) {
    return vector.map((_, index) => vector[(dimension - index) % dimension]);
  }

  function cosine(left, right) {
    const leftNorm = Math.sqrt(left.reduce((sum, value) => sum + value * value, 0));
    const rightNorm = Math.sqrt(right.reduce((sum, value) => sum + value * value, 0));
    if (leftNorm === 0 || rightNorm === 0) return 0;
    const dot = left.reduce((sum, value, index) => sum + value * right[index], 0);
    return dot / (leftNorm * rightNorm);
  }

  function drawBand(vector, y, label, color) {
    const width = canvas.width;
    const left = 132;
    const right = 18;
    const bandWidth = width - left - right;
    const cellWidth = bandWidth / dimension;
    const center = y + 28;
    const amplitude = 23;

    context.fillStyle = "#8e958d";
    context.font = "16px SFMono-Regular, Consolas, monospace";
    context.fillText(label, 12, center + 5);

    context.strokeStyle = "#293029";
    context.beginPath();
    context.moveTo(left, center);
    context.lineTo(width - right, center);
    context.stroke();

    vector.forEach((value, index) => {
      const height = Math.max(1.5, Math.abs(value) * amplitude * 3.6);
      const x = left + index * cellWidth + 1;
      const top = value >= 0 ? center - height : center;
      context.globalAlpha = 0.36 + Math.min(0.64, Math.abs(value) * 2.8);
      context.fillStyle = color;
      context.fillRect(x, top, Math.max(2, cellWidth - 2), height);
    });
    context.globalAlpha = 1;
  }

  function drawArrow(y, glyph) {
    context.fillStyle = "#4b554a";
    context.font = "18px SFMono-Regular, Consolas, monospace";
    context.fillText(glyph, 63, y);
  }

  function render() {
    const role = unitaryRole(seed);
    const value = randomVector(seed + 991);
    const bound = convolve(role, value);
    const recovered = convolve(bound, involution(role));
    const similarity = cosine(value, recovered);

    context.clearRect(0, 0, canvas.width, canvas.height);
    drawBand(role, 8, "ROLE", "#6fe7ff");
    drawArrow(92, "⊛");
    drawBand(value, 98, "VALUE", "#b6ff58");
    drawArrow(182, "↓");
    drawBand(bound, 188, "BOUND", "#ff9e5d");
    drawArrow(272, "⊘");
    drawBand(normalize(recovered), 278, "RECOVERED", "#b6ff58");

    similarityNode.textContent = similarity.toFixed(6);
    seedNode.textContent = String(seed).padStart(4, "0");
  }

  reseedButton.addEventListener("click", () => {
    seed += 1;
    render();
  });

  if (copyButton && commandNode) {
    copyButton.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(commandNode.textContent);
        copyButton.textContent = "Copied";
        window.setTimeout(() => { copyButton.textContent = "Copy"; }, 1400);
      } catch {
        copyButton.textContent = "Select text";
      }
    });
  }

  render();
})();
