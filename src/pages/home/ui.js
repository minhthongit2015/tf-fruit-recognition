/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs';

const CONTROLS = ['up', 'down', 'left', 'right'];
const CONTROL_CODES = [38, 40, 37, 39];

let resultText;
let resultPreview;
let trainStatusElement;

export function init(resultTextSelector, resultPreviewSelector, trainStatusSelector) {
  // document.getElementById('controller').style.display = '';
  // statusElement.style.display = 'none';

  resultText = document.querySelector(resultTextSelector);
  resultPreview = document.querySelector(resultPreviewSelector);
  trainStatusElement = document.querySelector(trainStatusSelector);
}

// Set hyper params from UI values.
export let learningRate = 0.0001;
export let batchSizeFraction = 0.4;
export let epochs = 20;
export let denseUnits = 100;

export function predictClass(classId, datasets) {
  if (resultText) resultText.textContent = datasets[classId].name;
  else console.log("Predict Result: ", classId);
  
  if (resultPreview) resultPreview.src = `assets/dataset/${datasets[classId].name}-1.jpg`;
  else console.log("Predict Result Preview: ", resultPreview.src);
}

export function isPredicting() {
  // statusElement.style.visibility = 'visible';
}
export function donePredicting() {
  // statusElement.style.visibility = 'hidden';
}
export function trainStatus(status) {
  // trainStatusElement = trainStatusElement || document.getElementById('train-status');
  // var trainStatus = document.getElementById('train-status');
  if (trainStatusElement) trainStatusElement.innerText = status;
  else alert(status);
}

export let addExampleHandler;
export function setExampleHandler(handler) {
  addExampleHandler = handler;
}
let mouseDown = false;
const totals = [0, 0, 0, 0];


const thumbDisplayed = {};

async function handler(label) {
  mouseDown = true;
  const className = CONTROLS[label];
  const total = document.getElementById(className + '-total');
  while (mouseDown) {
    addExampleHandler(label);
    document.body.setAttribute('data-active', CONTROLS[label]);
    total.innerText = totals[label]++;
    await tf.nextFrame();
  }
  document.body.removeAttribute('data-active');
}

// upButton.addEventListener('mousedown', () => handler(0));
// upButton.addEventListener('mouseup', () => mouseDown = false);

// downButton.addEventListener('mousedown', () => handler(1));
// downButton.addEventListener('mouseup', () => mouseDown = false);

// leftButton.addEventListener('mousedown', () => handler(2));
// leftButton.addEventListener('mouseup', () => mouseDown = false);

// rightButton.addEventListener('mousedown', () => handler(3));
// rightButton.addEventListener('mouseup', () => mouseDown = false);

export function drawThumb(img, label) {
  if (thumbDisplayed[label] == null) {
    const thumbCanvas = document.getElementById(CONTROLS[label] + '-thumb');
    draw(img, thumbCanvas);
  }
}

export function draw(image, canvas) {
  const [width, height] = [224, 224];
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
    imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
    imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
