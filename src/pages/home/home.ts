

import { CameraPreview, CameraPreviewPictureOptions, CameraPreviewOptions } from '@ionic-native/camera-preview';

import * as tf from '@tensorflow/tfjs';
// declare var tf;

import {ControllerDataset} from './controller_dataset';
import * as ui from './ui';
import {Webcam} from './webcam';

import { Component, ViewChild } from '@angular/core';
import { NavController, Platform } from 'ionic-angular';

import { Chart } from 'chart.js';

@Component({
  selector: 'page-home',
  templateUrl: 'home.html'
})
export class HomePage {
  base64Image: any = "assets/dataset/strawberry-5.jpg";
  resultPreview: any = "";

  datasetDir = "dataset";
  datasets: any[];
  ui: any = ui;

  Algorithms = [
    { type: "loss", name: "Categorical Crossentropy" },
    { type: "optimizer", name: "Adam" }
  ]

  constructor(
    private platform: Platform,
    public navCtrl: NavController,
    private cameraPreview: CameraPreview
    ) {
    this.datasets = [
      { name: "tomato", total: 12, train: 8, test: 4 },
      { name: "strawberry", total: 12, train: 8, test: 4 },
      { name: "grape", total: 12, train: 8, test: 4 }
    ]
    // this.datasets = [
    //   { name: "cam", total: 12, train: 3, test: 4 },
    //   { name: "nguyhiem", total: 12, train: 3, test: 4 },
    //   { name: "chidan", total: 12, train: 3, test: 4 }
    // ]

    // Đánh số
    for (let dataset of this.datasets) {
      dataset.totalset = [...Array(dataset.total)].map((z, i) => i);
      for (let i=0; i<dataset.totalset.length; i++) {
        dataset.totalset[i] = i;
      }
      dataset.trainset = dataset.totalset.slice(0, dataset.train);
      dataset.testset = dataset.totalset.slice(dataset.train, dataset.train + dataset.test);
    }
    console.log(this.datasets);
    
    this.platform.ready().then(() => {
      this.cameraPreview.getSupportedPictureSizes().then(rs => {
        console.log(rs);
      }).catch((err) => {
        console.log(err);
      });
  
      const cameraPreviewOpts: CameraPreviewOptions = {
        x: 0,
        y: 0,
        width: window.screen.width,
        height: window.screen.height,
        camera: 'rear',
        tapPhoto: true,
        previewDrag: true,
        toBack: true,
        alpha: 1
      };
      
      // start camera
      this.cameraPreview.startCamera(cameraPreviewOpts).then((res) => {
          console.log("start camera succeced!", res);
          this.cameraPreview.show().catch(err => console.log(err));
        }, (err) => {
          console.log("start camera failed!", err);
        }
      );
      this.realTime();

    });
  }


  ionViewDidLoad() {
    this.TFSetup();
    this.loadLossChart();
  }



  /**************************************************************************
   *                               Setup Camera                             *
   **************************************************************************/

  pictureOpts: CameraPreviewPictureOptions = {
    width: window.screen.width,
    height: window.screen.height,
    quality: 50
  }
  cameraReady = false;  // Kiểm tra phần cứng sẵn sàng
  activeCamera = false; // Kiểm tra bật chế độ live hay không
  realTime() {
    setTimeout(async () => {
      while (true) {
        await tf.nextFrame();
        await tf.nextFrame();
        if (this.cameraReady) this.renderFrame();
      }
    }, 30);
  }
  renderFrame() {
    this.cameraReady = false;
    this.cameraPreview.takePicture(this.pictureOpts).then((imageData) => {
      this.base64Image = 'data:image/jpeg;base64,' + imageData;
      this.cameraReady = this.activeCamera && true;
    }, (err) => {
      // alert(err);
      // this.base64Image = 'assets/img/test.jpg';
      setTimeout(()=>this.renderFrame(), 1000);
    });
  }



  /**************************************************************************
   *                              UI - METHODS                              *
   **************************************************************************/
  toggleShowTab() {
    document.getElementById("tabs").classList.toggle("show");
  }

  tabId: number;
  openTab(tabId) {
    this.tabId = tabId == this.tabId ? 0 : tabId;
  }

  // Nhận diện trái cây từ [Test Set] or [Training Set]
  async recognitionFruit(e) {
    // Xóa kết quả cũ
    if (ui.resultPreview) ui.resultPreview.src = `assets/waiting.png`;

    this.isPredicting = false;
    let img = e.currentTarget.firstElementChild;
    this.base64Image = img.src;
    await tf.nextFrame();
    this.TFPredictFruit();
  }

  // Sự kiện nhấn nút Train Model
  async startTraining() {
    await tf.nextFrame();
    await tf.nextFrame();
    this.isPredicting = false;
    this.TFTrain();
  }

  // Bật chế độ nhận diện trái cây thời gian thực qua camera
  realtimeRecognition() {
    this.isPredicting = !this.isPredicting;
    this.cameraReady = true;
    this.activeCamera = this.isPredicting;
    this.livePredict();
  }

  // Sự kiện nhấn nút chụp ảnh thêm ảnh vào [Training Set]
  fruitSampleCounter:any = [0,0,0];
  addFruitSample(fruitId) {
    this.fruitSampleCounter[+fruitId]++;
    this.controllerDataset.addExample(this.mobilenet.predict(this.webcam.capture("#vid-canvas")), +fruitId);
    ui.trainStatus(`Add example: ${+fruitId} (${this.fruitSampleCounter[+fruitId]})`)
  }

  clearDataset() {
    // this.datasets
  }

  // Phương thức nhận diện trái cây và hiển thị kết quả
  async TFPredictFruit(element="") {

    // Tính toán dự đoán
    const predictedClass = tf.tidy(() => {
      const img = this.webcam.capture(element || "#canvas");
      const activation = this.mobilenet.predict(img);
      const predictions = this.model.predict(activation);
      return predictions.as1D().argMax();
    });

    // Rút lấy kết quả
    const classId = (await predictedClass.data())[0];
    predictedClass.dispose();

    // Hiển thị kết quả
    ui.predictClass(classId, this.datasets);
    await tf.nextFrame();
  }

  // Sự kiện nhấn nút Reset Model
  resetModel() {
    this.TFResetModel();
  }

  // Setup Loss Chart
  @ViewChild('lineCanvas') lineCanvas;
  lineChart: any;
  loadLossChart() {
    this.lastLoss = +localStorage["lastLost"] || "---";
    this.lineChart = new Chart(this.lineCanvas.nativeElement, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: "Loss",
            fill: false,
            lineTension: 0.1,
            backgroundColor: "rgba(75,192,192,0.4)",
            borderColor: "rgba(75,192,192,1)",
            borderCapStyle: 'butt',
            borderDash: [],
            borderDashOffset: 0.0,
            borderJoinStyle: 'miter',
            pointBorderColor: "rgba(75,192,192,1)",
            pointBackgroundColor: "#fff",
            pointBorderWidth: 1,
            pointHoverRadius: 5,
            pointHoverBackgroundColor: "rgba(75,192,192,1)",
            pointHoverBorderColor: "rgba(220,220,220,1)",
            pointHoverBorderWidth: 2,
            pointRadius: 1,
            pointHitRadius: 10,
            data: [],
            spanGaps: false,
          }
        ]
      }
    });
  }


  /**************************************************************************
   *                         TENSORFLOW - METHODS                           *
   **************************************************************************/

  NUM_CLASSES;
  webcam;
  controllerDataset;

  mobilenet;
  model;
  lastLoss;

  async TFSetup() {
    ui.init("#result-text", "#result-preview", "#statustext");


    this.NUM_CLASSES = this.datasets.length;
    this.webcam = new Webcam(document.getElementById('vid-canvas'), document.getElementById('canvas'));

    this.webcam.setup();

    // The dataset object where we will store activations.
    this.controllerDataset = new ControllerDataset(this.NUM_CLASSES);
    
    // ui.setExampleHandler(label => {
    //   tf.tidy(() => {
    //     const img = webcam.capture();
    //     controllerDataset.addExample(mobilenet.predict(img), label);
    //     ui.drawThumb(img, label);
    //   });
    // });


    // document.getElementById('train').addEventListener('click', async () => {
    //   ui.trainStatus('Training...');
    //   await tf.nextFrame();
    //   await tf.nextFrame();
    //   isPredicting = false;
    //   train();
    // });
    // document.getElementById('predict').addEventListener('click', () => {
    //   ui.startPacman();
    //   isPredicting = true;
    //   predict();
    // });
    
    try {
      await this.webcam.setup();
    } catch (err) {
      alert(err);
    }

    await this.TFLoadModel();

    // tf.tidy(() => mobilenet.predict(webcam.capture("canvas")));

  }

  async TFLoadModel() {
    ui.trainStatus("ML Model loading...");
    this.mobilenet = await this.loadMobilenetModel();
    await tf.loadModel('/assets/models/fruit-recognition-model.json').catch((err) => {
      console.log(err);
    }).then((rs) => {
      this.model = rs;
      window['model'] = this.model;
    });
    ui.trainStatus("ML Model loaded!");
  }

  async loadMobilenetModel() {
    // this.mobilenet = await tf.loadModel(
    //     'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    this.mobilenet = await tf.loadModel('assets/models/mobilenet-model.json');

    const layer = this.mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({inputs: this.mobilenet.inputs, outputs: layer.output});
  }

  baseTrainingSetLoaded = false;
  async TFLoadTrainingSet() {
    let selector = "";
    
    if (this.baseTrainingSetLoaded) return;
    this.baseTrainingSetLoaded = true;

    tf.tidy(() => {
      for (let set in this.datasets) {
        let dataset = this.datasets[set];
        console.log("[SET] : ", +set);
        for (let i of dataset.trainset) {
          selector = `#${dataset.name}-${+i+1}`;
          let img = this.webcam.capture(selector);
          console.log(selector, `set [${set}]: `, img);
          this.controllerDataset.addExample(this.mobilenet.predict(img), +set);
        }
      }
    });
  }

  // Phương thức thiết lập Model chính
  TFResetModel() {
    this.lastLoss = "---";
    localStorage["lastLost"] = this.lastLoss;
    this.lineChart.data.labels = [];
    this.lineChart.data.datasets[0].data = [];
    this.lineChart.update();
    
    this.model = tf.sequential({
      layers: [
        tf.layers.flatten({inputShape: [7, 7, 256]}),
        // Layer 1
        tf.layers.dense({
          units: ui.denseUnits,
          activation: 'relu',
          kernelInitializer: 'varianceScaling',
          useBias: true
        }),
        tf.layers.dense({
          units: this.NUM_CLASSES,
          kernelInitializer: 'varianceScaling',
          useBias: false,
          activation: 'softmax'
        })
      ]
    });
  }

  async TFTrain() {
    ui.trainStatus('Load Training Set...');
    await this.TFLoadTrainingSet();

    if (this.controllerDataset.xs == null) {
      alert('Add some examples before training!');
      return;
    }

    ui.trainStatus('Training...');
    await tf.nextFrame();
    await tf.nextFrame();
    this.isPredicting = false;

    if (!this.model) this.TFResetModel();

    const optimizer = tf.train.adam(ui.learningRate);
    this.model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

    const batchSize = Math.floor(this.controllerDataset.xs.shape[0] * ui.batchSizeFraction);
    if (!(batchSize > 0)) {
      throw new Error(
          `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
    }

    // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
    this.model.fit(this.controllerDataset.xs, this.controllerDataset.ys, {
      batchSize,
      epochs: ui.epochs,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
          this.lastLoss = logs.loss;
          localStorage["lastLost"] = this.lastLoss;

          this.lineChart.data.labels.push(this.lineChart.data.labels.length+1);
          this.lineChart.data.datasets[0].data.push(this.lastLoss);
          this.lineChart.update();

          // console.log(logs);
        },
        onTrainEnd: async (trainResult) => {
          ui.trainStatus('[Train Complete]\r\nLast Loss: ' + this.lastLoss.toFixed(5));

          console.log("Train result: ", trainResult, this.model);
          window["model"] = this.model;

          const saveResult = await this.model.save('indexeddb://fruit-recognition-model');
          console.log("Save model result: ", saveResult);
        }
      }
    });
  }
  
  isPredicting = false;
  async livePredict() {
    ui.isPredicting();
    while (this.isPredicting) {
      const predictedClass = tf.tidy(() => {
        const img = this.webcam.capture("#canvas");
        const activation = this.mobilenet.predict(img);
        const predictions = this.model.predict(activation);
        return predictions.as1D().argMax();
      });

      const classId = (await predictedClass.data())[0];
      predictedClass.dispose();

      ui.predictClass(+classId, this.datasets);
      await tf.nextFrame();
    }
    ui.donePredicting();
  }
}
