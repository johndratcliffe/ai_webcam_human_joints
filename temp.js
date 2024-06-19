const video = document.getElementById('webcam')
const liveView = document.getElementById('liveView')
const demosSection = document.getElementById('demos')
const enableWebcamButton = document.getElementById('webcamButton')

// Check if webcam access is supported.
function getUserMediaSupported () {
  return !!(navigator.mediaDevices &&
  navigator.mediaDevices.getUserMedia)
}

// If webcam supported, add event listener to button for when user
// wants to activate it to call enableCam function
if (getUserMediaSupported()) {
  enableWebcamButton.addEventListener('click', enableCam)
} else {
  console.warn('getUserMedia() is not supported by your browser')
}

// Enable the live webcam view and start classification.
function enableCam (event) {
  // Only continue if the COCO-SSD has finished loading.
  if (!model) {
    return
  }

  // Hide the button once clicked.
  event.target.classList.add('removed')

  // getUsermedia parameters to force video but not audio.
  const constraints = {
    video: true
  }

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
    video.srcObject = stream
    video.addEventListener('loadeddata', predictWebcam)
  })
}

let model
cocoSsd.load().then(function (loadedModel) {
  model = loadedModel
  // Show demo section now that model is ready to use.
  demosSection.classList.remove('invisible')
})

const children = []
function predictWebcam () {
  // Start classifying a frame in the stream.
  model.detect(video).then(function (predictions) {
    // Remove any highlighting we did previous frame.
    for (let i = 0; i < children.length; i++) {
      liveView.removeChild(children[i])
    }
    children.splice(0)

    // Loop through predictions and draw them to the live view if
    // they have a high confidence score.
    for (let n = 0; n < predictions.length; n++) {
      // If we are over 66% sure we are sure we classified it right, draw it!
      if (predictions[n].score > 0.66) {
        const p = document.createElement('p')
        p.innerText = predictions[n].class + ' - with ' +
            Math.round(parseFloat(predictions[n].score) * 100) +
            '% confidence.'
        // Coordinates of the text
        p.style = 'margin-left: ' + predictions[n].bbox[0] + 'px; margin-top: ' +
            (predictions[n].bbox[1] - 10) + 'px; width: ' +
            (predictions[n].bbox[2] - 10) + 'px; top: 0; left: 0;'

        // Coordinates and dimensions of the highlighted region
        const highlighter = document.createElement('div')
        highlighter.setAttribute('class', 'highlighter')
        highlighter.style = 'left: ' + predictions[n].bbox[0] + 'px; top: ' +
            predictions[n].bbox[1] + 'px; width: ' +
            predictions[n].bbox[2] + 'px; height: ' +
            predictions[n].bbox[3] + 'px;'

        liveView.appendChild(highlighter)
        liveView.appendChild(p)
        children.push(highlighter)
        children.push(p)
        // Pose detection model
        if (predictions[n].class === 'person') {
          loadAndRunModel(video, predictions)
        }
      }
    }

    // Call this function again to keep predicting when the browser is ready.
    window.requestAnimationFrame(predictWebcam)
  })
}

const MODEL_PATH = 'https://www.kaggle.com/models/google/movenet/tfJs/singlepose-lightning/1'

let movenet
let dots = []
async function loadAndRunModel (video, predictions) {
  movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true })
  const imageTensor = tf.browser.fromPixels(video)
  console.log(imageTensor.shape)

  // 640 x 480 w-h
  // Model uses yx --> 480.640
  /* bbox: [x, y, width, height],
  class: "person",
  score: 0.8380282521247864*/
  let cropStartPoint = [15, 170, 0]
  let cropSize = [345, 345, 3]
  let x = predictions.bbox[0]
  let y = predictions.bbox[1]
  let width = predictions.bbox[2]
  let height = predictions.bbox[3]
  if (width === height) {
    cropStartPoint = [y, x, 0]
    cropSize = [width, height, 3]
  } else if (width > height) {
    y = y - (width - height) / 2
    height = width
    if (y < 0) {
      y = 0
    }
    if (imageTensor.shape()[0] < height) {
      height = imageTensor.shape()[0]
      width = imageTensor.shape()[0]
      y = 0
    }
    cropStartPoint = [y, x, 0]
    cropSize = [width, height, 3]
  } else {
    x = x - (height - width) / 2
    width = height
    if (x < 0) {
      x = 0
    }
    if (imageTensor.shape()[1] < width) {
      height = imageTensor.shape()[0]
      width = imageTensor.shape()[0]
      x = 0
    }
    cropStartPoint = [y, x, 0]
    cropSize = [width, height, 3]
  }
  const croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize)
  const resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).toInt()
  console.log(resizedTensor.shape)

  const tensorOutput = movenet.predict(tf.expandDims(resizedTensor))
  const arrayOutput = await tensorOutput.array()
  console.log(arrayOutput)
  // draw dots
  for (let i = 0; i < 17; i++) {
    const highlighter = document.createElement('div')
    highlighter.setAttribute('class', 'highlighter')
    highlighter.style = 'left: ' + arrayOutput[0][0][i][1] * 420 + 'px; top: ' +
    arrayOutput[0][0][i][0] * 420 + 'px; width: ' +
        10 + 'px; height: ' +
        10 + 'px;'
    liveView.appendChild(highlighter)
    dots.push(highlighter)
  }
}
