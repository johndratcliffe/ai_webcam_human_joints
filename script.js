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
  if (!cocoModel && !poseDetectionModel) {
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

let cocoModel
let poseDetectionModel
const MODEL_PATH = 'https://www.kaggle.com/models/google/movenet/tfJs/singlepose-lightning/1'
async function getModels () {
  cocoModel = await cocoSsd.load()
  poseDetectionModel = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true })
  demosSection.classList.remove('invisible')
}

getModels()

const children = []
function predictWebcam () {
  // Start classifying a frame in the stream.
  cocoModel.detect(video).then(function (predictions) {
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

let dots = []
async function loadAndRunModel (video, predictions) {
  const imageTensor = tf.browser.fromPixels(video)
  let cropStartPoint = [15, 170, 0]
  let cropSize = [345, 345, 3]

  // Let origin of box containing person have coordinates x,y
  let x = predictions[0].bbox[0]
  let y = predictions[0].bbox[1]

  // Let dimensions of box be width, height
  let width = predictions[0].bbox[2]
  let height = predictions[0].bbox[3]

  // Dimensions of entire webcam frame
  const vid_width = 640
  const vid_height = 480

  // Predictions from COCOSSD model may be outside of the video frame, therefore
  // correct the predictions, as they should be within the frame
  if (y < 0) {
    y = 0
  }
  if (x < 0) {
    x = 0
  }
  if (height > vid_height) {
    height = vid_height
  } else if (y + height > vid_height) {
    height = vid_height - y
  }
  if (width > vid_width) {
    width = vid_width
  } else if (x + width > vid_width) {
    width = vid_width - x
  }
  
  // Determine new x,y,width,height as input to posemodel is square
  // x,y,width,height will be the parameters of the input
  // Parameters need to be within frame otherwise issues with slicing occur
  if (width === height) {
    // Both origin and dimensions remain unchanged
    cropStartPoint = [y, x, 0]
    cropSize = [width, height, 3]
  } else if (width > height) {
    /*
      --------     --------
      |      |  -> |      |
      --------     |      |
                   --------
    */
    if (width >= vid_height) {
      y = 0
      x = x + width / 2 - 240
      height = vid_height
      width = vid_height
    } else {
      y = y - (width - height) / 2
      height = width
      if (y < 0) {
        y = 0
      } else if (y + height > vid_height) {
         y = vid_height - height
      }
    }
    cropStartPoint = [parseInt(y), parseInt(x), 0]
    cropSize = [parseInt(width), parseInt(height), 3]
  } else {
    /*
      -------    ----------
      |     |    |        |
      |     | -> |        |
      |     |    |        |
      -------    ----------
    */
    x = x - (height - width) / 2
    width = height
    if (x < 0) {
      x = 0
    } else if (x + width > vid_width) {
      x = vid_width - width
    }
    if (width === vid_height) {
      y = 0
    }
    cropStartPoint = [parseInt(y), parseInt(x), 0]
    cropSize = [parseInt(width), parseInt(height), 3]
  }

  // Create input image for model
  const croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize)
  const resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true)
  const resizedTensorInt = resizedTensor.toInt()
  const expandedTensor = tf.expandDims(resizedTensorInt)
  const tensorOutput = poseDetectionModel.predict(expandedTensor)
  const arrayOutput = await tensorOutput.array()

  // Remove previous pose
  for (let i = 0; i < dots.length; i++) {
    liveView.removeChild(dots[i])
  }
  dots.splice(0)

  // Dots to show new pose
  for (let i = 0; i < 17; i++) {
    if(arrayOutput[0][0][i][2] > 0.66) {
      const highlighter = document.createElement('div')
      highlighter.setAttribute('class', 'highlighted')
      highlighter.style = 'left: ' + (x + arrayOutput[0][0][i][1] * height - 2) + 'px; top: ' +
      (y + arrayOutput[0][0][i][0] * height - 2) + 'px; width: ' +
          4 + 'px; height: ' +
          4 + 'px;'
      liveView.appendChild(highlighter)
      dots.push(highlighter)
    }
  }

  // Clear all tensors
  imageTensor.dispose()
  resizedTensor.dispose()
  tensorOutput.dispose()
  croppedTensor.dispose()
  resizedTensorInt.dispose()
  expandedTensor.dispose()
}
