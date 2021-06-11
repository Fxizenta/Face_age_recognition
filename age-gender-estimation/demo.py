from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from src.factory import get_model
import json
import os
from flask import Flask, request, url_for, send_from_directory

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.28-3.73.hdf5)")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory; if set, images in image_dir are used instead of webcam")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    # capture video
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    '''image_dir = Path(image_dir)'''

    '''for image_path in image_dir.glob("*.*"):'''
    img = cv2.imread(str(image_dir), 1)

    if img is not None:
        h, w, _ = img.shape
        r = 640 / max(w, h)
        yield cv2.resize(img, (int(w * r), int(h * r)))


def analyze(img_dir):
    # args = get_args()
    # weight_file =
    # global json_output
    # global json_output, label, Dict
    margin = 0.4
    image_dir = img_dir

    # if not weight_file:
    weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model,
                           cache_subdir="pretrained_models",
                           file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
    model = get_model(cfg)
    model.load_weights(weight_file)

    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()

    for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # draw results
            for i, d in enumerate(detected):
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "M" if predicted_genders[i][0] < 0.5 else "F")
                draw_label(img, (d.left(), d.top()), label)

        # cv2.imshow("result", img)
        key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)
        # cv2.imwrite("G:\\java\\result.jpg",img)
        Dict = {"age": label.split(",")[0], "gender": label.split(",")[1]}
        json_output = json.dumps(Dict)
        if key == 27:  # ESC
            break
    print("secces")
    print(Dict)
    print(json_output)
    return json_output


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd() + '/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

html = '''
<!DOCTYPE html>
<script>
    /*   Magic UMD boilerplate Beginning  */
/**/ (function (root, factory) {
/**/     if (typeof define === 'function' && define.amd) {
/**/         define([], factory);
/**/     } else if (typeof module === 'object' && module.exports) {
/**/         module.exports = factory();
/**/         module.exports.default = module.exports
/**/     } else {
/**/         root.smokemachine = root.SmokeMachine = factory();
/**/   }
/**/ }(typeof self !== 'undefined' ? self : this, function () {



    var opacities = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,3,5,5,7,4,4,1,1,0,1,0,0,0,0,0,1,0,0,17,27,41,52,56,34,23,15,11,4,9,5,1,0,0,0,0,0,0,1,45,63,57,45,78,66,52,41,34,37,23,20,0,1,0,0,0,0,1,43,62,66,64,67,115,112,114,56,58,47,33,18,12,10,0,0,0,0,39,50,63,76,87,107,105,112,128,104,69,64,29,18,21,15,0,0,0,7,42,52,85,91,103,126,153,128,124,82,57,52,52,24,1,0,0,0,2,17,41,67,84,100,122,136,159,127,78,69,60,50,47,25,7,1,0,0,0,34,33,66,82,113,138,149,168,175,82,142,133,70,62,41,25,6,0,0,0,18,39,55,113,111,137,141,139,141,128,102,130,90,96,65,37,0,0,0,2,15,27,71,104,129,129,158,140,154,146,150,131,92,100,67,26,3,0,0,0,0,46,73,104,124,145,135,122,107,120,122,101,98,96,35,38,7,2,0,0,0,50,58,91,124,127,139,118,121,177,156,88,90,88,28,43,3,0,0,0,0,30,62,68,91,83,117,89,139,139,99,105,77,32,1,1,0,0,0,0,0,16,21,8,45,101,125,118,87,110,86,64,39,0,0,0,0,0,0,0,0,0,1,28,79,79,117,122,88,84,54,46,11,0,0,0,0,0,0,0,0,0,1,0,6,55,61,68,71,30,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,23,25,20,12,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,12,9,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,2,2,0,0,0,0,0,0,0,0]
    var smokeSpriteSize = 20

    var polyfillAnimFrame = window.requestAnimationFrame || window.mozRequestAnimationFrame ||
        window.webkitRequestAnimationFrame || window.msRequestAnimationFrame;

    function floatInRange(start, end){
        return start + Math.random()*(end - start)
    }

    function makeSmokeSprite(color){
        color = [100 ,149, 237]||color  
        var smokeSprite = document.createElement('canvas'),
            ctx = smokeSprite.getContext('2d'),
            data = ctx.createImageData(smokeSpriteSize, smokeSpriteSize),
            d = data.data

        for(var i=0;i<d.length;i+=4){
            d[i]=color[0]
            d[i+1]=color[1]
            d[i+2]=color[2]
            d[i+3]=opacities[i / 4]
        }

        smokeSprite.width = smokeSpriteSize
        smokeSprite.height = smokeSpriteSize

        ctx.putImageData(data,0,0)

        return smokeSprite
    }

    function createParticle(x,y,options){
        options = options || {}
        var lifetime = options.lifetime || 4000
        var particle = {
            x: x,
            y: y,
            vx: floatInRange(options.minVx || -4/100, options.maxVx || 4/100),
            startvy: floatInRange(options.minVy || -4/10, options.maxVy || -1/10),
            scale: floatInRange(options.minScale || 0, options.maxScale || 0.5),
            lifetime: floatInRange(options.minLifetime || 2000, options.maxLifetime || 8000),
            age: 0,
        }
        particle.finalScale = floatInRange(
            options.minScale || 25+particle.scale,
            options.maxScale || 30+particle.scale
        )
        particle.vy = particle.startvy
        return particle
    }

    function updateParticle(particle, deltatime){
        particle.x += particle.vx * deltatime
        particle.y += particle.vy * deltatime
        var frac = Math.sqrt(particle.age / particle.lifetime)
        particle.vy = (1-frac)*particle.startvy
        particle.age+=deltatime
        particle.scale=frac*particle.finalScale
    }

    function drawParticle(particle, smokeParticleImage, context){
        context.globalAlpha = (1-Math.abs(1-2*particle.age/particle.lifetime))/8
        var off = particle.scale*smokeSpriteSize/2
        var xmin = particle.x - off
        var xmax = xmin + off*2
        var ymin = particle.y - off
        var ymax = ymin + off*2
        context.drawImage(smokeParticleImage, xmin, ymin, xmax-xmin, ymax-ymin)
        // console.log(smokeParticleImage, xmin, ymin, xmax-xmin, ymax-ymin)
    }


    return function SmokeMachine(context, color){
        var smokeParticleImage = makeSmokeSprite(color),
            particles = [],
            preDrawCallback=function(){}

        function updateAndDrawParticles(deltatime){
            context.clearRect(0, 0, context.canvas.width, context.canvas.height);

            particles.forEach(function(p){ updateParticle(p, deltatime) })
            particles = particles.filter(function(p){ return p.age < p.lifetime })

            preDrawCallback(deltatime, particles)
            particles.forEach(function(p){ drawParticle(p, smokeParticleImage, context) })
        }

        var running = false, lastframe = performance.now()
        function frame(time){
            if(!running) return
            var dt = time-lastframe
            lastframe = time;

            updateAndDrawParticles(dt)
            polyfillAnimFrame(frame)
        }

        function addParticles(x,y,numParticles,options){
            numParticles = numParticles || 10
            if(numParticles < 1) return Math.random() <= numParticles && particles.push(createParticle(x,y,options));
            for (var i = 0; i < numParticles; i++) particles.push(createParticle(x,y,options))
        }

        return {
            step: function step(dt){
                dt = dt || 16
                console.log(dt)
                updateAndDrawParticles(dt)
            },
            start: function start(){
                running = true
                lastframe = performance.now()
                polyfillAnimFrame(frame)
            },
            setPreDrawCallback: function(f){
                preDrawCallback = f
            },
            stop: function stop(){ running = false },
            addsmoke: addParticles,
            addSmoke: addParticles,
        }
    }
}))
</script>
<style>
     body {
	    background-color:white;
	    margin: 0;
	    overflow: hidden;
     }

     canvas {
	    position: fixed;
     }

     .upload {
          width: 80px;
          height: 80px;
          cursor: pointer;
          opacity: 0;
          position: absolute; 
          top:500px;
          left: 825px;
          z-index: 9999;
     }

     .choose {
          width: 100px;
          height: 100px;
          cursor: pointer;
          opacity: 0;
          position: absolute; 
          top:500px;
          left:625px;
          z-index: 9999;
     }
     .file {
          width: 85px;
          height: 85px;
          cursor: pointer;
          position: absolute; 
          top:500px;
          left:625px;
          z-index: 9998;
     }
     .upload_ico {
          width: 80px;
          height: 80px;
          cursor: pointer;
          position: absolute; 
          top:500px;
          left: 825px;
          z-index: 9998;
     }
     .image {
         width:200px;
         height:200px;
         position:absolute;
         top:225px;
         left:672px;
         z-index:9998;
     }
     .age {
         font-family:"Trebuchet MS", Arial, Helvetica, sans-serif;
         font-size: 53px;
         color: #FF8C00;
         position: absolute;
         top: 21px;
         left: 646px;
    }
    .gender {
        font-family:"Trebuchet MS", Arial, Helvetica, sans-serif;
         font-size: 53px;
         color: #FF8C00;
         position: absolute;
         top: 80px;
         left: 646px;
    }
    .frame {
        width: 278px;
        height: 467px;
        position: absolute;
        top: 94px;
        left: 633px;
        z-index: 999;
    }
    .loading {
        display:none;
        width: 200px;
        height: 200px;
        position: absolute;
        top: 230px;
        left: 662px;
        z-index: 999;
    }
</style>
  
    <title>Upload File</title>
    <form method=post enctype=multipart/form-data>
         <input type=file class="choose" name=file id="input">
         <input type=submit class="upload" >
    </form>
    <img src="https://picgo-1304285457.cos.ap-guangzhou.myqcloud.com/images/754ac3d0b2d29dfef80497e106537de.png" class="file"></img>
    <img src="https://picgo-1304285457.cos.ap-guangzhou.myqcloud.com/images/76eb158d7b7c27cdc9872287bf0093c.png" class="upload_ico"></img>
    <img src="https://picgo-1304285457.cos.ap-guangzhou.myqcloud.com/images/%E5%8D%83%E5%BA%93%E7%BD%91_%E8%93%9D%E8%89%B2%E7%A7%91%E6%8A%80%E5%87%A0%E4%BD%95%E8%BE%B9%E6%A1%86_%E5%85%83%E7%B4%A0%E7%BC%96%E5%8F%B712557974.png" class=frame></img>
    <img id="load" src="https://picgo-1304285457.cos.ap-guangzhou.myqcloud.com/images/a8fe6c326657b037343d81022f780ea.png" class="loading"></img>
<script>
	function change_Attribute() {
	    var p= document.getElementById("load");
	    p.style.display="block";
	}
	document.getElementById("input").addEventListener("change",function () {
        console.log("change");
        change_Attribute();
    });

</script>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css?family=Lobster" rel="stylesheet">
<style type="text/css">
	html, body {
		position: absolute;
		margin: 0;
		width: 100%;
		overflow: hidden;
		height: 100%;
	}
	#hi{
		position: absolute;
		top: 40%;
		width: 100%;
		text-align: center;
	}
	#hi a {
		color: #fff;
		font-size: 80px;
		text-decoration: none;		
		font-family: Lobster;
	}

	.noselect {
		-webkit-touch-callout: none; /* iOS Safari */
    	-webkit-user-select: none; /* Safari */
     	-khtml-user-select: none; /* Konqueror HTML */
       	-moz-user-select: none; /* Firefox */
		-ms-user-select: none; /* Internet Explorer/Edge */
		user-select: none; /* Non-prefixed version, currently
					  supported by Chrome and Opera */
	}

	#gh {
		display: block;
	    position: absolute;
	    transform: rotate(45deg);
	    top: -30px;
	    right: -100px;
	    transform-origin: top left;
	    background: #38dea8;
	    padding: 10px 40px;
	    color: #fff;
	    font-size: 18px;
	    font-family: sans-serif;
	    text-decoration: none;
	    text-shadow: -1px -1px 0 #5aab00;
	    box-shadow: 0 2px 10px #0000003d;
	}
</style>



<canvas id="canvas"></canvas>

<script>
	var canvas = document.getElementById('canvas')
	var ctx = canvas.getContext('2d')
	canvas.width = innerWidth
	canvas.height = innerHeight

	var party = smokemachine(ctx, [18, 16, 54])
	party.start() // start animating
	party.setPreDrawCallback(function(dt){
		party.addSmoke(innerWidth/2, innerHeight, .5)
		canvas.width = innerWidth
		canvas.height = innerHeight
	})

	 party.addsmoke(innerWidth/2, innerHeight, 100)
	//onclick=e => {
	// 	console.log(e)
	// 	party.step()
	// }

	onmousemove = function (e) {
		var x = e.clientX
		var y = e.clientY
		var n = .5
		var t = Math.floor(Math.random() * 200) + 3800
		party.addsmoke(x, y, n, t)
	}

</script>
'''


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = '.' + url_for('uploaded_file', filename=filename)
            # print("."+file_url)
            js = json.loads(analyze(file_url))
            age = js["age"]
            gender = js["gender"]
            html + '<div>' + age + '</div>'
            return html + '<img class=image src=' + file_url + '>' + '<p class=age> Age:' + age + '</p>' + '<p class=gender> Gender:' + gender + '</p>'
    return html


if __name__ == '__main__':
    app.run()