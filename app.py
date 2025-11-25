# app.py


if os.path.exists(CLASSES_PATH):
with open(CLASSES_PATH, 'r') as f:
d = json.load(f)
# d maps class_name -> index; invert
class_idx_to_name = {int(v): k for k, v in d.items()}
print('Classes:', class_idx_to_name)
else:
class_idx_to_name = None




def prepare_image(pil_img):
pil_img = pil_img.convert('RGB')
pil_img = pil_img.resize(IMG_SIZE)
arr = np.array(pil_img).astype('float32') / 255.0
arr = np.expand_dims(arr, axis=0)
return arr




@app.route('/')
def index():
return render_template('index.html')




@app.route('/upload', methods=['POST'])
def upload():
if 'file' not in request.files and 'image' not in request.form:
return redirect(url_for('index'))


# If webcam image sent as base64 in form field 'image'
if 'image' in request.form and request.form['image']:
data_url = request.form['image']
header, encoded = data_url.split(',', 1)
data = base64.b64decode(encoded)
pil_img = Image.open(io.BytesIO(data))
else:
f = request.files['file']
pil_img = Image.open(f.stream)


x = prepare_image(pil_img)
if model is None or class_idx_to_name is None:
return render_template('result.html', error='Model or classes not found. Train the model first.')


preds = model.predict(x)[0]
idx = int(np.argmax(preds))
prob = float(preds[idx])
label = class_idx_to_name.get(idx, str(idx))


return render_template('result.html', label=label, probability=round(prob, 4))




if __name__ == '__main__':
load_resources()
app.run(host='0.0.0.0', port=5000, debug=True)