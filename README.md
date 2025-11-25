Create a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate # (or venv\Scripts\activate on Windows)
pip install -r requirements.txt


2. Prepare dataset:
dataset/
├── jersey/
├── sahiwal/
└── indian_humped/
Place images inside each folder.


3. Train model:
python train.py
After training, models/model.h5 and models/classes.json will be created.


4. Run app:
python app.py
Open http://localhost:5000 in your browser. Use file upload or camera capture.