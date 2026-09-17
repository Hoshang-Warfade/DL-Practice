\# Car Detection with YOLO



Practice implementation of object detection using a pretrained YOLO model, done as part of Andrew Ng's \[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) (Convolutional Neural Networks course) on Coursera. Detects and draws bounding boxes around cars (and other COCO classes) in an input image.



> This is a personal practice/learning repo based on a Coursera assignment. It's shared here for my own reference and learning history, not as an original project.



\## Project Structure



```

.

├── model\_data/              # Pretrained YOLO weights (not included — see Notes)

├── images/                  # Input images for prediction (not included — see Notes)

├── out/                     # Output images with drawn bounding boxes (generated)

├── font/                    # Font used for labeling boxes (FiraMono-Medium.otf)

├── yad2k/

│   ├── models/

│   │   └── keras\_yolo.py    # YOLO model architecture, yolo\_head, yolo\_eval

│   └── utils/

│       └── utils.py         # preprocess\_image, draw\_boxes, helper functions

└── notebook.ipynb           # Main notebook: load model, run predictions

```



\## Setup



\### 1. Environment



This project depends on an \*\*older TensorFlow/Keras model format\*\* (TF SavedModel with a custom Lambda layer), which is incompatible with plain Keras 3 (the default in TensorFlow ≥ 2.16). You'll need the legacy Keras compatibility package:



```bash

pip install tensorflow tf-keras numpy pandas matplotlib pillow scipy --break-system-packages

```



> Using a virtual environment is recommended instead of `--break-system-packages`:

> ```bash

> python -m venv venv

> source venv/bin/activate   # Windows: venv\\Scripts\\activate

> pip install tensorflow tf-keras numpy pandas matplotlib pillow scipy

> ```



\### 2. Pretrained weights and test images



The `model\_data/` (pretrained YOLO weights, \~194 MB) and sample `images/` used in this notebook were provided as part of the Coursera assignment and are \*\*not included in this repo\*\*, both due to size and because they aren't mine to redistribute. If you're working through the same course, you'll get these files from the assignment environment directly.



\### 3. Font file



Bounding box labels use `font/FiraMono-Medium.otf`. Make sure this file exists relative to wherever you run the notebook from — otherwise `draw\_boxes()` will raise `OSError: cannot open resource`.



\## Loading the Model



⚠️ \*\*Important:\*\* this model must be loaded using the \*\*legacy Keras engine\*\*, not Keras 3. Keras 3 cannot open this SavedModel format or its custom Lambda layer. Always run this as the \*\*first cell\*\*, before any other TensorFlow/Keras import:



```python

import os

os.environ\["TF\_USE\_LEGACY\_KERAS"] = "1"   # must be set BEFORE importing tensorflow



import tensorflow as tf

from tensorflow import keras



yolo\_model = keras.models.load\_model("model\_data/", compile=False)

yolo\_model.summary()

```



If `tensorflow` gets imported anywhere first (even indirectly), the override won't take effect and you'll see:

```

ValueError: File format not supported: filepath=model\_data/. Keras 3 only supports V3 `.keras` files...

```

\*\*Fix:\*\* fully restart the kernel and re-run the loading cell first.



\## Running a Prediction



```python

out\_scores, out\_boxes, out\_classes = predict("test.jpg")

```



This will:

1\. Preprocess the image to `(608, 608)`.

2\. Run it through the YOLO model.

3\. Filter/decode raw output via `yolo\_head` and `yolo\_eval`.

4\. Draw bounding boxes with class labels and confidence scores.

5\. Save the result to `out/test.jpg` and display it inline.



\## Known Compatibility Notes



I originally worked through this assignment on Coursera's environment (older Python/TF/Keras versions). Re-running it locally on a newer setup (Python 3.13, current TensorFlow) surfaced several compatibility issues — documented here in case it helps future-me or anyone else hitting the same errors.



| Issue | Cause | Fix |

|---|---|---|

| `load\_model()` rejects `model\_data/` | Keras 3 dropped support for loading SavedModel folders directly | Use `TF\_USE\_LEGACY\_KERAS=1` with `tf-keras` installed |

| Converting to `.keras` breaks the model (`EOFError` on load) | The model's custom `space\_to\_depth\_x2` Lambda layer stores pickled bytecode, which isn't portable across Python versions | Don't convert — load `model\_data/` directly every time with the legacy override |

| `Requested deserialization of a Lambda layer...` | Safety check against executing arbitrary pickled code | Only relevant if loading a `.keras` file; not needed when loading `model\_data/` directly |

| `imghdr` import error / deprecation | `imghdr` was removed in Python 3.13 | Removed from `utils.py`; use `Image.open(path).format` instead if format detection is needed |

| `draw.textsize()` fails | Removed in Pillow ≥ 10 | Replaced with `draw.textbbox()` |

| `OSError: cannot open resource` (font) | Relative path to font file doesn't match notebook's working directory | Ensure `font/FiraMono-Medium.otf` exists relative to where the notebook runs, or fall back to `ImageFont.load\_default()` |



\## Requirements



\- Python 3.13 (tested)

\- TensorFlow (latest)

\- `tf-keras` (legacy Keras 2 compatibility)

\- NumPy, Pandas, Matplotlib, Pillow, SciPy



\## Credits



\- Assignment and starter code: Andrew Ng's \[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), Convolutional Neural Networks course, Coursera.

\- Model architecture based on \[YAD2K](https://github.com/allanzelener/YAD2K) (Yet Another Darknet 2 Keras), an implementation of YOLO in Keras/TensorFlow.

