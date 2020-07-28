# Image to Latex Conversion

An application of image captioning is to convert the the equation present in the image to latex format. Basic Sequence-to-Sequence models is used. CNN is used as encoder and RNN as decoder. Im2latex dataset is used. 

## Setup

```code
pip install -r requirements.txt
```

## Running the application

Make sure the **`model.ckpt`** and **`vocab.pkl`** are present in the `model` folder.

```
streamlit run app.py
```

![utt_gen](../../../../assets/images/applications/generation/latex_app.png)
