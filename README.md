# Multi-lingual Neural Machine Translation Model

This repository contains code for training a multi-lingual neural machine translation model using Keras and TensorFlow. The model is designed to translate sentences from English to Danish and vice versa.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- nltk
- numpy
- pandas

## Dataset

The training data is stored in the `data` directory. You need to provide two text files: `english.txt` and `danish.txt`, containing English and Danish sentences respectively. Each sentence should be on a separate line.

## Usage

1. Clone the repository:
git clone https://github.com/your-username/multi-lingual-nmt-model.git

2. Install the required dependencies:

3. Modify the file paths in the script:
In the `load_data()` function, update the file paths to point to your `english.txt` and `danish.txt` files.

4. Run the script:
python main.py

This will start the training process. The model will be saved as `english_danish.h5` upon completion.


## Model Architecture

The neural machine translation model uses a sequence-to-sequence architecture with LSTM layers. The model consists of an encoder and a decoder, both using multiple LSTM layers. The input sentences are tokenized and converted to integer sequences for training.

## Results

After training the model, you can evaluate its performance using various metrics such as BLEU score or by translating new sentences. Feel free to modify the script to suit your specific requirements.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
