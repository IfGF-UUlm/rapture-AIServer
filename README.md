# rapture-AIServer

> Is a scientist not entitled to the perks of AI?

\
rapture-AIServer is a lightweight LAN server providing NLP services for scientists. \
The server was designed to run on high-end retail hardware (e.g., 1 NVIDIA RTX A6000 GPU). It can do extractive and abstractive summarization, paraphrasing and academic writing. It is accessed via a browser.

## Install
rapture-AIServer can only be installed on a Linux system and requires *conda* for package, dependency and environment management. At least 40 GB of VRAM are required for running the server.
For a quick installation, clone the repository and run the install-script

`git clone https://github.com/IfGF-UUlm/rapture-AIServer.git`

`bash install.sh`

The script creates a folder "rapture-AIServer" in your home directory, with a virtual enviroment where the required libraries are installed. To run the server, activate the enviroment and run the server.py script from within the folder.

`cd ~/rapture-AIServer`

`conda activate ./env`

`python3 server.py`

## Services
#### Extractive Summary
Extracts important sentences from the input and returns them as a summary. Multilingual. \
Uses the SBertSummarizer library and the ['paraphrase-multilingual-mpnet-base-v2'](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) model for extractive summarization. A k-means clustering algorithm is applied to the sentence embeddings. The optimal value for k is determined using an automated elbow method.

#### Abstractive Summary
Summarizes the input into one sentence. Uses the fine-tuned pegasus model ['google/pegasus-xsum'](https://huggingface.co/google/pegasus-xsum) that is optimized for extreme summarization.

#### Paraphrase
Returns 5 possible paraphrases of the input. Uses the fine-tuned pegasus model ['tuner007/pegasus_paraphrase'](https://huggingface.co/tuner007/pegasus_paraphrase) that is optimized for paraphrasing.

#### Citations
Returns 5 possible references that could be cited for the input. Uses the 30 billion parameter galactica model ['facebook/galactica-30b'](https://huggingface.co/facebook/galactica-30b).

#### Introduction
Returns a possible introduction to a research paper, using the input as context information. Uses the 30 billion parameter galactica model ['facebook/galactica-30b'](https://huggingface.co/facebook/galactica-30b).

#### Conclusion
Returns a possible conclusion to a research paper, using the input as context information. It is recommended to use the entire mansucript as input. Uses the fine-tuned pegasus model *'google/pegasus-xsum'* for the first sentence and the 30 billion parameter galactica model ['facebook/galactica-30b'](https://huggingface.co/facebook/galactica-30b) for the rest.

#### Continuation
Returns a possible continuation to the input prompt in style of a research paper. Uses the 30 billion parameter galactica model ['facebook/galactica-30b'](https://huggingface.co/facebook/galactica-30b).

## Rationale
rapture-AIServer was developed as a writing aid for scientists, especially those who are not native English speakers or are new to academic writing. Any output must be checked for truthfulness and adapted to the user's needs.

#### Model Choices
- ['paraphrase-multilingual-mpnet-base-v2'](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) can do multilingual extractive summarization. In addition, the model is applied to very long input prompts, in order to save VRAM when running [GALACTICA](https://github.com/paperswithcode/galai).
- ['google/pegasus-xsum'](https://huggingface.co/google/pegasus-xsum) gives the most accurate but very short abstractive summaries from all the pegasus models.
- ['tuner007/pegasus_paraphrase'](https://huggingface.co/tuner007/pegasus_paraphrase) is a powerful writing aid, especially for non-native speakers. The sentences offered are often grammatically elegant and easy to understand.
- ['facebook/galactica-30b'](https://huggingface.co/facebook/galactica-30b) hits the sweet spot between hardware requirements and performance. It helps find appropriate citations, find text continuations if you are stuck, and even suggests an introduction to your manuscripts.
