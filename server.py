from flask import Flask, render_template, request
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, OPTForCausalLM, AutoTokenizer
from summarizer.sbert import SBertSummarizer
import warnings

warnings.filterwarnings("ignore")
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Multilingual Extractive Summarizer
extractive_summarizer = SBertSummarizer(
    'paraphrase-multilingual-mpnet-base-v2')


def get_extractive_summary(input_text):
    k = extractive_summarizer.calculate_optimal_k(
        input_text, min_length=50, k_max=10)
    return extractive_summarizer(input_text, min_length=50, num_sentences=k)


def summarize_input(input_text, num_sentences=10):  # Save VRAM on long LLM inputs
    if len(input_text) > 2048:
        input_text = extractive_summarizer(
            input_text, min_length=50, num_sentences=num_sentences)
    return input_text


# Abstractive Summarizer
abstractive_summarizer = PegasusForConditionalGeneration.from_pretrained(
    "google/pegasus-xsum").to(torch_device)
abstractive_summarizer_tokenizer = AutoTokenizer.from_pretrained(
    "google/pegasus-xsum")


def get_abstractive_summary(input_text):
    inputs = abstractive_summarizer_tokenizer(
        input_text, truncation=True, padding=True, return_tensors="pt")
    summary_ids = abstractive_summarizer.generate(
        inputs["input_ids"].to(torch_device))
    return abstractive_summarizer_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# Paraphraser
pegasus = PegasusForConditionalGeneration.from_pretrained(
    'tuner007/pegasus_paraphrase').to(torch_device)
pegasus_tokenizer = PegasusTokenizer.from_pretrained(
    'tuner007/pegasus_paraphrase')


def get_paraphrases(input_text, num_return_sequences=5, num_beams=100):
    batch = pegasus_tokenizer([input_text], truncation=True, padding=True,
                              max_length=100, return_tensors="pt").to(torch_device)
    translated = pegasus.generate(
        **batch, max_length=100, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=0.0)
    paraphrases = pegasus_tokenizer.batch_decode(
        translated, skip_special_tokens=True)
    return '<br>'.join(paraphrases)


# Galactica AI
galai = OPTForCausalLM.from_pretrained(
    "facebook/galactica-30b", device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)
galai_tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-30b")


def get_citations(input_text):
    encoded_input = galai_tokenizer(
        input_text + " [START_REF]", return_tensors='pt')
    output_sequences = galai.generate(
        input_ids=encoded_input['input_ids'].to(torch_device),
        max_new_tokens=50,
        num_beams=5,
        num_return_sequences=5,
        num_beam_groups=5,
        diversity_penalty=0.9
    )
    citations = []
    for sequence in output_sequences:
        suggestion = galai_tokenizer.decode(
            sequence, skip_special_tokens=False)
        end_ref_pos = suggestion.find("[END_REF]")
        start_ref_pos = suggestion.find("[START_REF]") + 11
        if end_ref_pos == -1:
            citations.append(suggestion[start_ref_pos:].strip())
        else:
            citations.append(suggestion[start_ref_pos:end_ref_pos].strip())
    return '<br>'.join(citations)


def get_introduction(input_text, max_new_tokens=400):
    encoded_input = galai_tokenizer(
        input_text + '\n\n# Introduction\n\n', return_tensors='pt')
    output_sequences = galai.generate(
        input_ids=encoded_input['input_ids'].to(torch_device),
        max_new_tokens=max_new_tokens
    )
    introduction = galai_tokenizer.decode(
        output_sequences[0], skip_special_tokens=False)
    [introduction.replace("</s>", "").replace("<pad>", "")
     for text in introduction]
    start_ref_pos = introduction.find("# Introduction") + 16
    return introduction[start_ref_pos:]


def get_conclusion(input_text, max_new_tokens=200):
    encoded_input = galai_tokenizer(
        input_text + '\n\n# Conclusion\n\n', return_tensors='pt')
    output_sequences = galai.generate(
        input_ids=encoded_input['input_ids'].to(torch_device),
        max_new_tokens=max_new_tokens
    )
    conclusion = galai_tokenizer.decode(
        output_sequences[0], skip_special_tokens=False)
    [conclusion.replace("</s>", "").replace("<pad>", "")
     for text in conclusion]
    start_pos = conclusion.find("# Conclusion") + 14
    end_pos = conclusion.find("# References") - \
        1 if "# References" in conclusion else None
    return conclusion[start_pos:end_pos]


def get_continuation(input_text, max_new_tokens=200):
    encoded_input = galai_tokenizer(input_text, return_tensors='pt')
    output_sequences = galai.generate(
        input_ids=encoded_input['input_ids'].to(torch_device),
        max_new_tokens=max_new_tokens
    )
    continuation = galai_tokenizer.decode(
        output_sequences[0], skip_special_tokens=False)
    [continuation.replace("</s>", "").replace("<pad>", "")
     for text in continuation]
    return continuation


# Flask
app = Flask(__name__)


@app.route("/")
def index():
    return render_template('template.html', input_text='', result='')


@app.route('/process', methods=['GET', 'POST'])
def process():
    input_text = request.form['input_text'].strip()
    model_choice = request.form['model_choice']
    if model_choice == 'extractive_summary':
        result = get_extractive_summary(input_text)
    elif model_choice == 'abstractive_summary':
        result = get_abstractive_summary(input_text)
    elif model_choice == 'paraphrase':
        result = get_paraphrases(summarize_input(input_text))
    elif model_choice == 'citations':
        result = get_citations(summarize_input(input_text))
    elif model_choice == 'introduction':
        result = get_introduction(summarize_input(input_text))
    elif model_choice == 'conclusion':
        result = get_abstractive_summary(
            input_text) + ' ' + get_conclusion(summarize_input(input_text))
    elif model_choice == 'continuation':
        result = get_continuation(summarize_input(input_text),)
    else:
        result = ''
    return render_template('template.html', input_text=input_text, result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
