from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline
import spacy
import pytextrank

# Load tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')

text = """
As a successful deep model applied in image super-resolution (SR),
the Super-Resolution Convolutional Neural Network (SRCNN) [1,2] has demonstrated
superior performance to the previous hand-crafted models either in speed
and restoration quality. However, the high computational cost still hinders it from
practical usage that demands real-time performance (24 fps). In this paper, we aim
at accelerating the current SRCNN, and propose a compact hourglass-shape CNN
structure for faster and better SR. We re-design the SRCNN structure mainly in
three aspects. First, we introduce a deconvolution layer at the end of the network,
then the mapping is learned directly from the original low-resolution image
(without interpolation) to the high-resolution one. Second, we reformulate
the mapping layer by shrinking the input feature dimension before mapping and
expanding back afterwards. Third, we adopt smaller filter sizes but more mapping
layers. The proposed model achieves a speed up of more than 40 times with
even superior restoration quality. Further, we present the parameter settings that
can achieve real-time performance on a generic CPU while still maintaining good
performance. A corresponding transfer strategy is also proposed for fast training
and testing across different upscaling factors.
"""


# region Abstractive Summarization
def AbSummarization(abs_text):
    # Tokenize the text
    tokens = tokenizer(abs_text, truncation=True, padding='longest', return_tensors='pt')

    print(tokens)

    # Summarize. ** is used to unpack the contents
    summary = model.generate(**tokens)

    print(summary)

    # Convert from token to string
    tokenizer.decode(summary[0])


def AbSummarizationAlt(alt_text):
    # Another method for abstractive summarization
    summarizer = pipeline('summarization', model='google/pegasus-xsum', tokenizer=tokenizer, framework='pt')

    summary = summarizer(text)

    print(summary[0]['summary_text'])


# endregion

# region Extraction Summarization
# Run python3 -m spacy download en_core_web_lg

ext_txt = """
    As a successful deep model applied in image super-resolution (SR),
    the Super-Resolution Convolutional Neural Network (SRCNN) [1,2] has demonstrated
    superior performance to the previous hand-crafted models either in speed
    and restoration quality. However, the high computational cost still hinders it from
    practical usage that demands real-time performance (24 fps). In this paper, we aim
    at accelerating the current SRCNN, and propose a compact hourglass-shape CNN
    structure for faster and better SR. We re-design the SRCNN structure mainly in
    three aspects. First, we introduce a deconvolution layer at the end of the network,
    then the mapping is learned directly from the original low-resolution image
    (without interpolation) to the high-resolution one. Second, we reformulate
    the mapping layer by shrinking the input feature dimension before mapping and
    expanding back afterwards. Third, we adopt smaller filter sizes but more mapping
    layers. The proposed model achieves a speed up of more than 40 times with
    even superior restoration quality. Further, we present the parameter settings that
    can achieve real-time performance on a generic CPU while still maintaining good
    performance. A corresponding transfer strategy is also proposed for fast training
    and testing across different upscaling factors.
    """


def ExtSummarization(ext_text):
    nlp = spacy.load('en_core_web_lg')

    nlp.add_pipe('textrank')

    doc = nlp(ext_text)

    for item in doc._.textrank.summary(limit_sentences=3):
        print(item)
# endregion
