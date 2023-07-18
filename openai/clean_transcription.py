import nltk
import webvtt
from nltk.corpus import stopwords

# Download stopwords if necessary
nltk.download("stopwords")

# Define list of filler words
filler_words = ["um", "uh", "like", "you know", "ah"]

# Define path to input and output files
input_file_path = "test_vid.vtt"
output_file_path = "test_vid_clean.txt"

clean_list = []
for caption in webvtt.read(input_file_path):
    clean_list.append(caption.text)

text_content = " ".join(clean_list)

# Tokenize text content
tokens = nltk.word_tokenize(text_content)

# Remove filler words
stop_words = set(stopwords.words("english")).union(filler_words)
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# Join filtered tokens back into a single string
filtered_text = " ".join(filtered_tokens)

# Write cleaned text to output file
with open(output_file_path, "w") as output_file:
    output_file.write(filtered_text)
