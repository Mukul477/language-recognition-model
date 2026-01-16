from collections import Counter

def length_of_sentence(sentence):
    sentence_stored = sentence.split()
    sentence_length = len(sentence_stored)
    return sentence_length

def avg_num_char(sentence):
    words = sentence.split()
    num_words = len(words)
    if num_words == 0:
        return 0
    total_chars = sum(len(word) for word in words)
    return total_chars / num_words

def repeated_char_ratio(sentence):
    total_chars = len(sentence)
    if total_chars == 0:
        return 0
    counts = Counter(sentence)
    repeats = sum(freq-1 for freq in counts.values() if freq > 1)
    return repeats / total_chars

def sentence_features(sentence):
    words = sentence.split()
    num_words = len(words)
    avg_word_len = sum(len(w) for w in words) / num_words if num_words > 0 else 0
    total_chars = len(sentence)
    repeat_char_ratio = repeated_char_ratio(sentence)
    return [num_words, avg_word_len, total_chars, repeat_char_ratio]





       
         

    







    

    

    