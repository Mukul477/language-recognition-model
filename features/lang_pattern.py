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

def vowel_ratio(sentence):
    vowels = "aeiouAEIOU"
    total_chars = len(sentence)
    if total_chars == 0:
        return 0
    vowel_count = sum(1 for ch in sentence if ch in vowels)
    return vowel_count / total_chars

def stop_words(sentence):
    english_stopwords = [
"the","a","an","and","is","are","was","were","be","been","being",
"to","of","in","that","it","for","on","with","as","at","by","from",
"this","these","those","he","she","they","we","you","i","me","my","mine",
"your","yours","his","her","hers","our","ours","their","theirs",
"not","no","yes","but","or","if","then","so","because","while","when",
"where","who","whom","which","what","why","how","do","does","did","have","has","had"
]
    
    french_stopwords = [
"le","la","les","un","une","des","du","de","d'","et","en","à","au","aux",
"ce","cet","cette","ces","mon","ma","mes","ton","ta","tes","son","sa","ses",
"nous","vous","ils","elles","je","tu","il","elle","on",
"est","sont","était","étaient","être","avoir","faire","fait",
"que","qui","quoi","où","quand","pourquoi","comment",
"ne","pas","ni","mais","ou","donc","car","parce que"
]
    
    german_stopwords = [
"der","die","das","ein","eine","einer","eines","einem","einen",
"und","oder","aber","denn","sondern",
"zu","in","an","auf","bei","mit","von","für","über","nach","vor","aus","als",
"ich","du","er","sie","es","wir","ihr","sie",
"ist","sind","war","waren","sein","haben","hatte",
"dass","was","wer","wen","wem","welcher","welche","welches",
"nicht","kein","keine","nur","schon","noch","sehr"
]
    words = sentence.split()
    total_words = len(words)
    if total_words == 0:
        return 0, 0, 0
    eng_count = sum(1 for w in words if w.lower() in english_stopwords)
    fra_count = sum(1 for w in words if w.lower() in french_stopwords)
    ger_count = sum(1 for w in words if w.lower() in german_stopwords)
    return eng_count / total_words, fra_count / total_words, ger_count / total_words

def sentence_features(sentence):
    words = sentence.split()
    num_words = len(words)

    avg_word_len = sum(len(w) for w in words) / num_words if num_words > 0 else 0
    repeat_char_ratio = repeated_char_ratio(sentence)
    vowel_data = vowel_ratio(sentence)

    eng_stop, fra_stop, ger_stop = stop_words(sentence)

    return [
        num_words,
        avg_word_len,
        repeat_char_ratio,
        vowel_data,
        eng_stop,
        fra_stop,
        ger_stop
    ]





       
         

    







    

    

    