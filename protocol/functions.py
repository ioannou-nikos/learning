# -*- coding:utf-8 -*-

"""
Σε αυτό το αρχείο θα γράψουμε όλες τις συναρτήσεις που θεωρώ ότι θα χρειαστούμε
για την προετοιμασία των δεδομένων και την επεξεργασία των κειμένων.
"""
import re


def read_file(fullName):
    """
    Συνάρτηση για την ανάγνωση του αρχείου. Θα πρέπει να επιστρέφει ένα
    Dataframe το οποίο θα αποτελείται από συγκεκριμένες στήλες που θα είναι
    χρήσιμες για την επεξεργασία.
    Δύο από τις στήλες θα είναι ο τίτλος (title) και η περιγραφή (description)
    οι οποίες θα υποστούν και την περαιτέρω λεκτική επεξεργασία προκειμένου
    να δημιουργηθούν οι λίστες των λέξεων από τις οποίες θα προχωρήσουμε στη
    συνέχεια στην κατανομή των εγγράφων.
    """
    pass


def make_words(column):
    """
    Στη συνάρτηση περνάμε σαν παράμετρο μια στήλη από το dataframe (τίτλος,
    περιγραφή) και την επεξεργαζομαστε γραμμή γραμμή έτσι ώστε να επιστρέψουμε
    ένα dictionary με τις λέξεις που θα επεξεργαστούμε.
    """
    pass


def match_word(w1, w2):
    """
    Εδώ υποτίθεται ότι συμβαίνει όλη η μαγεία γιατί εδώ είναι η συνάρτηση
    που συγκρίνει τις λέξεις μεταξύ τους.
    """
    pass


def word_list(w, w_list):
    pass


# 1. TEXT PREPROCESSING
# 1.1 Noise Removal
def _remove_noise(input_text, noise_list):
    """
    Η συνάρτηση απομακρύνει από το κείμενο input_text τις λέξεις που
    εμπεριέχονται στη λίστα noise_list και επιστρέφει μόνο τη λίστα με
    τις λέξεις που δεν αποτελούν θόρυβο.
    """
    words = input_text.split()
    noise_free_words = [word for word in words if word not in noise_list]
    return noise_free_words


def _words_as_text(word_list, connector=" "):
    """
    Η συνάρτηση παίρνει σαν παράμετρο μια λίστα από λέξεις και επιστρέφει
    ένα κείμενο συνδεδεμένο με το όρισμα connector που έχει το κενό σαν
    εξ ορισμού τιμή.
    """
    return connector.join(word_list)


def _remove_regex(input_text, regex_pattern):
    """
    Η συνάρτηση απομακρύνει μια συγκεκριμένη τυπική έκφραση από το κείμενο
    το οποίο και το επιστρέφει καθαρό.
    """
    urls = re.finditer(regex_pattern, input_text)
    for i in urls:
        input_text = re.sub(i.group().stip(), '', input_text)
    return input_text

# 1.1.1 Use of Word word list
# 1.1.2 Use of Regular Expressions
# 1.2 Lexicon Normalization
# 1.2.1 Stemming
# 1.2.2 Lemmatization
# 1.3 Object Standardization


if __name__ == "__main__":
    from nltk.corpus import wordnet as wn
    text = "Το μικρο πουλί ΕΙΝΑΙ επανω στο Δένδρο"
    wlist = _remove_noise(text, [])
    di = {}
    for w in wlist:
        l = len(wn.synsets(w, lang="ell"))
        if l != 0:
            di[w] = l
    print(di)
