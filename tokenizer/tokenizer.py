class Tokenizer():
    """
        Basic tokenizer for encoding and decoding tokens based on the unique
        characters in a text
    """

    def __init__(self, text):
        """
            Initializes the tokenizer

            Args:
                text string: the data set text
        """
        self.chars = sorted(list(set(text)))

    def encode(self, chunk):
        """
            Encodes the unique characters to a list of integers representing
            the individual characters

            Args:
                chunk string: the string to encode
        """
        stoi = { ch:i for i,ch in enumerate(self.chars) }
        return [stoi[c] for c in chunk]

    def decode(self, list):
        """
            Decodes the integer list into a string of the corresponding characters

            Args:
                list int[]: list of integers representing characters
        """
        itos = { i:ch for i,ch in enumerate(self.chars) }
        return ''.join([itos[i] for i in list])
    
    def vocab_size(self):
        return len(self.chars)
