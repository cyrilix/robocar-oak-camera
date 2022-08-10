
class HostSeqSync:
    def __init__(self):
        self.imfFrames = []

    def add_msg(self, msg):
        self.imfFrames.append(msg)

    def get_msg(self, target_seq):
        for i, imgFrame in enumerate(self.imfFrames):
            if target_seq == imgFrame.getSequenceNum():
                self.imfFrames = self.imfFrames[i:]
                break
        return self.imfFrames[0]


class CTCCodec(object):
    """ Convert between text-label and text-index """

    def __init__(self, characters):
        # characters (str): set of the possible characters.
        dict_character = list(characters)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        self.characters = dict_character
        # print(self.characters)
        # input()

    def decode(self, preds):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        # Select max probabilty (greedy decoding) then decode index to character
        preds = preds.astype(np.float16)
        preds_index = np.argmax(preds, 2)
        preds_index = preds_index.transpose(1, 0)
        preds_index_reshape = preds_index.reshape(-1)
        preds_sizes = np.array([preds_index.shape[1]] * preds_index.shape[0])

        for l in preds_sizes:
            t = preds_index_reshape[index:index + l]

            # NOTE: t might be zero size
            if t.shape[0] == 0:
                continue

            char_list = []
            for i in range(l):
                # removing repeated characters and blank.
                if not (i > 0 and t[i - 1] == t[i]):
                    if self.characters[t[i]] != '#':
                        char_list.append(self.characters[t[i]])
            text = ''.join(char_list)
            texts.append(text)

            index += l

        return texts
