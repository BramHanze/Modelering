import random
import string

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.model = dict()
        self.charoccurence = dict()

    def fit(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        for i in range(len(text) - self.n):
            key = text[i:i+self.n]
            next_char = text[i+self.n]
            if key in self.model.keys():
                if next_char in self.model[key].keys():
                    self.model[key][next_char] += 1
                else:
                    self.model[key][next_char] = 1
            else:
                self.model[key] = {}
                self.model[key][next_char] = 1

        for key in self.model:
            for predicted_key in self.model[key]:
                total = sum(self.model[key].values())
                for predicted_key in self.model[key].keys():
                    self.model[key][predicted_key] /= total

        for char in list(map(chr, range(97, 123))):
            self.charoccurence[char] = text.count(char)

    def predict_proba(self, context):
        try:
            return self.model[context]
        except:
            return random.choices(population=list(self.charoccurence.keys()),weights=list(self.charoccurence.values()), k=1)

    def predict(self, seed, length):
        context = seed[-self.n:]
        result = seed
        for _ in range(length):
            try:
                result += random.choices(population=list(self.model[context].keys()),weights=list(self.model[context].values()), k=1)[0]
            except:
                print('new')
                result += random.choices(population=list(self.charoccurence.keys()),weights=list(self.charoccurence.values()), k=1)[0]
            context = result[-self.n:]
        return result


seed = 'kanker in'


model = NGramModel(len(seed))
#text = 'weten welke wet geldt'
text = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'
text = ''
with open('CasusB/LLM.txt', 'r') as file:
    for line in file:
        text += line.strip()
model.fit(text.lower())

#print("Model:", model.model)

print("Generated text:", model.predict(seed, 400))

