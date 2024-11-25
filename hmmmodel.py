import numpy as np

class HiddenMarkovModel:
    def __init__(self, n_components, n_features):
        """
        Initialiseer een Hidden Markov Model.
        
        Parameters:
        - n_components: int, aantal toestanden.
        - n_features: int, aantal verschillende emissies.
        """
        self.n_components = n_components
        self.n_features = n_features
        self.startprob_ = np.zeros(n_components)  # Begintoestandsverdeling
        self.transmat_ = np.zeros((n_components, n_components))  # Overgangswaarschijnlijkheden
        self.emissionprob_ = np.zeros((n_components, n_features))  # Emissiekansen

    def __str__(self):
        return (f"HiddenMarkovModel(n_components={self.n_components}, "
                f"n_features={self.n_features})")

    def __repr__(self):
        return self.__str__()

    def sample(self, n_samples):
        """
        Genereer een sequentie van toestanden en waarnemingen.

        Parameters:
        - n_samples: int, aantal te genereren waarnemingen.

        Returns:
        - emissions: lijst van gegenereerde waarnemingen.
        - states: lijst van gegenereerde toestanden.
        """
        states = []
        emissions = []

        # Begin met de initiële toestand
        current_state = np.random.choice(self.n_components, p=self.startprob_)
        states.append(current_state)

        # Voeg de eerste emissie toe
        emissions.append(
            np.random.choice(self.n_features, p=self.emissionprob_[current_state])
        )

        # Genereer de rest
        for _ in range(1, n_samples):
            current_state = np.random.choice(self.n_components, p=self.transmat_[current_state])
            states.append(current_state)
            emissions.append(
                np.random.choice(self.n_features, p=self.emissionprob_[current_state])
            )

        return np.array(emissions), np.array(states)
