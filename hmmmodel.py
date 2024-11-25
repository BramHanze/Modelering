import numpy as np

class HiddenMarkovModel:
    def __init__(self, n_components, n_features):
        """
        n_components: int, aantal tafels.
        n_features: int, aantal verschillende kleuren.
        """
        self.n_components = n_components
        self.n_features = n_features
        self.startprob_ = np.zeros(n_components)
        self.transmat_ = np.zeros((n_components, n_components))
        self.emissionprob_ = np.zeros((n_components, n_features))

    def __str__(self):
        return (f"HiddenMarkovModel(n_components={self.n_components}, "
                f"n_features={self.n_features})")

    def __repr__(self):
        return self.__str__()

    def sample(self, n_samples):
        states = []
        emissions = []

        current_state = np.random.choice(self.n_components, p=self.startprob_)
        states.append(current_state)

        emissions.append(
            np.random.choice(self.n_features, p=self.emissionprob_[current_state])
        )

        for _ in range(1, n_samples):
            current_state = np.random.choice(self.n_components, p=self.transmat_[current_state])
            states.append(current_state)
            emissions.append(
                np.random.choice(self.n_features, p=self.emissionprob_[current_state])
            )

        return np.array(emissions), np.array(states)
