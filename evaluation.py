class Evaluator:

    def conf_matrix(self, yt, yp, classes=(0, 1, 2)):

        if len(yt) != len(yp):
            raise ValueError(
                "Actual and predicted labels must have the same length."
            )

        class_to_idx = {
            cls: idx for idx, cls in enumerate(classes)
        }

        nclass = len(classes)

        cm = [
            [0 for _ in range(nclass)]
            for _ in range(nclass)
        ]

        corrects = 0
        total = len(yt)

        for actual, predicted in zip(yt, yp):

            if actual not in class_to_idx or predicted not in class_to_idx:
                raise ValueError(
                    "Unexpected class label during evaluation."
                )

            cm[class_to_idx[actual]][class_to_idx[predicted]] += 1

            if actual == predicted:
                corrects += 1

        accuracy = corrects / total if total > 0 else 0.0

        return cm, accuracy