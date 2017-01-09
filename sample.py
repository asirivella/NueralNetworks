import random,sys

with open("winequality-red.csv") as data:
    with open("red.tra", 'w') as test:
        with open("winequality-red.tes", 'w') as train:

            header = next(data)
            test.write(header)
            train.write(header)
            for line in data:
                if random.random() > float(sys.argv[1])/100:
                    train.write(line)
                else:
                    test.write(line)