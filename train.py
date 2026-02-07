from DatasetManager.Preprocess import Preprocess


def main():

    features = ["NumberStrides", "Duration", "AverageCadence", "AverageStrideSpeed", "AverageStrideLength"]

    pre_proc = Preprocess(features)
    data = pre_proc.process_data(10376)

    print(data)
        


if __name__ == "__main__":
    main()
