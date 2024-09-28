from datasets.data import extract_preprocessed
from preprocess import preprocess
from train_model import train, log_metadata


def run(version=None):
    if version == None:
        version = "v2"
    
    train_df = extract_preprocessed(name="train", version=version)
    test_df = extract_preprocessed(name="test", version=version)
    print("Train dataset: ", train_df.info(), version)
    print("Test dataset: ", test_df.info(), version)

    
    X_train, y_train, X_test, y_test = preprocess(train_df, test_df)


    model = train(X_train, y_train)
    
    log_metadata(model, X_test, y_test)


def main():
    run()


if __name__ == "__main__":
    main()
