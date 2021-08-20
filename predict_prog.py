import regression

if __name__ == "__main__":
    model = regression.linear_regression()
    model.load()
    mileage = float(input("Enter mileage: "))
    print(f"Read as {mileage}")
    if mileage < 0:
        print("Negative mileage is not acceptable")
        exit(0)
    print(f"Predicted price: {model(mileage)}")
