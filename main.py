from clf_main import main_clf
from reg_main import main_reg


if __name__ == "__main__":
    print("Starting benchmarking for classification and regression...")
    main_clf()
    main_reg()
    print("Benchmarking completed for both classification and regression.")