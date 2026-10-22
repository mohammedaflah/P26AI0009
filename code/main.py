import os
import time
from dataset import prepare_data
from q1_bagging import run_q1
from q2_boosting import run_q2
from q3_bias_variance import run_q3

def main():
    t0 = time.time()
    os.makedirs("results", exist_ok=True)

    print("Preparing dataset...")
    prepare_data()

    print("\nRunning Question 1...")
    run_q1()

    print("\nRunning Question 2...")
    run_q2()

    print("\nRunning Question 3...")
    run_q3()

    print(f"\nDone. Completed in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()